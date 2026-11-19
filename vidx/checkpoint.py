"""Durable checkpoints for long runs (only imported when a --checkpoint-* flag is given).

A snapshot is one generation folder `gen-NNNN/` holding the gzipped output, the done sidecar,
progress.json and, written LAST, manifest.json (sizes + sha256 of the other files, the number
of done videos and the run fingerprint). A generation without a matching manifest is ignored,
so nothing relies on atomic renames (Drive FUSE has none). Stores: a folder (e.g. a Google
Drive mount) and/or a private Hugging Face dataset repo. Standard library only; huggingface_hub
is imported lazily inside HFStore. A snapshot failure only logs a warning.
"""
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from .output import sidecar_path

log = logging.getLogger("vidx.checkpoint")
KEEP = 2  # generations kept per store
PAYLOAD = ("results.jsonl.gz", "done.txt", "progress.json")
FILES = PAYLOAD + ("manifest.json",)  # manifest last
TIMER_KEYS = ("decode", "infer", "aggregate", "frames", "videos", "errors")


class CheckpointError(RuntimeError):
    """A checkpoint cannot be used or would be overwritten; the run must not go on."""


def gen_name(n):
    return f"gen-{n:04d}"


def make_fingerprint(input_path, limit_videos, output, run_name):
    """What must be identical for a resume. The parquet files themselves are checked separately
    (Checkpointer.inputs_used), so files added to the input folder later do not matter."""
    fp = {"input": str(input_path), "limit_videos": limit_videos, "output": Path(output).name, "run_name": run_name}
    return json.loads(json.dumps(fp))


class _HashingFile:
    """Write-through file wrapper that counts bytes and hashes them, so nothing is re-read later."""

    def __init__(self, f):
        self.f, self.size, self.sha = f, 0, hashlib.sha256()

    def write(self, data):
        self.sha.update(data)
        self.size += len(data)
        return self.f.write(data)

    def flush(self):
        self.f.flush()

    def entry(self):
        return {"size": self.size, "sha256": self.sha.hexdigest()}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_generation(gen_dir, output, sidecar, progress, fingerprint, inputs_used=None):
    """Build one generation folder; the manifest is written last. Size and sha256 are taken while writing."""
    gen_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    with open(gen_dir / PAYLOAD[0], "wb") as raw:
        hashed = _HashingFile(raw)
        with gzip.GzipFile(fileobj=hashed, mode="wb", compresslevel=6) as dst, open(output, "rb") as src:
            shutil.copyfileobj(src, dst)
    files[PAYLOAD[0]] = hashed.entry()
    done_data = Path(sidecar).read_bytes()
    progress_data = json.dumps(progress, indent=1).encode("utf-8")
    for name, data in ((PAYLOAD[1], done_data), (PAYLOAD[2], progress_data)):
        (gen_dir / name).write_bytes(data)
        files[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {
        "files": files,
        "done_count": len(_complete_lines(done_data.decode("utf-8"))),
        "output_bytes": Path(output).stat().st_size,
        "fingerprint": fingerprint,
        "inputs_used": dict(inputs_used or {}),  # parquet name -> size, files with at least one video processed
    }
    (gen_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def _complete_lines(text):
    """Complete sidecar entries (a truncated last line has no newline and is dropped)."""
    lines = text.split("\n")
    lines.pop()
    return [ln for ln in lines if ln]


def _done_ids(sidecar):
    return _complete_lines(Path(sidecar).read_text(encoding="utf-8"))


def validate_generation(gen_dir, deep=True):
    """The manifest dict if every file matches its recorded size (and sha256 when deep), else None."""
    try:
        manifest = json.loads((gen_dir / "manifest.json").read_text(encoding="utf-8"))
        files = manifest["files"]
        for name in PAYLOAD:
            path = gen_dir / name
            if path.stat().st_size != files[name]["size"] or (deep and _sha256(path) != files[name]["sha256"]):
                return None
        manifest["done_count"], manifest["fingerprint"]  # must exist
        return manifest
    except (OSError, ValueError, KeyError, TypeError):
        return None


class DirStore:
    """Generations under <root>/<run_name>/ (root may be a Google Drive mount)."""

    frames = False

    def __init__(self, root, run_name):
        self.base = Path(root)
        self.root = self.base / run_name
        self.label = f"folder {root}"

    def _require_parent(self):
        """The folder itself may be created, nothing above it (an unmounted Drive must not become a local folder)."""
        if not self.base.parent.is_dir():
            raise CheckpointError(f"checkpoint dir parent missing — is Google Drive mounted? ({self.base.parent})")

    def list_gens(self):
        self._require_parent()
        found = (re.fullmatch(r"gen-(\d+)", p.name) for p in self.root.glob("gen-*"))
        return sorted(int(m.group(1)) for m in found if m)

    def is_valid(self, n):
        return validate_generation(self.root / gen_name(n), deep=False) is not None

    def delete(self, n):
        shutil.rmtree(self.root / gen_name(n), ignore_errors=True)

    def publish(self, staging, n):
        self._require_parent()
        dst = self.root / gen_name(n)
        shutil.rmtree(dst, ignore_errors=True)  # a half-copied earlier try
        dst.mkdir(parents=True)
        for name in FILES:  # manifest last
            shutil.copyfile(staging / gen_name(n) / name, dst / name)

    def fetch(self, n, scratch):
        return self.root / gen_name(n)


class HFStore:
    """Generations in a private HF dataset repo at <run_name>/gen-NNNN/, one commit each.

    `frames/` (dump-frames files) sits next to the generations and is never pruned.
    """

    frames = True

    def __init__(self, repo, run_name, api=None):
        self.repo, self.run = repo, run_name
        self.label = f"HF dataset {repo}"
        self._api = api
        self._invalid = set()  # generations seen without a manifest

    @property
    def api(self):
        if self._api is None:
            from huggingface_hub import HfApi  # lazy: not needed for a plain run

            self._api = HfApi(token=_hf_token())
        return self._api

    def list_gens(self):
        pat = re.compile(re.escape(self.run) + r"/gen-(\d+)/")
        files = self.api.list_repo_files(self.repo, repo_type="dataset")
        gens = sorted({int(m.group(1)) for m in map(pat.match, files) if m})
        self._invalid = {n for n in gens if f"{self.run}/{gen_name(n)}/manifest.json" not in files}
        return gens

    def is_valid(self, n):  # one commit per generation, so a manifest in the listing means it is complete
        return n not in self._invalid

    def delete(self, n):
        self.api.delete_folder(path_in_repo=f"{self.run}/{gen_name(n)}", repo_id=self.repo, repo_type="dataset")

    def publish(self, staging, n):
        self.api.upload_folder(
            folder_path=str(staging), path_in_repo=self.run, repo_id=self.repo,
            repo_type="dataset", commit_message=f"{self.run} {gen_name(n)}",
        )

    def fetch(self, n, scratch):
        for name in FILES:
            self.api.hf_hub_download(
                repo_id=self.repo, filename=f"{self.run}/{gen_name(n)}/{name}", repo_type="dataset",
                local_dir=str(scratch),
            )
        return Path(scratch) / self.run / gen_name(n)


def prune_store(store, gens, keep=KEEP):
    """Keep the newest `keep` valid generations, delete invalid and older ones; returns the kept numbers."""
    kept = [n for n in gens if store.is_valid(n)][-keep:]
    for n in gens:
        if n not in kept:
            store.delete(n)
    return kept


def _hf_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from google.colab import userdata  # Colab Secrets

            token = userdata.get("HF_TOKEN")
        except Exception:
            token = None
    return token


def _iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


class Checkpointer:
    def __init__(self, stores, output, fingerprint, minutes=10, dump_dir=None, clock=time.monotonic, wall=time.time,
                 inputs=None):
        self.stores = stores
        self.output = Path(output)
        self.sidecar = sidecar_path(output)
        self.fingerprint = fingerprint
        self.interval = minutes * 60
        self.dump_dir = Path(dump_dir) if dump_dir else None
        self.clock, self.wall = clock, wall
        self.scratch = Path(str(output) + ".ckpt")
        self._start, self._wall_start = clock(), wall()
        self._last = self._start
        self.inputs = dict(inputs or {})  # parquet name -> size of every input file now
        self.inputs_used = {}  # the subset that has had at least one video processed (name -> size)
        self.gens = {}  # store index -> generation numbers: listed once at startup, then kept up to date
        self._frames_sent = set()
        self.prior = {"compute_s": 0.0, "sessions": 0, "timers": {}}  # progress from earlier sessions

    def startup(self):
        """List every store once. An unreachable store is an error, never "no checkpoint"."""
        if len(self.gens) == len(self.stores):
            return
        failed = []
        for i, store in enumerate(self.stores):
            try:
                self.gens[i] = store.list_gens()
            except Exception as e:
                failed.append(f"{store.label}: {e}")
        if failed:
            self.gens = {}
            raise CheckpointError("cannot list the checkpoint store(s), refusing to start: " + "; ".join(failed))

    def has_existing(self):
        self.startup()
        return any(self.gens.values())

    def note_input(self, parquet_name):
        """Call when a video of this parquet file is done: a resume then insists on the same file size."""
        if parquet_name in self.inputs:
            self.inputs_used[parquet_name] = self.inputs[parquet_name]

    # ---- restore ----
    def restore(self):
        """Put the best checkpoint into place if it is ahead of the local files. Returns a message."""
        for p in (Path(str(self.output) + ".tmp"), Path(str(self.sidecar) + ".tmp")):
            p.unlink(missing_ok=True)
        shutil.rmtree(self.scratch, ignore_errors=True)
        self.startup()
        found = []  # (done_count, store index, gen dir, manifest): newest valid generation per store
        for i, store in enumerate(self.stores):
            for n in reversed(self.gens[i]):
                try:
                    gen_dir = store.fetch(n, self.scratch / str(i))
                    manifest = validate_generation(gen_dir)
                except Exception as e:
                    log.warning("%s: %s could not be fetched (%s: %s), trying an older one",
                                store.label, gen_name(n), type(e).__name__, e)
                    continue
                if manifest:
                    found.append((manifest["done_count"], -i, gen_dir, manifest, store))
                    break
                log.warning("%s: %s is incomplete or corrupt, using an older one", store.label, gen_name(n))
        if not found:
            return "no checkpoint found, continuing with the local files"
        done, _, gen_dir, manifest, store = max(found, key=lambda c: c[:2])
        if manifest["fingerprint"] != self.fingerprint:
            raise CheckpointError(
                f"{store.label} holds a checkpoint of a different run (input path / --limit-videos / output name / "
                "run name differ). Use another --run-name, or the same inputs and options."
            )
        used = manifest.get("inputs_used", {})
        for name, size in used.items():  # only files that were already processed matter
            if self.inputs.get(name) != size:
                now = "is missing" if name not in self.inputs else f"has size {self.inputs[name]}"
                raise CheckpointError(
                    f"cannot resume: input file {name} {now}, but the checkpoint processed it at size {size}. "
                    "Download it again unchanged, or use another --run-name."
                )
        self.inputs_used = dict(used)
        try:
            self.prior = json.loads((gen_dir / PAYLOAD[2]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        local = len(_done_ids(self.sidecar)) if self.output.exists() and self.sidecar.exists() else 0
        if local >= done:
            return f"local files are as new as the checkpoint ({local} videos); using them"
        tmp = Path(str(self.output) + ".tmp")
        with gzip.open(gen_dir / PAYLOAD[0], "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        side_tmp = Path(str(self.sidecar) + ".tmp")
        shutil.copyfile(gen_dir / PAYLOAD[1], side_tmp)
        os.replace(tmp, self.output)
        os.replace(side_tmp, self.sidecar)
        return f"restored {done} videos from {store.label}"

    # ---- snapshot ----
    def maybe_snapshot(self, timers):
        if self.clock() - self._last >= self.interval:
            self.snapshot(timers)

    def snapshot(self, timers):
        """Never raises: a failing store only costs a warning."""
        try:
            self._snapshot(timers)
        except Exception as e:
            log.warning("checkpoint failed (%s: %s); the run goes on", type(e).__name__, e)
        self._last = self.clock()

    def _snapshot(self, timers):
        stage = self.scratch / "stage"
        shutil.rmtree(stage, ignore_errors=True)
        staged = stage / "gen-staging"
        write_generation(staged, self.output, self.sidecar, self._progress(timers), self.fingerprint, self.inputs_used)
        new_frames = self._stage_frames(stage) if any(s.frames for s in self.stores) else []
        ok = 0
        for i, store in enumerate(self.stores):
            try:
                if i not in self.gens:  # startup() did not run: list this store now
                    self.gens[i] = store.list_gens()
                n = 1 + max(self.gens[i], default=0)  # numbered per store: one failing store never blocks another
                if staged.name != gen_name(n):
                    staged = staged.rename(stage / gen_name(n))
                store.publish(stage, n)
            except Exception as e:
                log.warning("%s: snapshot failed (%s: %s)", store.label, type(e).__name__, e)
                continue
            ok += 1
            self.gens[i].append(n)
            if store.frames:
                self._frames_sent.update(new_frames)
            try:
                self.gens[i] = prune_store(store, self.gens[i])
            except Exception as e:  # the snapshot itself is safe; old generations are pruned next time
                log.warning("%s: pruning old generations failed (%s: %s)", store.label, type(e).__name__, e)
        shutil.rmtree(stage, ignore_errors=True)
        if ok:
            log.info("checkpoint written to %d store(s)", ok)

    def _stage_frames(self, stage):
        """Copy dump-frames files not uploaded yet into stage/frames; returns their names."""
        names = []
        if self.dump_dir and self.dump_dir.is_dir():
            names = [p.name for p in self.dump_dir.glob("*.npz")
                     if not p.name.endswith(".tmp.npz") and p.name not in self._frames_sent]
            if names:
                (stage / "frames").mkdir()
                for name in names:
                    shutil.copyfile(self.dump_dir / name, stage / "frames" / name)
        return names

    def _progress(self, timers):
        prior = self.prior
        return {
            "compute_s": round(prior.get("compute_s", 0.0) + self.clock() - self._start, 1),
            "wall_first_start": prior.get("wall_first_start") or _iso(self._wall_start),
            "wall_last_end": _iso(self.wall()),
            "sessions": prior.get("sessions", 0) + 1,
            "timers": {k: round(prior.get("timers", {}).get(k, 0) + getattr(timers, k, 0), 3) for k in TIMER_KEYS},
        }
