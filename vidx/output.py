"""JSONL writer with a `<output>.done` sidecar so a run can be resumed. Standard library only."""
import json
import os
from pathlib import Path


class ResumeError(RuntimeError):
    """--resume was requested but the previous run cannot be continued safely."""


def sidecar_path(output):
    return Path(str(output) + ".done")


class OutputWriter:
    """One write + flush per video, then the video id is appended to the sidecar.

    A video is "done" only once its sidecar entry (`video_id<TAB>line_count`) exists
    and, on resume, all those lines are in the output, so a half-written video is never
    counted as done. Plain `video_id` entries from older runs are still accepted.
    Without resume, both files start empty.
    """

    def __init__(self, path, resume=False):
        self.path = Path(path)
        self.sidecar = sidecar_path(path)
        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = self._recover() if resume else set()
        mode = "a" if resume else "w"
        self._out = open(self.path, mode, encoding="utf-8", newline="\n")
        self._side = open(self.sidecar, mode, encoding="utf-8", newline="\n")

    def _recover(self):
        if not self.sidecar.exists() and self.path.exists() and self.path.stat().st_size > 0:
            raise ResumeError(
                f"cannot resume: {self.path} has results but the sidecar {self.sidecar} is missing. "
                "Restore the .done file, or run without --resume to start fresh (this overwrites the output)."
            )
        # sidecar entry: "video_id<TAB>line_count" (older files: plain "video_id", count unknown)
        claimed = {}
        if self.sidecar.exists():
            lines = self.sidecar.read_text(encoding="utf-8").split("\n")
            lines.pop()  # last piece is "" or an incomplete (possibly truncated) entry
            for ln in lines:
                video, _, n = ln.partition("\t")
                if video:
                    claimed[video] = int(n) if n.isdigit() else None
        found = {}  # video -> its complete output lines
        if self.path.exists():
            with open(self.path, encoding="utf-8", errors="replace") as f:
                for ln in f:
                    if not ln.endswith("\n"):
                        continue  # half-written line
                    try:
                        video = json.loads(ln).get("video")
                    except (ValueError, AttributeError):
                        continue
                    if video in claimed:
                        found.setdefault(video, []).append(ln)
        # done = in the sidecar AND has at least one complete line in the output (exactly n if the count is
        # known); otherwise the video is redone
        done = {v: len(found[v]) for v, n in claimed.items() if v in found and n in (None, len(found[v]))}
        self._atomic_write(self.sidecar, "".join(f"{v}\t{n}\n" for v, n in sorted(done.items())))
        self._atomic_write(self.path, "".join("".join(ls) for v, ls in found.items() if v in done))
        return set(done)

    @staticmethod
    def _atomic_write(path, text):
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)

    def is_done(self, video_id):
        return video_id in self.done

    def write_video(self, video_id, lines):
        """lines: list of dicts for this video (seconds and/or one error line); never empty."""
        if not lines:
            raise ValueError(f"{video_id}: refusing to record a video with no output lines")
        text = "".join(json.dumps(ln, ensure_ascii=False) + "\n" for ln in lines)
        self._out.write(text)
        self._out.flush()
        self._side.write(f"{video_id}\t{len(lines)}\n")
        self._side.flush()
        self.done.add(video_id)

    def close(self):
        self._out.close()
        self._side.close()
