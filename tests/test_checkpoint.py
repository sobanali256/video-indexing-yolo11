import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidx.checkpoint import (  # noqa: E402
    Checkpointer, CheckpointError, DirStore, HFStore, make_fingerprint, prune_store, validate_generation,
)
from vidx.output import OutputWriter, sidecar_path  # noqa: E402

FP = make_fingerprint("/in", None, "results.jsonl", "r")
TIMERS = SimpleNamespace(decode=1.0, infer=2.0, aggregate=0.5, frames=10, videos=2, errors=0)


class FakeHfApi:
    """The few HfApi calls HFStore uses, on an in-memory repo (no network)."""

    def __init__(self):
        self.files = {}
        self.commits = 0

    def list_repo_files(self, repo, repo_type=None):
        return sorted(self.files)

    def upload_folder(self, folder_path, path_in_repo, repo_id, repo_type, commit_message):
        self.commits += 1
        root = Path(folder_path)
        for p in root.rglob("*"):
            if p.is_file():
                self.files[f"{path_in_repo}/{p.relative_to(root).as_posix()}"] = p.read_bytes()

    def delete_folder(self, path_in_repo, repo_id, repo_type):
        self.commits += 1
        self.files = {k: v for k, v in self.files.items() if not k.startswith(path_in_repo + "/")}

    def hf_hub_download(self, repo_id, filename, repo_type, local_dir):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[filename])
        return str(target)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / "work" / "results.jsonl"
        self.ckdir = self.root / "drive"

    def tearDown(self):
        self.tmp.cleanup()

    def write_videos(self, ids, resume=False):
        w = OutputWriter(self.out, resume=resume)
        for v in ids:
            w.write_video(v, [{"video": v, "second": s, "objects": [], "people": 0} for s in range(2)])
        w.close()

    def ckpt(self, stores=None, **kw):
        return Checkpointer(stores or [DirStore(self.ckdir, "r")], self.out, kw.pop("fp", FP), **kw)

    def wipe_local(self):
        self.out.unlink()
        sidecar_path(self.out).unlink()

    def gen_dir(self, n):
        return self.ckdir / "r" / f"gen-{n:04d}"


class DirCheckpointTest(Base):
    def test_snapshot_restore_round_trip(self):
        self.write_videos(["a:0", "a:1", "a:2"])
        before = (self.out.read_bytes(), sidecar_path(self.out).read_bytes())
        self.ckpt().snapshot(TIMERS)
        self.wipe_local()
        msg = self.ckpt().restore()
        self.assertIn("restored 3 videos", msg)
        self.assertEqual((self.out.read_bytes(), sidecar_path(self.out).read_bytes()), before)
        w = OutputWriter(self.out, resume=True)
        self.assertEqual(w.done, {"a:0", "a:1", "a:2"})
        w.close()

    def test_keeps_only_two_generations(self):
        self.write_videos(["a:0"])
        c = self.ckpt()
        for _ in range(4):
            c.snapshot(TIMERS)
        self.assertEqual(DirStore(self.ckdir, "r").list_gens(), [3, 4])

    def test_generation_without_manifest_or_with_bad_hash_is_skipped(self):
        self.write_videos(["a:0"])
        c = self.ckpt()
        c.snapshot(TIMERS)  # gen 1: 1 video
        self.write_videos(["a:1"], resume=True)
        c.snapshot(TIMERS)  # gen 2: 2 videos
        self.write_videos(["a:2"], resume=True)
        c.snapshot(TIMERS)  # gen 3: 3 videos (gen 1 pruned, gen 2 + 3 remain)
        self.assertTrue(validate_generation(self.gen_dir(3)))
        (self.gen_dir(3) / "manifest.json").unlink()  # crash before the manifest was written
        self.wipe_local()
        self.assertIn("restored 2 videos", self.ckpt().restore())

        data = (self.gen_dir(2) / "done.txt").read_bytes()  # same size, different content
        (self.gen_dir(2) / "done.txt").write_bytes(data.replace(b"a:", b"b:"))
        self.assertIsNone(validate_generation(self.gen_dir(2)))
        self.wipe_local()
        self.assertIn("no checkpoint found", self.ckpt().restore())

    def test_mismatched_fingerprint_is_refused(self):
        self.write_videos(["a:0"])
        self.ckpt().snapshot(TIMERS)
        self.wipe_local()
        other = make_fingerprint("/elsewhere", None, "results.jsonl", "r")
        with self.assertRaises(CheckpointError):
            self.ckpt(fp=other).restore()
        self.assertFalse(self.out.exists())

    def test_local_newer_than_checkpoint_wins(self):
        self.write_videos(["a:0"])
        self.ckpt().snapshot(TIMERS)
        self.write_videos(["a:1"], resume=True)
        local = self.out.read_bytes()
        self.assertIn("local files are as new", self.ckpt().restore())
        self.assertEqual(self.out.read_bytes(), local)

    def test_output_without_sidecar_is_unusable_so_checkpoint_wins(self):
        self.write_videos(["a:0", "a:1"])
        self.ckpt().snapshot(TIMERS)
        sidecar_path(self.out).unlink()
        self.assertIn("restored 2 videos", self.ckpt().restore())

    def test_restore_removes_leftover_tmp_files(self):
        self.write_videos(["a:0"])
        self.ckpt().snapshot(TIMERS)
        tmps = [Path(str(self.out) + ".tmp"), Path(str(sidecar_path(self.out)) + ".tmp")]
        for t in tmps:
            t.write_text("junk")
        self.ckpt().restore()
        self.assertFalse(any(t.exists() for t in tmps))

    def test_processed_input_files_must_match_but_new_ones_are_fine(self):
        self.write_videos(["a:0"])
        c = self.ckpt(inputs={"a.parquet": 100, "b.parquet": 5})
        c.note_input("a.parquet")
        c.snapshot(TIMERS)
        self.wipe_local()
        c2 = self.ckpt(inputs={"a.parquet": 100, "b.parquet": 7, "c.parquet": 1})  # b changed, c is new: both fine
        self.assertIn("restored 1 videos", c2.restore())
        self.assertEqual(c2.inputs_used, {"a.parquet": 100})
        self.wipe_local()
        for inputs in ({"a.parquet": 101}, {"c.parquet": 1}):
            with self.assertRaises(CheckpointError) as cm:
                self.ckpt(inputs=inputs).restore()
            self.assertIn("a.parquet", str(cm.exception))

    def test_unlistable_store_refuses_both_fresh_and_resume(self):
        class Unlistable(DirStore):
            def list_gens(self):
                raise OSError("Drive disconnected")

        c = self.ckpt([Unlistable(self.ckdir, "r")])
        with self.assertRaises(CheckpointError):
            c.has_existing()
        with self.assertRaises(CheckpointError):
            c.restore()
        c2 = self.ckpt([DirStore(self.ckdir, "r"), Unlistable(self.ckdir, "r")])  # one good store does not help
        with self.assertRaises(CheckpointError):
            c2.has_existing()

    def test_restore_falls_back_to_older_generation_when_fetch_or_validation_raises(self):
        self.write_videos(["a:0"])
        c = self.ckpt()
        c.snapshot(TIMERS)
        self.write_videos(["a:1"], resume=True)
        c.snapshot(TIMERS)
        self.wipe_local()

        class FlakyFetch(DirStore):
            def fetch(self, n, scratch):
                if n == 2:
                    raise ConnectionError("Drive hiccup")
                return super().fetch(n, scratch)

        with self.assertLogs("vidx.checkpoint", "WARNING"):
            self.assertIn("restored 1 videos", self.ckpt([FlakyFetch(self.ckdir, "r")]).restore())
        (self.gen_dir(2) / "manifest.json").write_text("{not json")  # validation failure is also just skipped
        self.wipe_local()
        self.assertIn("restored 1 videos", self.ckpt().restore())

    def test_restore_falls_back_to_other_store(self):
        class Broken(DirStore):
            def fetch(self, n, scratch):
                raise OSError("boom")

        self.write_videos(["a:0"])
        self.ckpt().snapshot(TIMERS)
        self.wipe_local()
        c = self.ckpt([Broken(self.ckdir, "r"), DirStore(self.ckdir, "r")])
        with self.assertLogs("vidx.checkpoint", "WARNING"):
            self.assertIn("restored 1 videos", c.restore())

    def test_missing_parent_dir_is_refused_and_nothing_is_created(self):
        store = DirStore(self.root / "not-mounted" / "MyDrive" / "ckpt", "r")
        with self.assertRaises(CheckpointError) as cm:
            store.list_gens()
        self.assertIn("checkpoint dir parent missing", str(cm.exception))
        self.assertIn("Google Drive mounted", str(cm.exception))
        with self.assertRaises(CheckpointError):
            store.publish(self.root, 1)
        self.assertFalse((self.root / "not-mounted").exists())
        DirStore(self.root / "ckpt", "r").list_gens()  # parent exists, folder itself may be missing: fine

    def test_generation_numbers_are_per_store_and_one_failing_store_does_not_block_another(self):
        self.write_videos(["a:0"])
        good = DirStore(self.ckdir, "r")
        ahead = DirStore(self.root / "other", "r")
        c0 = self.ckpt([ahead])
        for _ in range(3):
            c0.snapshot(TIMERS)  # `ahead` is at generation 3
        c = self.ckpt([ahead, good])
        c.startup()
        c.snapshot(TIMERS)
        self.assertEqual(ahead.list_gens(), [3, 4])
        self.assertEqual(good.list_gens(), [1])

        class ListFails(DirStore):
            def list_gens(self):
                raise OSError("no")

        c = self.ckpt([ListFails(self.root / "x", "r"), good])  # no startup(): the failing store only warns
        with self.assertLogs("vidx.checkpoint", "WARNING"):
            c.snapshot(TIMERS)
        self.assertEqual(good.list_gens(), [1, 2])

    def test_prune_failure_still_counts_as_published_and_frames_are_marked_sent(self):
        api = FakeHfApi()

        def boom(**kw):
            raise ConnectionError("offline")

        api.delete_folder = boom
        frames = self.root / "frames"
        frames.mkdir()
        (frames / "a__0.npz").write_bytes(b"x")
        self.write_videos(["a:0"])
        store = HFStore("me/ckpt", "r", api=api)
        c = self.ckpt([store], dump_dir=frames)
        c.snapshot(TIMERS)
        c.snapshot(TIMERS)
        with self.assertLogs("vidx.checkpoint", "WARNING") as logs:
            c.snapshot(TIMERS)  # gen 3 is published, pruning gen 1 fails
        self.assertTrue(any("pruning" in m for m in logs.output))
        self.assertEqual(store.list_gens(), [1, 2, 3])
        self.assertIn("a__0.npz", c._frames_sent)
        self.assertEqual(sum(1 for k in api.files if k.endswith("a__0.npz")), 1)

    def test_prune_keeps_newest_two_valid_and_drops_invalid_and_older(self):
        self.write_videos(["a:0"])
        c = self.ckpt()
        for _ in range(3):
            c.snapshot(TIMERS)
        store = DirStore(self.ckdir, "r")
        self.assertEqual(store.list_gens(), [2, 3])
        (self.gen_dir(3) / "done.txt").write_bytes(b"short")  # size no longer matches: gen 3 is invalid
        (self.ckdir / "r" / "gen-0004").mkdir()  # partial upload without manifest
        self.assertEqual(store.list_gens(), [2, 3, 4])
        self.assertEqual(prune_store(store, [2, 3, 4]), [2])
        self.assertEqual(store.list_gens(), [2])

    def test_generation_lists_are_kept_in_memory_not_relisted(self):
        class Counting(DirStore):
            lists = 0

            def list_gens(self):
                Counting.lists += 1
                return super().list_gens()

        self.write_videos(["a:0"])
        c = self.ckpt([Counting(self.ckdir, "r")])
        c.startup()
        for _ in range(4):
            c.snapshot(TIMERS)
        self.assertEqual(Counting.lists, 1)

    def test_manifest_hash_and_size_are_correct_when_computed_while_writing(self):
        self.write_videos(["a:0", "a:1"])
        self.ckpt().snapshot(TIMERS)
        self.assertTrue(validate_generation(self.gen_dir(1)))  # deep check recomputes sha256 from the files

    def test_has_existing(self):
        self.assertFalse(self.ckpt().has_existing())
        self.write_videos(["a:0"])
        self.ckpt().snapshot(TIMERS)
        self.assertTrue(self.ckpt().has_existing())
        self.assertFalse(self.ckpt([DirStore(self.ckdir, "other-run")]).has_existing())

    def test_failing_store_only_warns_and_the_other_store_still_gets_the_snapshot(self):
        class Broken(DirStore):
            def publish(self, staging, n):
                raise OSError("Drive disconnected")

        self.write_videos(["a:0"])
        c = self.ckpt([Broken(self.root / "bad", "r"), DirStore(self.ckdir, "r")])
        with self.assertLogs("vidx.checkpoint", "WARNING"):
            c.snapshot(TIMERS)  # must not raise
        self.assertEqual(DirStore(self.ckdir, "r").list_gens(), [1])

    def test_unreadable_output_does_not_raise(self):
        c = self.ckpt()  # output file does not exist
        with self.assertLogs("vidx.checkpoint", "WARNING"):
            c.snapshot(TIMERS)

    def test_maybe_snapshot_waits_for_the_interval(self):
        self.write_videos(["a:0"])
        now = [0.0]
        c = self.ckpt(minutes=10, clock=lambda: now[0])
        c.maybe_snapshot(TIMERS)
        self.assertEqual(DirStore(self.ckdir, "r").list_gens(), [])
        now[0] = 601
        c.maybe_snapshot(TIMERS)
        c.maybe_snapshot(TIMERS)  # interval restarted
        self.assertEqual(DirStore(self.ckdir, "r").list_gens(), [1])

    def test_compute_time_accumulates_across_two_sessions(self):
        self.write_videos(["a:0"])
        now = [1000.0]
        c1 = self.ckpt(clock=lambda: now[0], wall=lambda: 1_700_000_000.0)
        now[0] = 1100.0
        c1.snapshot(TIMERS)
        self.wipe_local()  # runtime reset: new session, new clock
        now = [5.0]
        c2 = self.ckpt(clock=lambda: now[0], wall=lambda: 1_700_001_000.0)
        c2.restore()
        now[0] = 55.0
        c2.snapshot(TIMERS)
        progress = json.loads((self.gen_dir(2) / "progress.json").read_text())
        self.assertEqual(progress["compute_s"], 150.0)
        self.assertEqual(progress["sessions"], 2)
        self.assertEqual(progress["timers"]["frames"], 20)
        self.assertEqual(progress["wall_first_start"], "2023-11-14T22:13:20+00:00")
        self.assertEqual(progress["wall_last_end"], "2023-11-14T22:30:00+00:00")


class HFCheckpointTest(Base):
    def test_one_commit_per_generation_and_round_trip(self):
        api = FakeHfApi()
        store = HFStore("me/ckpt", "r", api=api)
        self.write_videos(["a:0"])
        c = self.ckpt([store])
        c.snapshot(TIMERS)
        self.assertEqual(api.commits, 1)
        self.assertIn("r/gen-0001/manifest.json", api.files)
        self.write_videos(["a:1"], resume=True)
        c.snapshot(TIMERS)
        self.assertEqual(api.commits, 2)
        c.snapshot(TIMERS)  # third generation: gen-0001 is pruned (one extra delete commit)
        self.assertEqual(store.list_gens(), [2, 3])
        self.assertEqual(api.commits, 4)
        self.wipe_local()
        self.assertIn("restored 2 videos", self.ckpt([HFStore("me/ckpt", "r", api=api)]).restore())

    def test_dump_frames_are_uploaded_once_outside_the_generations(self):
        api = FakeHfApi()
        frames = self.root / "frames"
        frames.mkdir()
        (frames / "a__0.npz").write_bytes(b"x")
        (frames / "a__1.tmp.npz").write_bytes(b"half")
        self.write_videos(["a:0"])
        c = self.ckpt([HFStore("me/ckpt", "r", api=api)], dump_dir=frames)
        c.snapshot(TIMERS)
        (frames / "a__2.npz").write_bytes(b"y")
        c.snapshot(TIMERS)
        c.snapshot(TIMERS)  # prunes gen 1; frames stay
        self.assertEqual(sorted(k for k in api.files if "/frames/" in k), ["r/frames/a__0.npz", "r/frames/a__2.npz"])

    def test_hf_failure_only_warns(self):
        api = FakeHfApi()

        def boom(**kw):
            raise ConnectionError("offline")

        api.upload_folder = boom
        self.write_videos(["a:0"])
        with self.assertLogs("vidx.checkpoint", "WARNING"):
            self.ckpt([HFStore("me/ckpt", "r", api=api)]).snapshot(TIMERS)


if __name__ == "__main__":
    unittest.main()
