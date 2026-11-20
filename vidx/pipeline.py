"""Stage 2a: overlapped decode + inference in one process.

Decode threads (PyAV and cv2 release the GIL) decode, sample, rotate and letterbox frames and put them
on a bounded queue. The calling thread owns the GPU: it collects frames into buckets by letterboxed
shape ACROSS videos, runs a bucket when it holds `batch_size` frames (or when the queue runs dry, since
the GPU would idle anyway), and emits a video's result once its end marker has arrived and none of its
frames is still waiting in a bucket.

Results are yielded as tuples, in completion order (the output line order does not matter):
  ("ok", video_id, lines, last_batch)      lines as run.video_lines() builds them (incl. error lines);
                                           last_batch = newest successful batch that held its frames (0: none)
  ("source_error", video_id, lines, None)  the parquet row could not be read (the model was not used)
  ("fail", video_id, message, batch_no)    inference failed on a batch holding frames of this video

The heavy parts are injected, so this module is testable with fakes:
  decode_fn(data, counter) -> iterator of vidx.decode.Frame (counter.decoded_frames is incremented)
  make_prepare() -> prepare(image) -> letterboxed image   (one per thread)
  infer_fn(list of same-shape letterboxed images) -> list of (counts, conf)
"""
import logging
import queue
import threading
import time
from types import SimpleNamespace

from .aggregate import error_line

log = logging.getLogger("vidx.pipeline")

_FRAME, _END, _DONE = 0, 1, 2


class _Video:
    __slots__ = ("records", "outstanding", "ended", "error", "failed", "last_batch")

    def __init__(self):
        self.records = []  # (t, counts)
        self.outstanding = 0  # frames queued in buckets, not yet inferred
        self.ended = False
        self.error = None  # decode error text (frames before it are kept)
        self.failed = None  # (message, batch_no) of an inference failure
        self.last_batch = 0  # number of the last successful batch that held frames of this video


class Pipeline:
    def __init__(self, videos, decode_fn, make_prepare, infer_fn, batch_size, threads, timers,
                 run_inference, video_lines, queue_frames=None, idle_wait=0.05):
        """videos: iterator of (video_id, data, err); shared by the decode threads under a lock."""
        self.videos = videos
        self.decode_fn = decode_fn
        self.make_prepare = make_prepare
        self.infer_fn = infer_fn
        self.batch_size = max(1, batch_size)
        self.threads = max(1, threads)
        self.timers = timers
        self.run_inference = run_inference
        self.video_lines = video_lines
        self.q = queue.Queue(maxsize=queue_frames or 4 * self.batch_size)
        self.idle_wait = idle_wait
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.batch_no = 0
        self.source_exc = None  # exception raised by the shared videos iterator
        self.thread_errors = []  # unexpected decode-thread failures (outside a single video)
        self.state = {}  # video_id -> _Video
        self.buckets = {}  # letterboxed shape -> [(video_id, t, image)]

    # ------------------------------------------------------------- decode threads
    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _next_video(self):
        """Next (video_id, data, err) or None. An error from the shared iterator (e.g. an unexpected
        parquet read error) stops the whole pipeline; run() re-raises it on the main thread."""
        with self.lock:
            if self.source_exc is not None:
                return None
            try:
                return next(self.videos, None)
            except BaseException as e:
                self.source_exc = e
                self.stop.set()
                return None

    def _producer(self):
        try:
            prepare = self.make_prepare()
            while not self.stop.is_set():
                item = self._next_video()
                if item is None:
                    break
                video_id, data, err = item
                if err:
                    if not self._put((_END, video_id, 0, 0.0, err, True)):
                        return
                    continue
                counter = SimpleNamespace(decoded_frames=0)
                busy = 0.0  # decode + letterbox time only, not time blocked on a full queue
                error = None
                try:
                    frames = iter(self.decode_fn(data, counter))
                    while True:
                        t0 = time.perf_counter()
                        try:
                            f = next(frames)
                        except StopIteration:
                            busy += time.perf_counter() - t0
                            break
                        image = prepare(f.image)
                        busy += time.perf_counter() - t0
                        if not self._put((_FRAME, video_id, f.t, image)):
                            return
                except Exception as e:  # decode failure: frames already queued stay valid
                    error = f"{type(e).__name__}: {e}"[:200]
                if not self._put((_END, video_id, counter.decoded_frames, busy, error, False)):
                    return
        except Exception as e:  # never let a thread die silently: report, the thread still ends with _DONE
            log.error("decode thread failed: %s: %s", type(e).__name__, e)
            self.thread_errors.append(f"{type(e).__name__}: {e}")
        finally:
            self._put((_DONE,))

    # ------------------------------------------------------------- GPU side
    def _flush(self, shape):
        batch = self.buckets.pop(shape, [])
        live = [b for b in batch if self.state[b[0]].failed is None]
        for vid, _, _ in batch:
            if self.state[vid].failed is not None:
                self.state[vid].outstanding -= 1
        if not live:
            return {vid for vid, _, _ in batch}
        self.batch_no += 1
        t0 = time.perf_counter()
        try:
            results = self.run_inference(self.infer_fn, [img for _, _, img in live])
        except Exception as e:  # InferenceError (run_inference wraps everything else)
            message = str(e)
            for vid, _, _ in live:
                st = self.state[vid]
                st.outstanding -= 1
                if st.failed is None:
                    st.failed = (message, self.batch_no)
                    st.records = []
            return {vid for vid, _, _ in batch}
        finally:
            self.timers.infer += time.perf_counter() - t0
        self.timers.inferred_frames += len(live)
        for (vid, t, _), (counts, _conf) in zip(live, results):
            st = self.state[vid]
            st.records.append((t, counts))
            st.outstanding -= 1
            st.last_batch = self.batch_no
        return {vid for vid, _, _ in batch}

    def _flush_all(self):
        touched = set()
        for shape in sorted(self.buckets, key=lambda s: -len(self.buckets[s])):
            touched |= self._flush(shape)
        return touched

    def _complete(self, video_ids):
        """Yield results of finished videos among video_ids."""
        for vid in sorted(video_ids):
            st = self.state.get(vid)
            if st is None or not st.ended or st.outstanding:
                continue
            del self.state[vid]
            if st.failed is not None:
                yield ("fail", vid, st.failed[0], st.failed[1])
                continue
            t0 = time.perf_counter()
            records = [SimpleNamespace(t=t, counts=c) for t, c in st.records]
            lines = self.video_lines(vid, records, st.error)
            self.timers.aggregate += time.perf_counter() - t0
            yield ("ok", vid, lines, st.last_batch)

    def run(self):
        threads = [threading.Thread(target=self._producer, name=f"decode-{i}", daemon=True)
                   for i in range(self.threads)]
        for th in threads:
            th.start()
        done = 0
        try:
            while done < len(threads):
                if self.source_exc is not None:
                    raise self.source_exc
                waiting = any(self.buckets.values())
                try:
                    msg = self.q.get(timeout=self.idle_wait if waiting else 1.0)
                except queue.Empty:
                    if waiting:  # nothing new: the GPU would idle, so run partial buckets now
                        yield from self._complete(self._flush_all())
                    continue
                kind = msg[0]
                if kind == _FRAME:
                    _, vid, t, image = msg
                    st = self.state.setdefault(vid, _Video())
                    st.outstanding += 1
                    shape = tuple(image.shape)
                    bucket = self.buckets.setdefault(shape, [])
                    bucket.append((vid, t, image))
                    if len(bucket) >= self.batch_size:
                        yield from self._complete(self._flush(shape))
                elif kind == _END:
                    _, vid, decoded, decode_s, error, source_error = msg
                    self.timers.decoded_frames += decoded
                    self.timers.decode += decode_s
                    if source_error:  # parquet row could not be read: one error line, nothing decoded
                        self.state.pop(vid, None)
                        yield ("source_error", vid, [error_line(vid, error)], None)
                        continue
                    st = self.state.setdefault(vid, _Video())
                    st.ended = True
                    st.error = error
                    yield from self._complete({vid})
                else:  # _DONE
                    done += 1
            if self.source_exc is not None:
                raise self.source_exc
            if self.thread_errors and len(self.thread_errors) >= len(threads):
                raise RuntimeError(f"all decode threads failed: {self.thread_errors[0]}")
            yield from self._complete(self._flush_all())
        finally:
            self.stop.set()
            while True:  # unblock producers waiting on a full queue
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            for th in threads:
                th.join(timeout=5)
