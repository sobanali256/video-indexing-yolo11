"""PyAV decode -> (BGR ndarray, exact Fraction timestamp), rotation applied.

av, cv2 and numpy are imported lazily so the pure helpers (Timestamper,
normalize_rotation) can be tested without them.
"""
import io
import logging
from collections import deque
from fractions import Fraction
from statistics import median
from typing import Any, NamedTuple

log = logging.getLogger("vidx.decode")

ATTACHED_PIC = 0x400  # AV_DISPOSITION_ATTACHED_PIC (cover art)
FALLBACK_FPS = 30


class Frame(NamedTuple):
    image: Any  # BGR uint8 ndarray, contiguous, rotation already applied
    t: Fraction  # seconds since the first decoded frame, >= 0
    pts: Any  # raw pts (int) or None
    rotation: int  # counter-clockwise degrees applied: 0, 90, 180 or 270


class Timestamper:
    """t = (pts - first_pts) * time_base as an exact Fraction, clamped to >= 0.

    A frame without pts gets last_t + step, where step is 1/rate if the rate is
    known, else the median of the positive gaps seen so far, else 1/30. If pts
    only appear after such frames, first_pts is chosen so the first pts frame
    continues that estimated timeline. Every fallback and clamp is logged once
    per video.
    """

    def __init__(self, time_base, rate=None):
        self.time_base = Fraction(time_base) if time_base else None
        self.rate = Fraction(rate) if rate else None
        self.first_pts = None
        self.last_t = None
        self.gaps = deque(maxlen=101)
        self.index = 0
        self.fallbacks = 0
        self.clamped = 0

    def _step(self):
        if self.rate:
            return 1 / self.rate
        if self.gaps:
            return median(self.gaps)
        return Fraction(1, FALLBACK_FPS)

    def stamp(self, pts):
        i = self.index
        self.index += 1
        if pts is not None and self.time_base is not None:
            if self.first_pts is None:
                # t of this frame: 0 for the first frame, else continue the estimated timeline
                start = Fraction(0) if self.last_t is None else self.last_t + self._step()
                self.first_pts = pts - start / self.time_base
            t = (pts - self.first_pts) * self.time_base
            if t < 0:
                self.clamped += 1
                if self.clamped == 1:
                    log.warning("negative timestamp clamped to 0 (frame %d)", i)
                t = Fraction(0)
        else:
            self.fallbacks += 1
            if self.fallbacks == 1:
                log.warning("frame %d has no usable pts: estimating timestamps", i)
            t = Fraction(0) if self.last_t is None else self.last_t + self._step()
        if self.last_t is not None and t > self.last_t:
            self.gaps.append(t - self.last_t)
        self.last_t = t
        return t


def normalize_rotation(angle):
    """Counter-clockwise degrees (any number) -> 0, 90, 180 or 270."""
    try:
        a = float(angle)
    except (TypeError, ValueError):
        return 0
    k = round(a / 90.0) % 4
    if abs(a - round(a / 90.0) * 90.0) > 1.0:
        log.warning("unusual rotation %.1f deg treated as %d", a, k * 90)
    return k * 90


def _is_attached_pic(stream):
    try:
        d = stream.disposition
        return bool(int(getattr(d, "value", d)) & ATTACHED_PIC)
    except Exception:
        return False


def pick_video_stream(container):
    for s in container.streams.video:
        if not _is_attached_pic(s):
            return s
    return None


def stream_rotation(stream):
    """Rotation (ccw degrees) from stream side data, else the legacy rotate tag. 0 if none."""
    try:
        sd = getattr(stream, "side_data", None)
        if sd:
            for key in ("DISPLAYMATRIX", "displaymatrix", "Display Matrix"):
                if key in sd and isinstance(sd[key], (int, float)):
                    return normalize_rotation(sd[key])
    except Exception:
        pass
    try:
        tag = stream.metadata.get("rotate")
        if tag not in (None, ""):
            return normalize_rotation(-float(tag))  # tag is clockwise degrees
    except Exception:
        pass
    return 0


def frame_rotation(frame, fallback):
    try:
        r = getattr(frame, "rotation", None)
        if r:  # PyAV >= 14: display-matrix angle, counter-clockwise
            return normalize_rotation(r)
    except Exception:
        pass
    return fallback


def rotate_image(image, ccw_degrees):
    import cv2

    if ccw_degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if ccw_degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if ccw_degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def _open(data):
    import av

    return av.open(io.BytesIO(memoryview(data)))


def decode_video(data):
    """Generator of Frame for the first real video stream. Raises on open/decode errors.

    Frames yielded before an error are valid; the caller decides what to do with them.
    """
    container = _open(data)
    try:
        stream = pick_video_stream(container)
        if stream is None:
            raise ValueError("no video stream")
        stream.thread_type = "AUTO"
        rate = stream.guessed_rate or stream.average_rate
        stamper = Timestamper(stream.time_base, rate)
        base_rotation = stream_rotation(stream)
        for frame in container.decode(stream):
            t = stamper.stamp(frame.pts)
            rot = frame_rotation(frame, base_rotation)
            image = frame.to_ndarray(format="bgr24")
            if rot:
                image = rotate_image(image, rot)
            yield Frame(image, t, frame.pts, rot)
    finally:
        container.close()


def probe_rotation(data):
    """(stream_rotation, first_frame_rotation, legacy_rotate_tag) for tools/find_rotated.py."""
    container = _open(data)
    try:
        stream = pick_video_stream(container)
        if stream is None:
            return 0, 0, None
        base = stream_rotation(stream)
        first = 0
        for frame in container.decode(stream):
            first = frame_rotation(frame, 0)
            break
        return base, first, stream.metadata.get("rotate")
    finally:
        container.close()
