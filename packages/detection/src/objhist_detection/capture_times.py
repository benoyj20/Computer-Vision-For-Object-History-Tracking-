"""Timezone-aware capture times for the photos of one session.

Placement intervals come from these times, so they must be real capture times with a UTC
offset, never file modification times (see `data/README.md`). A session's `manifest.csv` wins
when present; otherwise every photo must carry its EXIF original time and offset. One session
never mixes the two sources, so all its times come from one clock.
"""

import csv
import re
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path

from PIL import ExifTags, Image

from objhist_core import CaptureTimeSource
from objhist_detection.counter_labels.sessions import SessionPhoto

SESSION_MANIFEST_FILENAME = "manifest.csv"
MANIFEST_COLUMNS = ("filename", "captured_at")

EXIF_DATETIME_ORIGINAL_TAG = 0x9003
EXIF_OFFSET_TIME_ORIGINAL_TAG = 0x9011
EXIF_SUBSEC_TIME_ORIGINAL_TAG = 0x9291
EXIF_OFFSET_PATTERN = re.compile(r"[+-]\d{2}:\d{2}")


class CaptureTimeError(ValueError):
    """A photo has no usable capture time, or the session's times are out of order."""


def read_exif_capture_time(path: Path) -> datetime:
    """The photo's EXIF original capture time, with its sub-seconds and UTC offset."""
    with Image.open(path) as image:
        exif = image.getexif().get_ifd(ExifTags.IFD.Exif)
    taken = exif.get(EXIF_DATETIME_ORIGINAL_TAG)
    offset = exif.get(EXIF_OFFSET_TIME_ORIGINAL_TAG)
    if not isinstance(taken, str):
        raise CaptureTimeError(
            f"{path} has no EXIF capture time; add a {SESSION_MANIFEST_FILENAME}"
        )
    if not isinstance(offset, str) or not EXIF_OFFSET_PATTERN.fullmatch(offset.strip()):
        raise CaptureTimeError(
            f"{path} has no EXIF UTC offset, so its time zone is unknown; "
            f"add a {SESSION_MANIFEST_FILENAME} with offset times"
        )
    try:
        # EXIF stores the local time ("2026:10:12 13:47:54") and its offset ("-04:00") apart.
        captured_at = datetime.strptime(f"{taken.strip()}{offset.strip()}", "%Y:%m:%d %H:%M:%S%z")
    except ValueError as error:
        raise CaptureTimeError(f"{path} has an unreadable EXIF capture time {taken!r}") from error
    subseconds = exif.get(EXIF_SUBSEC_TIME_ORIGINAL_TAG)
    if isinstance(subseconds, str) and subseconds.strip().isdigit():
        digits = subseconds.strip()
        captured_at += timedelta(seconds=int(digits) / 10 ** len(digits))
    return captured_at


def read_manifest(path: Path, photos: list[SessionPhoto]) -> dict[str, datetime]:
    """Capture times by photo filename from a session manifest, which lists every photo once."""
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise CaptureTimeError(f"{path} must have the columns {','.join(MANIFEST_COLUMNS)}")
        times: dict[str, datetime] = {}
        previous: tuple[str, datetime] | None = None
        for row in reader:
            filename, text = row["filename"], row["captured_at"]
            if filename in times:
                raise CaptureTimeError(f"{path} lists {filename} twice")
            try:
                captured_at = datetime.fromisoformat(text)
            except (TypeError, ValueError) as error:
                raise CaptureTimeError(
                    f"{path}: {filename} has an invalid time {text!r}"
                ) from error
            if captured_at.utcoffset() is None:
                raise CaptureTimeError(f"{path}: {filename} time {text} has no UTC offset")
            # The manifest is in capture order (data/README.md); a row out of order is a typo.
            if previous is not None and captured_at <= previous[1]:
                raise CaptureTimeError(f"{path}: {filename} is not captured after {previous[0]}")
            times[filename] = captured_at
            previous = (filename, captured_at)
    photo_names = {photo.filename for photo in photos}
    missing = sorted(photo_names - times.keys())
    extra = sorted(times.keys() - photo_names)
    if missing or extra:
        raise CaptureTimeError(
            f"{path} does not match the session photos (missing: {missing or 'none'}, "
            f"without a photo: {extra or 'none'})"
        )
    return times


def session_capture_times(
    session_dir: Path, photos: list[SessionPhoto]
) -> tuple[CaptureTimeSource, dict[str, datetime]]:
    """Capture times by filename for one session's photos, and where they came from.

    Two photos with the same time would give a zero-length placement interval, so ties are
    rejected; EXIF sub-seconds usually separate burst shots.
    """
    manifest = session_dir / SESSION_MANIFEST_FILENAME
    if manifest.is_file():
        return "manifest", read_manifest(manifest, photos)
    times = {photo.filename: read_exif_capture_time(photo.path) for photo in photos}
    ordered = sorted(times.items(), key=lambda item: item[1])
    for (earlier, earlier_time), (later, later_time) in pairwise(ordered):
        if later_time == earlier_time:
            raise CaptureTimeError(
                f"{session_dir}: {earlier} and {later} have the same EXIF time {later_time}; "
                f"add a {SESSION_MANIFEST_FILENAME} to order them"
            )
    return "exif", times
