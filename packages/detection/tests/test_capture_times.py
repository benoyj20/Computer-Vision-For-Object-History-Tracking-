from datetime import datetime, timedelta
from pathlib import Path

import pytest
from PIL import ExifTags, Image

from objhist_detection.capture_times import (
    EXIF_DATETIME_ORIGINAL_TAG,
    EXIF_OFFSET_TIME_ORIGINAL_TAG,
    EXIF_SUBSEC_TIME_ORIGINAL_TAG,
    CaptureTimeError,
    read_exif_capture_time,
    session_capture_times,
)
from objhist_detection.counter_labels.sessions import SessionPhoto, find_session_photos

SESSION_ID = "2026-10-12-evening"


def save_photo(
    path: Path,
    taken: str | None = "2026:10:12 13:47:54",
    offset: str | None = "-04:00",
    subseconds: str | None = None,
) -> Path:
    exif = Image.Exif()
    ifd = exif.get_ifd(ExifTags.IFD.Exif)
    for tag, value in (
        (EXIF_DATETIME_ORIGINAL_TAG, taken),
        (EXIF_OFFSET_TIME_ORIGINAL_TAG, offset),
        (EXIF_SUBSEC_TIME_ORIGINAL_TAG, subseconds),
    ):
        if value is not None:
            ifd[tag] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 20), "white").save(path, exif=exif)
    return path


def session_photos(tmp_path: Path) -> list[SessionPhoto]:
    return find_session_photos(tmp_path)


def write_manifest(tmp_path: Path, text: str) -> None:
    (tmp_path / SESSION_ID / "manifest.csv").write_text(text, encoding="utf-8")


def test_exif_time_keeps_its_offset_and_subseconds(tmp_path: Path) -> None:
    photo = save_photo(tmp_path / "a.jpg", subseconds="365")

    captured_at = read_exif_capture_time(photo)

    assert captured_at == datetime.fromisoformat("2026-10-12T13:47:54.365-04:00")
    assert captured_at.utcoffset() == timedelta(hours=-4)


def test_exif_time_without_an_offset_is_rejected(tmp_path: Path) -> None:
    photo = save_photo(tmp_path / "a.jpg", offset=None)

    with pytest.raises(CaptureTimeError, match="no EXIF UTC offset"):
        read_exif_capture_time(photo)


def test_photo_without_an_exif_time_is_rejected(tmp_path: Path) -> None:
    photo = save_photo(tmp_path / "a.jpg", taken=None)

    with pytest.raises(CaptureTimeError, match="no EXIF capture time"):
        read_exif_capture_time(photo)


def test_session_uses_exif_times_without_a_manifest(tmp_path: Path) -> None:
    save_photo(tmp_path / SESSION_ID / "a.jpg", taken="2026:10:12 14:00:00")
    save_photo(tmp_path / SESSION_ID / "b.jpg", taken="2026:10:12 13:00:00")

    source, times = session_capture_times(tmp_path / SESSION_ID, session_photos(tmp_path))

    assert source == "exif"
    assert times["b.jpg"] < times["a.jpg"]


def test_session_rejects_two_photos_with_the_same_exif_time(tmp_path: Path) -> None:
    save_photo(tmp_path / SESSION_ID / "a.jpg")
    save_photo(tmp_path / SESSION_ID / "b.jpg")

    with pytest.raises(CaptureTimeError, match="the same EXIF time"):
        session_capture_times(tmp_path / SESSION_ID, session_photos(tmp_path))


def test_manifest_wins_over_exif(tmp_path: Path) -> None:
    save_photo(tmp_path / SESSION_ID / "a.jpg", taken=None)
    save_photo(tmp_path / SESSION_ID / "b.jpg", taken=None)
    write_manifest(
        tmp_path,
        "filename,captured_at\na.jpg,2026-10-12T14:05:00-04:00\nb.jpg,2026-10-12T18:06:00Z\n",
    )

    source, times = session_capture_times(tmp_path / SESSION_ID, session_photos(tmp_path))

    assert source == "manifest"
    assert times["b.jpg"] - times["a.jpg"] == timedelta(minutes=1)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("a.jpg,2026-10-12T14:05:00\nb.jpg,2026-10-12T14:06:00-04:00", "has no UTC offset"),
        ("a.jpg,yesterday\nb.jpg,2026-10-12T14:06:00-04:00", "invalid time"),
        ("a.jpg,2026-10-12T14:05:00-04:00\na.jpg,2026-10-12T14:06:00-04:00", "twice"),
        ("b.jpg,2026-10-12T14:06:00-04:00\na.jpg,2026-10-12T14:06:00-04:00", "not captured after"),
        ("a.jpg,2026-10-12T14:05:00-04:00", r"missing: \['b.jpg'\]"),
        (
            (
                "a.jpg,2026-10-12T14:05:00-04:00\nb.jpg,2026-10-12T14:06:00-04:00\n"
                "c.jpg,2026-10-12T14:07:00-04:00"
            ),
            r"without a photo: \['c.jpg'\]",
        ),
    ],
)
def test_invalid_manifests_are_rejected(tmp_path: Path, rows: str, message: str) -> None:
    save_photo(tmp_path / SESSION_ID / "a.jpg")
    save_photo(tmp_path / SESSION_ID / "b.jpg")
    write_manifest(tmp_path, f"filename,captured_at\n{rows}\n")

    with pytest.raises(CaptureTimeError, match=message):
        session_capture_times(tmp_path / SESSION_ID, session_photos(tmp_path))


def test_manifest_with_wrong_columns_is_rejected(tmp_path: Path) -> None:
    save_photo(tmp_path / SESSION_ID / "a.jpg")
    write_manifest(tmp_path, "file,time\na.jpg,2026-10-12T14:05:00-04:00\n")

    with pytest.raises(CaptureTimeError, match="must have the columns"):
        session_capture_times(tmp_path / SESSION_ID, session_photos(tmp_path))
