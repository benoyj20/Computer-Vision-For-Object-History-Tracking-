"""Find the photos of provided counter sessions.

Layout (see `data/README.md`): `<sequences-dir>/<session-id>/<photo>.jpg`. Other files in a
session folder, such as `manifest.csv` or a change log, are not photos and are skipped.
"""

import re
from dataclasses import dataclass
from pathlib import Path

PHOTO_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
UNSUPPORTED_PHOTO_SUFFIXES = frozenset({".heic", ".heif", ".dng"})
# Session IDs and photo stems become exported file names, so only allow path-safe characters.
SAFE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class SessionError(ValueError):
    """The sessions folder does not match the expected layout."""


@dataclass(frozen=True)
class SessionPhoto:
    session_id: str
    filename: str
    path: Path

    @property
    def key(self) -> str:
        """Identifier that is unique across sessions, used in every intermediate file."""
        return f"{self.session_id}/{self.filename}"

    @property
    def export_stem(self) -> str:
        return f"{self.session_id}__{Path(self.filename).stem}"


def find_session_photos(
    sequences_dir: Path, session_ids: list[str] | None = None
) -> list[SessionPhoto]:
    """Return the photos of every session (or only `session_ids`), sorted by session and name."""
    if not sequences_dir.is_dir():
        raise SessionError(f"{sequences_dir} is not a folder of session folders")
    session_dirs = sorted(path for path in sequences_dir.iterdir() if path.is_dir())
    if session_ids is not None:
        known = {path.name for path in session_dirs}
        missing = sorted(set(session_ids) - known)
        if missing:
            raise SessionError(f"no session folder for {', '.join(missing)} in {sequences_dir}")
        session_dirs = [path for path in session_dirs if path.name in session_ids]
    photos: list[SessionPhoto] = []
    for session_dir in session_dirs:
        if not SAFE_NAME_PATTERN.fullmatch(session_dir.name):
            raise SessionError(f"session folder name {session_dir.name!r} has unsafe characters")
        session_photos = []
        for path in sorted(session_dir.iterdir()):
            suffix = path.suffix.lower()
            if suffix in UNSUPPORTED_PHOTO_SUFFIXES:
                raise SessionError(
                    f"{path} is {suffix[1:].upper()}; save photos as JPEG "
                    "(on Android, Open Camera's default; on iPhone, Camera > Formats > Most Compatible)"
                )
            if not path.is_file() or suffix not in PHOTO_SUFFIXES:
                continue
            if not SAFE_NAME_PATTERN.fullmatch(path.stem):
                raise SessionError(f"photo name {path.name!r} has unsafe characters")
            session_photos.append(SessionPhoto(session_dir.name, path.name, path))
        stems = [Path(photo.filename).stem for photo in session_photos]
        if len(stems) != len(set(stems)):
            raise SessionError(f"{session_dir} has photos that differ only by extension")
        photos.extend(session_photos)
    if not photos:
        raise SessionError(f"no photos found under {sequences_dir}")
    return photos
