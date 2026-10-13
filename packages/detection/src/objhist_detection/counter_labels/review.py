"""Reviews of counter-photo proposals: the `reviews.json` contract and the review page.

A photo's review lists every object in it: proposed boxes the reviewer kept (with the class
the reviewer chose, and the box itself if the reviewer adjusted it) and boxes the reviewer drew
for missed objects. Every proposal of a
reviewed photo is either kept or rejected, so a reviewed photo's labels are complete; an
unlabeled object would teach the detector that the object is background.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from objhist_detection.annotation import CLASS_NAMES, PixelBox
from objhist_detection.counter_labels.proposals import LOW_CONFIDENCE_HINT, PhotoRecord, ProposedBox

REVIEWS_FORMAT = "objhist-counter-reviews/1"
MIN_DRAWN_BOX_SIDE_PX = 8.0
"""Drawn or adjusted boxes smaller than this on either side are treated as accidental drags."""
PROPOSED_BOX_TOLERANCE_PX = 0.05
"""A kept proposal that is not marked `adjusted` must keep the proposal's box within rounding,
so an unintended change is caught instead of exported."""

Corners = Annotated[list[float], Field(min_length=4, max_length=4)]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReviewedObject(_Model):
    class_name: str
    box: Corners
    source: Literal["proposed", "drawn"]
    proposal_id: str | None = None
    adjusted: bool = False
    """The reviewer resized or moved a proposed box to fit the object more tightly."""

    @field_validator("class_name")
    @classmethod
    def _known_class(cls, class_name: str) -> str:
        if class_name not in CLASS_NAMES:
            raise ValueError(f"unknown class {class_name!r}")
        return class_name

    @model_validator(mode="after")
    def _proposal_id_matches_source(self) -> Self:
        if (self.source == "proposed") != (self.proposal_id is not None):
            raise ValueError(
                "proposal_id is required for proposed objects and absent for drawn ones"
            )
        if self.adjusted and self.source != "proposed":
            raise ValueError("only proposed boxes can be marked adjusted")
        return self

    def pixel_box(self) -> PixelBox:
        return PixelBox(*self.box)


class PhotoReview(_Model):
    key: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    reviewed_at: AwareDatetime
    objects: tuple[ReviewedObject, ...] = ()
    rejected_proposals: tuple[str, ...] = ()


class ReviewsFile(_Model):
    format: Literal["objhist-counter-reviews/1"]
    photos: tuple[PhotoReview, ...]


class ReviewError(ValueError):
    """`reviews.json` does not match the photos and proposals it reviews."""


def _check_photo_review(
    review: PhotoReview, photo: PhotoRecord, proposals: Sequence[ProposedBox]
) -> None:
    if not review.reviewer.strip():
        raise ReviewError(f"{review.key}: reviewer is empty")
    proposals_by_id = {proposed.box_id: proposed for proposed in proposals}
    kept_ids: list[str] = []
    for reviewed in review.objects:
        try:
            box = reviewed.pixel_box()
        except ValueError as error:
            raise ReviewError(f"{review.key}: {error}") from error
        if not box.fits_within(photo.width_px, photo.height_px):
            raise ReviewError(
                f"{review.key}: box {reviewed.box} exceeds the {photo.width_px}x{photo.height_px} image"
            )
        reviewer_box = reviewed.proposal_id is None or reviewed.adjusted
        if (
            reviewer_box
            and min(box.x_max - box.x_min, box.y_max - box.y_min) < MIN_DRAWN_BOX_SIDE_PX
        ):
            kind = "drawn" if reviewed.proposal_id is None else "adjusted"
            raise ReviewError(f"{review.key}: {kind} box {reviewed.box} is too small")
        if reviewed.proposal_id is None:
            continue
        proposed = proposals_by_id.get(reviewed.proposal_id)
        if proposed is None:
            raise ReviewError(f"{review.key}: unknown proposal {reviewed.proposal_id}")
        expected = (proposed.box.x_min, proposed.box.y_min, proposed.box.x_max, proposed.box.y_max)
        if not reviewed.adjusted and any(
            abs(a - b) > PROPOSED_BOX_TOLERANCE_PX
            for a, b in zip(reviewed.box, expected, strict=True)
        ):
            raise ReviewError(
                f"{review.key}: proposal {reviewed.proposal_id} box was changed without being "
                "marked adjusted"
            )
        kept_ids.append(reviewed.proposal_id)
    decided = kept_ids + list(review.rejected_proposals)
    if len(decided) != len(set(decided)):
        raise ReviewError(f"{review.key}: a proposal is kept or rejected more than once")
    undecided = sorted(set(proposals_by_id) - set(decided))
    unknown = sorted(set(decided) - set(proposals_by_id))
    if undecided or unknown:
        raise ReviewError(
            f"{review.key}: proposals without a verdict {undecided}, unknown proposals {unknown}"
        )


def read_reviews(
    path: Path,
    photos: Mapping[str, PhotoRecord],
    proposals: Mapping[str, Sequence[ProposedBox]],
) -> dict[str, PhotoReview]:
    """Read and validate reviews downloaded from the review page, keyed by photo key."""
    try:
        reviews_file = ReviewsFile.model_validate_json(path.read_bytes())
    except ValidationError as error:
        raise ReviewError(f"{path} is not a valid reviews file:\n{error}") from error
    reviews: dict[str, PhotoReview] = {}
    for review in reviews_file.photos:
        if review.key not in photos:
            raise ReviewError(f"{path}: review of unknown photo {review.key!r}")
        if review.key in reviews:
            raise ReviewError(f"{path}: duplicate review of {review.key}")
        _check_photo_review(review, photos[review.key], proposals[review.key])
        reviews[review.key] = review
    return reviews


def _page_photo(
    photo: PhotoRecord,
    image_path: Path,
    proposals: Sequence[ProposedBox],
    review: PhotoReview | None,
) -> dict[str, object]:
    return {
        "key": photo.key,
        "session": photo.session_id,
        "src": image_path.resolve().as_uri(),
        "width": photo.width_px,
        "height": photo.height_px,
        "proposals": [
            {
                "id": proposed.box_id,
                "className": proposed.class_name,
                "confidence": proposed.confidence,
                "box": [
                    proposed.box.x_min,
                    proposed.box.y_min,
                    proposed.box.x_max,
                    proposed.box.y_max,
                ],
            }
            for proposed in proposals
        ],
        "review": None if review is None else json.loads(review.model_dump_json()),
    }


def render_review_page(
    photos: Mapping[str, PhotoRecord],
    proposals: Mapping[str, Sequence[ProposedBox]],
    image_paths: Mapping[str, Path],
    reviews: Mapping[str, PhotoReview],
) -> str:
    """Return a self-contained HTML page for reviewing every photo's boxes."""
    keys = sorted(photos)
    # The page keys its saved browser progress by the photo set, so a new set starts fresh.
    photo_set_id = hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]
    data = {
        "format": REVIEWS_FORMAT,
        "storageKey": f"objhist-counter-review:{photo_set_id}",
        "classes": list(CLASS_NAMES),
        "lowConfidence": LOW_CONFIDENCE_HINT,
        "minDrawnSide": MIN_DRAWN_BOX_SIDE_PX,
        "photos": [
            _page_photo(photos[key], image_paths[key], proposals[key], reviews.get(key))
            for key in keys
        ],
    }
    # Escape "<" so no value can close the <script> element that embeds the data.
    data_json = json.dumps(data).replace("<", "\\u003c")
    template = resources.files("objhist_detection.counter_labels").joinpath("review_page.html")
    return template.read_text("utf-8").replace("__REVIEW_DATA__", data_json)
