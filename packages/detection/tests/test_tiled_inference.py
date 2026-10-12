from PIL import Image

from objhist_detection.annotation import PixelBox
from objhist_detection.tiled_inference import (
    ScoredBox,
    TilingConfig,
    detect_tiled,
    is_cut_by_inner_edge,
    merge_boxes,
    tile_origins,
    to_detection,
)

CONFIG = TilingConfig()


def test_tiles_cover_the_photo_and_end_at_its_edge() -> None:
    assert tile_origins(4000, 1280, 0.25) == [0, 907, 1813, 2720]
    assert tile_origins(2252, 1280, 0.25) == [0, 486, 972]
    assert tile_origins(2240, 1280, 0.25) == [0, 960]
    assert tile_origins(1280, 1280, 0.25) == [0]
    assert tile_origins(500, 1280, 0.25) == [0]


def test_only_inner_tile_edges_cut_boxes() -> None:
    corner_tile = PixelBox(0, 0, 1280, 1280)
    inner_tile = PixelBox(907, 972, 2187, 2252)

    assert not is_cut_by_inner_edge(PixelBox(0, 0, 50, 50), corner_tile, 4000, 2252, 4)
    assert is_cut_by_inner_edge(PixelBox(1200, 10, 1279, 50), corner_tile, 4000, 2252, 4)
    assert is_cut_by_inner_edge(PixelBox(909, 1500, 1100, 1600), inner_tile, 4000, 2252, 4)
    # The bottom of this tile is the photo's bottom edge, so touching it is not a cut.
    assert not is_cut_by_inner_edge(PixelBox(1500, 2100, 1600, 2252), inner_tile, 4000, 2252, 4)


def scored(class_name: str, confidence: float, *corners: float) -> ScoredBox:
    return ScoredBox(class_name, confidence, PixelBox(*corners))


def test_merge_removes_same_class_partial_views() -> None:
    whole = scored("fork", 0.9, 100, 100, 200, 400)
    partial = scored("fork", 0.6, 100, 100, 200, 250)

    assert merge_boxes([partial, whole], CONFIG) == [whole]


def test_merge_keeps_a_spoon_lying_on_a_plate() -> None:
    plate = scored("plate", 0.95, 0, 0, 1000, 600)
    spoon = scored("spoon", 0.8, 400, 200, 460, 500)

    assert merge_boxes([spoon, plate], CONFIG) == [plate, spoon]


def test_merge_joins_one_object_named_two_ways() -> None:
    as_glass = scored("glass", 0.9, 100, 100, 200, 200)
    as_cup = scored("cup", 0.7, 101, 101, 200, 200)

    assert merge_boxes([as_cup, as_glass], CONFIG) == [as_glass]


def test_merge_joins_a_mug_seen_with_and_without_its_handle() -> None:
    as_cup = scored("cup", 0.97, 1860, 603, 2278, 1072)
    as_glass = scored("glass", 0.96, 1860, 604, 2159, 1075)

    assert merge_boxes([as_glass, as_cup], CONFIG) == [as_cup]


def test_merge_does_not_depend_on_input_order() -> None:
    boxes = [
        scored("cup", 0.5, 0, 0, 10, 10),
        scored("cup", 0.5, 20, 0, 30, 10),
        scored("knife", 0.5, 0, 0, 10, 10),
    ]

    assert merge_boxes(boxes, CONFIG) == merge_boxes(list(reversed(boxes)), CONFIG)


class FakeDetector:
    """Sees a scene in photo pixels: large objects only in the full-photo pass (small ones are
    too small there) and every object inside a tile, partially if the tile cuts it."""

    def __init__(self, scene: list[tuple[str, PixelBox]], photo_size: tuple[int, int]) -> None:
        self.scene = scene
        width, height = photo_size
        self.tiles = [
            PixelBox(x, y, min(x + CONFIG.tile_px, width), min(y + CONFIG.tile_px, height))
            for y in tile_origins(height, CONFIG.tile_px, CONFIG.overlap)
            for x in tile_origins(width, CONFIG.tile_px, CONFIG.overlap)
        ]
        self.calls: list[tuple[tuple[int, int], int]] = []

    def __call__(
        self, image: Image.Image, inference_px: int, confidence_threshold: float
    ) -> list[ScoredBox]:
        self.calls.append((image.size, inference_px))
        if len(self.calls) == 1:
            return [ScoredBox(name, 0.9, box) for name, box in self.scene if box.area > 500_000]
        tile = self.tiles[len(self.calls) - 2]
        seen = []
        for name, box in self.scene:
            x_min, y_min = max(box.x_min, tile.x_min), max(box.y_min, tile.y_min)
            x_max, y_max = min(box.x_max, tile.x_max), min(box.y_max, tile.y_max)
            if x_max > x_min and y_max > y_min:
                visible = (x_max - x_min) * (y_max - y_min) / box.area
                seen.append(
                    ScoredBox(
                        name,
                        0.5 + 0.4 * visible,
                        PixelBox(
                            x_min - tile.x_min,
                            y_min - tile.y_min,
                            x_max - tile.x_min,
                            y_max - tile.y_min,
                        ),
                    )
                )
        return seen


def test_detect_tiled_finds_small_and_large_objects_in_photo_pixels() -> None:
    knife = PixelBox(2062, 1286, 2183, 1911)  # spans the boundary between two tile columns
    fork = PixelBox(463, 1242, 795, 1855)
    plate = PixelBox(2508, 1037, 3673, 1728)  # wider than a tile
    detector = FakeDetector([("knife", knife), ("fork", fork), ("plate", plate)], (4000, 2252))

    detections = detect_tiled(Image.new("RGB", (4000, 2252)), detector, CONFIG)

    found = {(d.class_name, d.box.x_min, d.box.y_min, d.box.x_max, d.box.y_max) for d in detections}
    assert found == {
        ("knife", 2062, 1286, 2183, 1911),
        ("fork", 463, 1242, 795, 1855),
        ("plate", 2508, 1037, 3673, 1728),
    }
    assert detector.calls[0] == ((4000, 2252), CONFIG.full_photo_inference_px)
    assert len(detector.calls) == 1 + 4 * 3
    assert all(call == ((1280, 1280), CONFIG.tile_inference_px) for call in detector.calls[1:])


def test_small_photo_is_one_tile_and_detections_stay_inside_it() -> None:
    detector = FakeDetector([("cup", PixelBox(10, 10, 90, 60))], (100, 80))

    detections = detect_tiled(Image.new("RGB", (100, 80)), detector, CONFIG)

    assert [(d.class_name, d.box.x_max, d.box.y_max) for d in detections] == [("cup", 90, 60)]
    assert detector.calls[1] == ((100, 80), CONFIG.tile_inference_px)


def test_to_detection_clips_to_the_photo() -> None:
    detection = to_detection(scored("plate", 0.9, 10, 10, 120.4, 90), 120, 80)

    assert (detection.box.x_max, detection.box.y_max) == (120, 80)
