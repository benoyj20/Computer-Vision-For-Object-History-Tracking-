"""Shared data contracts for the object history pipeline.

These types are the interface between the detector (`objhist_detection`) and the
tracker (`objhist_tracking`). Changing them affects both owners; see AGENTS.md.
"""

from objhist_core.contracts import BoundingBox, Detection, FrameDetections

__all__ = ["BoundingBox", "Detection", "FrameDetections"]
