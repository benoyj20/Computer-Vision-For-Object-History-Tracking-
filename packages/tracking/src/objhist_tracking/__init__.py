"""Object identity across frames and the object history log.

Owner: Justin. Matches each frame's detections to existing objects by position
and appearance, and records added, removed, and moved events with the time
interval in which each change could have happened.
"""
