"""Vocabulary-level MLLM-OVOD feedback: revise the concept list, not the boxes.

The earlier agentic loop worked on detections -- inspect a crop, keep or move
the box. It could never add an object the first vocabulary missed, which is
where the headroom plausibly is. This package operates one level up: the
detector's per-concept support is summarised back to the MLLM, which revises
the **vocabulary**, and the detector then re-grounds the revised list on the
full image. The MLLM never sees or emits a coordinate.
"""
