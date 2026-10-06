"""GPU selection, applied before any framework imports CUDA.

Threading a ``cuda:N`` string through every library is fragile: Ultralytics
derives YOLO-World's CLIP text-encoder device separately from the detector's,
``transformers`` has its own ``device_map`` rules, and the OpenAI CLIP package
pins tokens to whatever device it was built with. A single mismatch surfaces as
``Expected all tensors to be on the same device``.

Masking the GPU instead removes the whole class of bug: ``CUDA_VISIBLE_DEVICES``
makes the requested physical GPU the only one any framework can see, so
everything lands on it without being told. Inside the process it is then
``cuda:0`` -- the same physical card the caller asked for.

This must run **before torch is imported**, because CUDA device visibility is
read once at initialisation.
"""

from __future__ import annotations

import os
import re
import sys

_CUDA_INDEX = re.compile(r"^cuda:(\d+)$")


def pin_cuda_device(spec: str | None) -> str:
    """Mask all GPUs except the one in ``spec``; return the in-process device.

    ``"cuda:1"`` sets ``CUDA_VISIBLE_DEVICES=1`` and returns ``"cuda:0"``.
    ``"cuda"``, ``"cpu"`` and ``None`` pass through untouched. An existing
    ``CUDA_VISIBLE_DEVICES`` set by the caller is left alone, since they have
    already chosen.
    """
    if not spec:
        return "cuda"
    m = _CUDA_INDEX.match(spec.strip())
    if not m:
        return spec
    if "torch" in sys.modules:
        # Too late to mask; fall back to the explicit string and let the
        # frameworks sort it out.
        return spec
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        # The caller has already chosen which GPUs are visible -- typically to
        # keep several visible so one process can place models on different
        # cards. Masking further would undo that, and renumbering to cuda:0
        # would silently move the model to the wrong GPU, so pass the request
        # through untouched.
        return spec
    os.environ["CUDA_VISIBLE_DEVICES"] = m.group(1)
    return "cuda:0"


def pin_from_argv(argv: list[str] | None = None) -> str | None:
    """Read ``--device`` out of raw argv and pin it, before argparse runs."""
    argv = argv or sys.argv
    for i, arg in enumerate(argv):
        if arg == "--device" and i + 1 < len(argv):
            return pin_cuda_device(argv[i + 1])
        if arg.startswith("--device="):
            return pin_cuda_device(arg.split("=", 1)[1])
    return None
