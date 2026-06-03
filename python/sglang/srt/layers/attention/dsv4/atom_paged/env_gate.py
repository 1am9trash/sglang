"""Single source of truth for whether the experimental ATOM paged-attention
backend is active.

Activated by ``SGLANG_HACK_FLASHMLA_BACKEND=atom_paged`` (the same env var the
existing FlashMLA backend dispatch reads). When inactive, NONE of the atom_paged
code paths must run — every hook in the memory pool / model / backend is guarded
by ``is_atom_paged()`` so the default behaviour is byte-for-byte unchanged.
"""

from __future__ import annotations

import functools
import os

_ENV = "SGLANG_HACK_FLASHMLA_BACKEND"
_VALUE = "atom_paged"


@functools.lru_cache(maxsize=1)
def is_atom_paged() -> bool:
    return os.environ.get(_ENV, "") == _VALUE
