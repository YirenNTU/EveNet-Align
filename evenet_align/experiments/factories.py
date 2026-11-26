"""User code boundary: resolve ``package.module:name`` and build it on the training events."""
from __future__ import annotations

import importlib
from typing import Any, Optional

from evenet_align.core.types import EventSet, Factory


def resolve(path: str) -> Any:
    module, _, name = path.partition(":")
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as error:
        raise ImportError(f"cannot import {path!r}: {error}") from error


def build(factory: Optional[Factory], train: EventSet) -> Any:
    """``function(train, **params)``, or None when no factory is configured."""
    if factory is None:
        return None
    return resolve(factory.function)(train, **factory.params)
