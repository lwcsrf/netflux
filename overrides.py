"""Minimal stub of the `overrides` package used in tests."""

from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable[..., object])


def override(func: F) -> F:
    """No-op decorator that mirrors the real package's signature."""

    return func
