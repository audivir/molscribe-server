"""Fixtures for molscribe_server tests."""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

import pytest
from helpers import make_2d_molblock

if TYPE_CHECKING:
    from collections.abc import Callable

PREDICTED_SMILES = "CCO"
PREDICTED_MOLFILE = make_2d_molblock(PREDICTED_SMILES)


class FakeMolScribe:
    """Stand-in for `molscribe.MolScribe` that avoids the real (torch-heavy) model."""

    def __init__(self, ckpt_path: str, device: object) -> None:
        """Store the given (unused) checkpoint path and device."""
        self.ckpt_path = ckpt_path
        self.device = device

    def predict_image_file(self, unused_path: str) -> dict[str, str]:
        """Return a fixed prediction regardless of the input image."""
        return {"smiles": PREDICTED_SMILES, "molfile": PREDICTED_MOLFILE}


def _fake_hf_hub_download(*unused_args: object, **unused_kwargs: object) -> str:
    return "fake-ckpt-path"


@pytest.fixture
def captured_uvicorn_run(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the heavy dependencies used by `main()` and capture the built app."""
    fake_molscribe_module = types.ModuleType("molscribe")
    fake_molscribe_module.MolScribe = FakeMolScribe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "molscribe", fake_molscribe_module)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake_hf_hub_download)

    captured: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: object) -> None:
        captured["app"] = app
        captured["kwargs"] = kwargs

    monkeypatch.setattr("uvicorn.run", fake_run)
    return captured


@pytest.fixture
def make_fake_model() -> Callable[[dict[str, str]], object]:
    """Factory for a fake model exposing `predict_image_file`."""

    def _make(prediction: dict[str, str]) -> object:
        class _FakeModel:
            def predict_image_file(self, unused_path: str) -> dict[str, str]:
                return prediction

        return _FakeModel()

    return _make
