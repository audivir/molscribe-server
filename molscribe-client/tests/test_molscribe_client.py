"""Tests for molscribe_client."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any

import pytest

from molscribe_client import cli, molscribe_client

if TYPE_CHECKING:
    from pathlib import Path


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def json(self) -> object:
        return self._payload


def test_no_inputs_without_stdin_raises() -> None:
    with pytest.raises(ValueError, match=r"^No input files provided\.$"):
        molscribe_client("smiles", "localhost", 8001, inputs=None, use_stdin=False)


def test_posts_expected_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image-bytes")

    calls: list[dict[str, Any]] = []

    def fake_post(url: str, json: object, timeout: float) -> _FakeResponse:
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse("CCO")

    monkeypatch.setattr("requests.post", fake_post)

    molscribe_client("smiles", "example.com", 1234, inputs=[image])

    assert calls == [
        {
            "url": "http://example.com:1234/predict/smiles",
            "json": [base64.b64encode(b"image-bytes").decode()],
            "timeout": 30,
        }
    ]
    assert capsys.readouterr().out == "CCO\n"


def test_reads_inputs_from_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image-bytes")
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{image}\n"))

    received: dict[str, Any] = {}

    def fake_post(url: str, json: object, timeout: float) -> _FakeResponse:
        received["url"] = url
        received["json"] = json
        received["timeout"] = timeout
        return _FakeResponse("CCO")

    monkeypatch.setattr("requests.post", fake_post)

    molscribe_client("molfile", "localhost", 8001, inputs=None, use_stdin=True)

    assert received["url"] == "http://localhost:8001/predict/molfile"
    assert received["json"] == [base64.b64encode(b"image-bytes").decode()]


def test_copies_output_to_clipboard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image-bytes")
    monkeypatch.setattr("requests.post", lambda *unused_a, **unused_k: _FakeResponse("CCO"))

    copied: list[object] = []
    monkeypatch.setattr("pyperclip.copy", copied.append)

    molscribe_client("smiles", "localhost", 8001, inputs=[image], use_clipboard=True)

    assert copied == ["CCO"]


def test_cli_wires_doctyper_app(monkeypatch: pytest.MonkeyPatch) -> None:
    registered: list[object] = []
    called: list[bool] = []

    class _FakeApp:
        def command(self) -> Any:
            def _register(func: object) -> object:
                registered.append(func)
                return func

            return _register

        def __call__(self) -> None:
            called.append(True)

    monkeypatch.setattr("doctyper.SlimTyper", _FakeApp)

    cli()

    assert registered == [molscribe_client]
    assert called == [True]
