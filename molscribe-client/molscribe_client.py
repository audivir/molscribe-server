"""Predict the molfile or smiles from images."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Literal

__version__ = "0.1.0"


def molscribe_client(
    mode: Literal["molfile", "smiles"],
    server: str,
    port: int,
    inputs: list[Path] | None = None,
    use_stdin: bool = False,
    use_clipboard: bool = False,
) -> None:
    """Predict the molfile or smiles from images.

    Args:
        mode: Return type of the prediction.
        server: IP or hostname of the server.
        port: Port of the server.
        inputs: List of input files.
        use_stdin: Read input files from stdin. Ignored if inputs are provided.
        use_clipboard: Copy the prediction to the clipboard instead of printing it.
    """
    import pyperclip
    import requests

    if not inputs:
        if not use_stdin:
            raise ValueError("No input files provided.")
        inputs = [Path(file) for file in sys.stdin.read().splitlines()]

    data = [base64.b64encode(file.read_bytes()).decode() for file in inputs]
    response = requests.post(f"http://{server}:{port}/predict/{mode}", json=data, timeout=30)
    output = response.json()
    if use_clipboard:
        pyperclip.copy(output)
    else:
        doctyper.echo(output)


def cli():
    """CLI entrypoint."""
    import doctyper

    app = doctyper.SlimTyper()
    app.command()(molscribe_client)
    app()


if __name__ == "__main__":
    cli()
