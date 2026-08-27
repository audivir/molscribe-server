"""Tests for molscribe_server."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, cast

import pytest
from conftest import PREDICTED_MOLFILE
from fastapi.testclient import TestClient
from helpers import ETHANOL_SMILES, make_2d_mol
from rdkit import Chem
from rdkit.Chem import AllChem

from molscribe_server import base64_to_mol_data, get_bounding_box, main, translate_mol

if TYPE_CHECKING:
    from collections.abc import Callable

    from molscribe import MolScribe


def test_get_bounding_box() -> None:
    mol = make_2d_mol()
    conf = mol.GetConformer()
    xs, ys = zip(*((p[0], p[1]) for p in conf.GetPositions()), strict=True)
    assert get_bounding_box(mol) == (min(xs), max(xs), min(ys), max(ys))


def test_get_bounding_box_raises_for_3d_molecule() -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles(ETHANOL_SMILES))
    AllChem.EmbedMolecule(mol, randomSeed=42)  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match=r"^Molecule must be 2D\.$"):
        get_bounding_box(mol)


def test_translate_mol_right() -> None:
    placed = make_2d_mol()
    placed_bbox = get_bounding_box(placed)

    moved = translate_mol(make_2d_mol(), placed_bbox, "right")
    moved_bbox = get_bounding_box(moved)

    assert moved_bbox[0] == pytest.approx(placed_bbox[1] + 1)
    assert moved_bbox[2] == pytest.approx(placed_bbox[2])


def test_translate_mol_below() -> None:
    placed = make_2d_mol()
    placed_bbox = get_bounding_box(placed)

    moved = translate_mol(make_2d_mol(), placed_bbox, "below")
    moved_bbox = get_bounding_box(moved)

    assert moved_bbox[0] == pytest.approx(placed_bbox[0])
    assert moved_bbox[2] == pytest.approx(placed_bbox[3] + 1)


def test_base64_to_mol_data(make_fake_model: Callable[[dict[str, str]], object]) -> None:
    image_bytes = b"not a real png, just bytes"
    encoded = base64.b64encode(image_bytes).decode()
    model = make_fake_model({"smiles": "CCO", "molfile": "MOLFILE"})

    smiles, molfile = base64_to_mol_data(cast("MolScribe", model), encoded)

    assert smiles == "CCO"
    assert molfile == "MOLFILE"


def test_main_predict_smiles_single_image(captured_uvicorn_run: dict[str, object]) -> None:
    main()
    client = TestClient(captured_uvicorn_run["app"])  # type: ignore[arg-type]

    image = base64.b64encode(b"fake-image-bytes").decode()
    response = client.post("/predict/smiles", json=image)

    assert response.status_code == 200
    assert response.json() == "CCO"


def test_main_predict_smiles_multiple_images(captured_uvicorn_run: dict[str, object]) -> None:
    main()
    client = TestClient(captured_uvicorn_run["app"])  # type: ignore[arg-type]

    image = base64.b64encode(b"fake-image-bytes").decode()
    response = client.post("/predict/smiles", json=[image, image])

    assert response.status_code == 200
    assert response.json() == "CCO.CCO"


def test_main_predict_molfile_single_image(captured_uvicorn_run: dict[str, object]) -> None:
    main()
    client = TestClient(captured_uvicorn_run["app"])  # type: ignore[arg-type]

    image = base64.b64encode(b"fake-image-bytes").decode()
    response = client.post("/predict/molfile", json=image)

    assert response.status_code == 200
    combined = Chem.MolFromMolBlock(response.json())
    single = Chem.MolFromMolBlock(PREDICTED_MOLFILE)
    assert combined.GetNumAtoms() == single.GetNumAtoms()


def test_main_predict_molfile_multiple_images(captured_uvicorn_run: dict[str, object]) -> None:
    main()
    client = TestClient(captured_uvicorn_run["app"])  # type: ignore[arg-type]

    image = base64.b64encode(b"fake-image-bytes").decode()
    response = client.post("/predict/molfile", json=[image, image])

    assert response.status_code == 200
    combined = Chem.MolFromMolBlock(response.json())
    single_atoms = Chem.MolFromMolBlock(PREDICTED_MOLFILE).GetNumAtoms()
    assert combined.GetNumAtoms() == single_atoms * 2


def test_main_starts_uvicorn_with_expected_args(captured_uvicorn_run: dict[str, object]) -> None:
    main()
    assert captured_uvicorn_run["kwargs"] == {"host": "0.0.0.0", "port": 8001}  # noqa: S104
