"""Run API server for MolScribe."""

from __future__ import annotations

import base64
import logging
import tempfile
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from molscribe import MolScribe
    from rdkit import Chem
__version__ = "0.1.1"

logger = logging.getLogger("molscribe-server")

BBox = tuple[float, float, float, float]


def get_bounding_box(mol: Chem.Mol) -> BBox:
    """Get the bounding box of a RDKit molecule."""
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    if not all(z == 0 for _, _, z in positions):
        raise ValueError("Molecule must be 2D.")
    x_min = min(x for x, _, _ in positions)
    x_max = max(x for x, _, _ in positions)
    y_min = min(y for _, y, _ in positions)
    y_max = max(y for _, y, _ in positions)
    return x_min, x_max, y_min, y_max


def translate_mol(mol: Chem.Mol, other_bbox: BBox, mode: Literal["right", "below"]) -> Chem.Mol:
    """Translate a RDKit molecule right or below the bounding box."""
    from rdkit.Geometry import Point3D

    this_bbox = get_bounding_box(mol)
    if mode == "right":
        # prev xmax - this xmin => prev xmax == this xmin
        tr_x = other_bbox[1] - this_bbox[0] + 1  # for buffer
        # prev ymin - this ymin => prev ymin == this ymin => same top boundary
        tr_y = other_bbox[2] - this_bbox[2]
    else:
        # prev xmin - this xmin => prev xmin == this xmin => same left boundary
        tr_x = other_bbox[0] - this_bbox[0]
        # prev ymax - this ymax => prev ymax == this ymin
        tr_y = other_bbox[3] - this_bbox[2] + 1  # for buffer
    translate_point = Point3D(tr_x, tr_y, 0.0)
    conf = mol.GetConformer()
    for ix in range(mol.GetNumAtoms()):
        old_atom_ps: Point3D = conf.GetAtomPosition(ix)
        new_atom_ps = old_atom_ps + translate_point
        conf.SetAtomPosition(ix, new_atom_ps)
    return mol


def base64_to_mol_data(model: MolScribe, base64_string: str) -> tuple[str, str]:
    """Predict the SMILES string and Molfile from a base64 string.

    Args:
        model: Initialized MolScribe model.
        base64_string: Base64 string of the image to predict.

    Returns:
        Tuple of SMILES string and the Molfile.
    """
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        f.write(base64.b64decode(base64_string))
        f.seek(0)
        prediction = model.predict_image_file(f.name)
        return prediction["smiles"], prediction["molfile"]


def main() -> None:
    """Main entrypoint."""
    import torch
    import uvicorn
    from fastapi import FastAPI
    from huggingface_hub import hf_hub_download
    from molscribe import MolScribe
    from rdkit import Chem

    logging.basicConfig(level=logging.INFO)

    logger.info("Loading model...")
    ckpt_path = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m.pth")

    logger.info("Initializing model...")
    device = torch.device("cpu")
    model = MolScribe(ckpt_path, device)

    logger.info("Initializing app...")
    app = FastAPI()

    @app.post("/predict/smiles")
    def predict_as_smiles(images: str | list[str]) -> str:
        """Predict the SMILES string from base64 strings."""
        if isinstance(images, str):
            images = [images]
        smiles = [base64_to_mol_data(model, image)[0] for image in images]
        return ".".join(smiles)

    @app.post("/predict/molfile")
    def predict_as_molfile(images: str | list[str]) -> str:
        """Predict the Molfile from base64 strings."""
        if isinstance(images, str):
            images = [images]

        molfiles = [base64_to_mol_data(model, image)[1] for image in images]
        mols = [Chem.MolFromMolBlock(molfile) for molfile in molfiles]
        final = mols[0]
        bbox = get_bounding_box(final)
        mode: Literal["right", "below"] = "right"
        for raw_mol in mols[1:]:
            # put the next mol next to the previous one or below if second column
            mol = translate_mol(raw_mol, bbox, mode)
            final = Chem.CombineMols(final, mol)
            mode = "below" if mode == "right" else "right"
            bbox = get_bounding_box(final)
        return Chem.MolToMolBlock(final)

    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)  # noqa: S104


if __name__ == "__main__":
    main()
