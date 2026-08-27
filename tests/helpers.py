"""Shared helpers for molscribe_server tests."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

ETHANOL_SMILES = "CCO"


def make_2d_mol(smiles: str = ETHANOL_SMILES) -> Chem.Mol:
    """Build an RDKit molecule with 2D coordinates."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)  # type: ignore[attr-defined]
    return mol


def make_2d_molblock(smiles: str = ETHANOL_SMILES) -> str:
    """Build a Molfile block for a molecule with 2D coordinates."""
    return Chem.MolToMolBlock(make_2d_mol(smiles))
