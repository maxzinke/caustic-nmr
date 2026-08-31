"""Shared fixtures: paths to the bundled example structures."""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"


@pytest.fixture(scope="session")
def ubq_pdb() -> Path:
    p = EXAMPLES / "1ubq.pdb"
    assert p.exists(), f"missing example structure {p}"
    return p


@pytest.fixture(scope="session")
def af_cif() -> Path:
    cifs = sorted(EXAMPLES.glob("AF-*-model_v*.cif"))
    assert cifs, "missing AlphaFold example structure under examples/"
    return cifs[0]
