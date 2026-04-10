"""Tadpole — tiny rama-zpu assistant."""

from pathlib import Path
import sys

_symbiome_python = Path(__file__).resolve().parents[2] / "symbiome" / "python"
if _symbiome_python.exists() and str(_symbiome_python) not in sys.path:
    sys.path.insert(0, str(_symbiome_python))

from .config import TadpoleConfig, TrainConfig
from symbiome.biology import CellGenome, CellState, CellStepResult, DigitalCell, DigitalMonkeySpecies, DifferentiationProgram, Gene, MasterRegulator, MonkeyTaskResult, Organism, OrganismStepResult, Organelle, SDNACellState, SDNADevelopmentProgram, SDNAExpressionLayer, SDNAGene, SDNAOrganelle, SDNAProtein, SDNARNA, SDNAV2, TextFileTaskResult, Tissue, TissueStepResult, TriCellSpecies, default_programs, run_demo

__version__ = "0.1.0"
