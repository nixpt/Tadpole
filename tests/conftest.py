from pathlib import Path
import sys

_symbiome_python = Path(__file__).resolve().parents[2] / "symbiome" / "python"
if _symbiome_python.exists() and str(_symbiome_python) not in sys.path:
    sys.path.insert(0, str(_symbiome_python))
