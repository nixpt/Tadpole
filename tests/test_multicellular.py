import unittest

from tadpole.cell import DigitalCell
from tadpole.multicellular import Organism, Tissue


class MulticellularTests(unittest.TestCase):
    def test_tissue_divides_high_energy_cells(self):
        cell = DigitalCell.default()
        cell.state.metabolites["ATP"] = 100.0
        cell.state.health = 1.0
        tissue = Tissue(name="growth", cells=[cell], division_threshold=0.8)

        result = tissue.step({"signal.glucose": 1.0})

        self.assertGreaterEqual(result.divisions, 1)
        self.assertGreaterEqual(len(tissue.cells), 2)

    def test_tissue_apoptosis_removes_weak_cells(self):
        cell = DigitalCell.default()
        cell.state.health = 0.0
        cell.state.metabolites["ATP"] = 0.0
        cell.state.metabolites["glucose"] = 0.0
        cell.state.metabolites["oxygen"] = 0.0
        cell.state.organelles["mitochondria"].health = 0.0
        tissue = Tissue(name="damage", cells=[cell], apoptosis_threshold=0.2)

        result = tissue.step({})

        self.assertEqual(result.apoptosis, 1)
        self.assertEqual(len(tissue.cells), 0)

    def test_organism_summary(self):
        organism = Organism(name="cluster", tissues={"core": Tissue(name="core", cells=[DigitalCell.default()])})

        summary = organism.summary()

        self.assertEqual(summary["name"], "cluster")
        self.assertEqual(summary["tissues"]["core"]["cells"], 1)


if __name__ == "__main__":
    unittest.main()
