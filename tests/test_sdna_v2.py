import unittest

from tadpole.cell import DigitalCell
from tadpole.sdna_v2 import SDNAV2


class SDNAV2Tests(unittest.TestCase):
    def test_from_cell_builds_layered_snapshot(self):
        cell = DigitalCell.default()
        cell.step({"signal.glucose": 1.0})

        sdna = SDNAV2.from_cell(cell)

        self.assertEqual(sdna.ancestor, "GuppyLM")
        self.assertEqual(sdna.genome_name, "tadpole-cell")
        self.assertGreaterEqual(len(sdna.genes), 3)
        self.assertGreaterEqual(len(sdna.organelles), 4)
        self.assertGreaterEqual(len(sdna.expression.transcripts), 0)
        self.assertEqual(sdna.cell.age, 1)

    def test_round_trip_serialization(self):
        sdna = SDNAV2.from_cell(DigitalCell.default())
        restored = SDNAV2.from_dict(sdna.to_dict())

        self.assertEqual(restored.genome_name, sdna.genome_name)
        self.assertEqual(restored.ancestor, sdna.ancestor)
        self.assertEqual(restored.summary()["gene_count"], sdna.summary()["gene_count"])


if __name__ == "__main__":
    unittest.main()
