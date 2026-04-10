import unittest

from tadpole.cell import DigitalCell
from tadpole.differentiation import default_programs


class DifferentiationTests(unittest.TestCase):
    def test_apply_program_changes_fate_and_accessibility(self):
        cell = DigitalCell.default()
        program = default_programs()["metabolic"]

        result = program.apply(cell)

        self.assertEqual(cell.state.cell_fate, "metabolic-specialist")
        self.assertIn("gene.energy_sense", cell.state.locked_clusters)
        self.assertEqual(result["fate"], "metabolic-specialist")
        self.assertGreater(cell.state.epigenetic_accessibility["gene.energy_sense"], 0.0)

    def test_suppressed_clusters_reduce_expression(self):
        cell = DigitalCell.default()
        cell.differentiate(
            fate="stress-response",
            accessibility={"gene.energy_sense": 0.0},
            suppressed_clusters=["gene.energy_sense"],
        )

        result = cell.step({"signal.glucose": 1.0})

        self.assertNotIn("gene.energy_sense", result.active_genes)
        self.assertEqual(cell.state.cell_fate, "stress-response")


if __name__ == "__main__":
    unittest.main()
