import unittest

from tadpole.cell import CellGenome, DigitalCell, Gene


class DigitalCellTests(unittest.TestCase):
    def test_default_cell_exports_cell_state(self):
        cell = DigitalCell.default()
        data = cell.to_dict()

        self.assertEqual(data["genome"]["name"], "tadpole-cell")
        self.assertEqual(data["genome"]["primary_skill"], "generalist")
        self.assertIn("metabolism", data["genome"]["skills"])
        self.assertIn("mitochondria", data["state"]["organelles"])
        self.assertEqual(len(data["genome"]["genes"]), 3)

    def test_step_activates_gene_and_consumes_energy(self):
        genome = CellGenome(
            name="test-cell",
            ancestor="GuppyLM",
            genes=[
                Gene(
                    id="gene.growth",
                    product="protein.growth",
                    trigger="signal.growth",
                    activation_threshold=0.5,
                    transcript_rate=1.0,
                    translation_rate=1.0,
                    transcription_cost=0.1,
                    translation_cost=0.1,
                    transcript_decay=0.0,
                    protein_decay=0.0,
                )
            ],
        )
        cell = DigitalCell(genome=genome)
        before = cell.state.metabolites["ATP"]
        result = cell.step({"signal.growth": 1.0})

        self.assertIn("gene.growth", result.active_genes)
        self.assertGreater(cell.state.transcripts["protein.growth"], 0.0)
        self.assertGreater(cell.state.proteins["protein.growth"], 0.0)
        self.assertGreater(cell.state.metabolites["ATP"], before)
        self.assertEqual(cell.state.age, 1)

    def test_round_trip_serialization(self):
        cell = DigitalCell.default()
        cell.step({"signal.growth": 1.0, "signal.glucose": 1.0})

        restored = DigitalCell.from_dict(cell.to_dict())

        self.assertEqual(restored.genome.name, cell.genome.name)
        self.assertEqual(restored.state.age, cell.state.age)
        self.assertEqual(restored.state.transcripts, cell.state.transcripts)

    def test_assign_skill_updates_primary_skill(self):
        cell = DigitalCell.default()
        cell.assign_skill("reader")

        self.assertTrue(cell.has_skill("reader"))
        self.assertEqual(cell.genome.primary_skill, "reader")
        self.assertIn("reader", cell.genome.skills)


if __name__ == "__main__":
    unittest.main()
