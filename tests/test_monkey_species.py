import tempfile
import unittest
from pathlib import Path

from tadpole.multicellular import DigitalMonkeySpecies


class DigitalMonkeySpeciesTests(unittest.TestCase):
    def test_default_species_has_16_cells(self):
        species = DigitalMonkeySpecies.default()

        self.assertEqual(len(species.cells), 16)
        self.assertEqual(species.brain.genome.primary_skill, "brain")
        self.assertTrue(species.brain.has_skill("brain"))
        self.assertGreaterEqual(len(species.readers), 3)
        self.assertGreaterEqual(len(species.writers), 3)

    def test_training_layout_uses_dataset_roles(self):
        species = DigitalMonkeySpecies.training_layout(128)

        self.assertEqual(len(species.cells), 128)
        self.assertEqual(species.name, "digital-monkey-cns-128")
        self.assertEqual(species.brain.genome.primary_skill, "cns")
        self.assertIn("dataset=Roman1111111/claude-opus-4.6-10000x", species.brain.genome.notes)
        self.assertGreaterEqual(len(species._cells_with_skill("cns")), 8)
        self.assertGreaterEqual(len(species._cells_with_skill("sensory_language")), 18)
        self.assertGreaterEqual(len(species._cells_with_skill("memory_write")), 12)
        self.assertGreaterEqual(len(species._cells_with_skill("motor_response")), 20)
        self.assertGreaterEqual(len(species._cells_with_skill("motor_action")), 14)
        sensory = species._cells_with_skill("sensory_language")[0]
        self.assertEqual(sensory.genome.skills, ["sensory_language", "report_to_cns"])

    def test_copy_text_file(self):
        species = DigitalMonkeySpecies.default()

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            src = tmpdir / "input.txt"
            dst = tmpdir / "output.txt"
            src.write_text("monkey file", encoding="utf-8")

            result = species.process_text_file(str(src), str(dst), task="copy")

            self.assertEqual(dst.read_text(encoding="utf-8"), "monkey file")
            self.assertEqual(result.cell_count, 16)
            self.assertEqual(result.task, "copy")
            self.assertEqual(result.brain_summary["primary_skill"], "brain")
            self.assertGreaterEqual(len(result.reader_summaries), 3)
            self.assertGreaterEqual(len(result.writer_summaries), 3)

    def test_uppercase_text_file(self):
        species = DigitalMonkeySpecies.default()

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            src = tmpdir / "input.txt"
            dst = tmpdir / "output.txt"
            src.write_text("mixed Case", encoding="utf-8")

            result = species.process_text_file(str(src), str(dst), task="uppercase")

            self.assertEqual(dst.read_text(encoding="utf-8"), "MIXED CASE")
            self.assertEqual(result.output_text, "MIXED CASE")


if __name__ == "__main__":
    unittest.main()
