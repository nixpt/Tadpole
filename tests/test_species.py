import tempfile
import unittest
from pathlib import Path

from tadpole.multicellular import TriCellSpecies


class TriCellSpeciesTests(unittest.TestCase):
    def test_copy_text_file(self):
        species = TriCellSpecies.default()

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            src = tmpdir / "input.txt"
            dst = tmpdir / "output.txt"
            src.write_text("hello tri-cell", encoding="utf-8")

            result = species.process_text_file(str(src), str(dst), task="copy")

            self.assertEqual(dst.read_text(encoding="utf-8"), "hello tri-cell")
            self.assertEqual(result.input_text, "hello tri-cell")
            self.assertEqual(result.output_text, "hello tri-cell")
            self.assertEqual(result.task, "copy")
            self.assertEqual(species.brain.state.cell_fate, "brain")
            self.assertEqual(species.brain.genome.primary_skill, "brain")
            self.assertTrue(species.reader.has_skill("read_text"))
            self.assertTrue(species.writer.has_skill("write_text"))

    def test_annotate_text_file(self):
        species = TriCellSpecies.default()

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            src = tmpdir / "input.txt"
            dst = tmpdir / "output.txt"
            src.write_text("line one", encoding="utf-8")

            result = species.process_text_file(str(src), str(dst), task="annotate")

            self.assertIn("# processed by tri-cell-species", dst.read_text(encoding="utf-8"))
            self.assertEqual(result.task, "annotate")


if __name__ == "__main__":
    unittest.main()
