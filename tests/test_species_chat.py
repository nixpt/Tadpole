import unittest

from tadpole.config import GuppyConfig, TadpoleConfig
from tadpole.inference import species_chat_prompt
from tadpole.multicellular import DigitalMonkeySpecies, TriCellSpecies


class SpeciesChatTests(unittest.TestCase):
    def test_guppy_config_alias(self):
        self.assertIs(GuppyConfig, TadpoleConfig)

    def test_species_chat_prompt_mentions_species_shape(self):
        prompt = species_chat_prompt(TriCellSpecies.default())
        self.assertIn("tri-cell-species", prompt)
        self.assertIn("Cell count: 3", prompt)
        self.assertIn("brain=brain", prompt)

    def test_monkey_species_chat_prompt_mentions_cell_count(self):
        prompt = species_chat_prompt(DigitalMonkeySpecies.training_layout(128))
        self.assertIn("digital-monkey-cns-128", prompt)
        self.assertIn("Cell count: 128", prompt)
        self.assertIn("cns=", prompt)
        self.assertIn("Training sources:", prompt)
        self.assertIn("report_to_cns", prompt)


if __name__ == "__main__":
    unittest.main()
