import unittest

from tadpole.inference import GuppyInference, species_cell_clusters
from tadpole.multicellular import DigitalMonkeySpecies


class MultiCellInferenceTests(unittest.TestCase):
    def test_training_layout_clusters_by_role(self):
        species = DigitalMonkeySpecies.training_layout(128)
        clusters = species_cell_clusters(species)

        self.assertEqual([cluster.role for cluster in clusters], [
            "cns",
            "sensory_language",
            "sensory_context",
            "memory_read",
            "memory_write",
            "motor_response",
            "motor_action",
            "reflex",
            "support_glia",
            "cerebellum",
        ])
        self.assertEqual(sum(cluster.count for cluster in clusters), 128)

    def test_species_chat_aggregates_cluster_outputs(self):
        species = DigitalMonkeySpecies.training_layout(128)
        engine = object.__new__(GuppyInference)
        calls = []

        def fake_chat_completion(messages, temperature=0.7, max_tokens=64, top_k=50, **kwargs):
            calls.append(messages)
            return {"choices": [{"message": {"role": "assistant", "content": f"OUT-{len(calls)}"}}]}

        engine.chat_completion = fake_chat_completion
        result = GuppyInference.chat_completion_for_species(
            engine,
            species,
            [{"role": "user", "content": "What do you know about the dataset?"}],
        )

        self.assertEqual(len(calls), 11)
        synthesis_prompt = calls[-1][0]["content"]
        self.assertIn("OUT-1", synthesis_prompt)
        self.assertIn("OUT-10", synthesis_prompt)
        self.assertEqual(result["choices"][0]["message"]["content"], "OUT-11")


if __name__ == "__main__":
    unittest.main()
