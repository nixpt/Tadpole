import json
import tempfile
import unittest
from pathlib import Path

from tadpole.prepare_data import prepare


class MixedPrepareTests(unittest.TestCase):
    def test_prepare_mixed_pack_writes_training_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            pack = tmpdir / "hf_mixed_starter_pack"
            pack.mkdir()

            train_rows = [
                {"text": "<|im_start|>user\nhello<|im_end|>", "source": "demo", "category": "chat"},
                {"text": "<|im_start|>user\nprint(1)<|im_end|>", "source": "demo", "category": "code"},
            ]
            eval_rows = [
                {"text": "<|im_start|>user\nwhy?<|im_end|>", "source": "demo", "category": "reasoning"}
            ]

            (pack / "train.jsonl").write_text("\n".join(json.dumps(row) for row in train_rows) + "\n", encoding="utf-8")
            (pack / "eval.jsonl").write_text("\n".join(json.dumps(row) for row in eval_rows) + "\n", encoding="utf-8")

            out = tmpdir / "prepared"
            prepare(data_dir=str(out), source="mixed", mixed_pack_dir=str(pack))

            self.assertTrue((out / "train.jsonl").exists())
            self.assertTrue((out / "eval.jsonl").exists())
            self.assertTrue((out / "tokenizer.json").exists())


if __name__ == "__main__":
    unittest.main()
