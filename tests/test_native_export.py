import json
import tempfile
import unittest
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from servedistill.native_export import repackage_native


class NativeExportTests(unittest.TestCase):
    def test_native_wrapper_and_index_keep_language_weights_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source", root / "export"
            source.mkdir()
            output.mkdir()
            configuration = {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {"hidden_size": 4},
                "vision_config": {"hidden_size": 8},
            }
            (source / "config.json").write_text(json.dumps(configuration))
            weights = {"model.embed_tokens.weight": torch.arange(16).reshape(4, 4).bfloat16(),
                       "lm_head.weight": torch.ones(4, 4).bfloat16()}
            save_file(weights, output / "model-00001-of-00001.safetensors")
            index = {"weight_map": {key: "model-00001-of-00001.safetensors" for key in weights}}
            (output / "model.safetensors.index.json").write_text(json.dumps(index))
            report = repackage_native(source, output)
            actual = load_file(output / "model-00001-of-00001.safetensors")
            torch.testing.assert_close(actual["model.language_model.embed_tokens.weight"],
                                       weights["model.embed_tokens.weight"], rtol=0, atol=0)
            torch.testing.assert_close(actual["lm_head.weight"], weights["lm_head.weight"], rtol=0, atol=0)
            keys = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
            self.assertEqual(set(keys), set(actual))
            self.assertEqual(json.loads((output / "config.json").read_text())["text_config"],
                             configuration["text_config"])
            self.assertEqual(report["language_tensors"], 2)
            self.assertFalse(report["vision_weights_included"])


if __name__ == "__main__":
    unittest.main()
