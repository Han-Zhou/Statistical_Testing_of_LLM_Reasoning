"""CPU-safe diagnostics for the registered Qwen FP8 loading safeguards."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from models.adapters.qwen_adapter import QwenAdapter, QwenFp8Adapter
from models.core_models.llm import (
    LLM,
    _correct_qwen_fp8_skip_modules,
    _validate_qwen_fp8_loading_info,
)


class QwenFp8QuantizationConfigTests(unittest.TestCase):
    def test_corrects_dictionary_config_precisely(self):
        quantization_config = {
            "modules_to_not_convert": [
                "model.layers.0.mlp.gate",
                "model.layers.0.mlp.gate_proj",
                "model.layers.0.self_attn",
            ]
        }

        removed_count = _correct_qwen_fp8_skip_modules(quantization_config)

        self.assertEqual(removed_count, 1)
        self.assertEqual(
            quantization_config["modules_to_not_convert"],
            [
                "model.layers.0.mlp.gate_proj",
                "model.layers.0.self_attn",
            ],
        )

    def test_corrects_object_config(self):
        quantization_config = SimpleNamespace(
            modules_to_not_convert=[
                "model.layers.1.mlp.gate",
                "model.visual",
            ]
        )

        removed_count = _correct_qwen_fp8_skip_modules(quantization_config)

        self.assertEqual(removed_count, 1)
        self.assertEqual(
            quantization_config.modules_to_not_convert,
            ["model.visual"],
        )

    def test_rejects_malformed_config(self):
        with self.assertRaisesRegex(RuntimeError, "quantization_config"):
            _correct_qwen_fp8_skip_modules(None)
        with self.assertRaisesRegex(RuntimeError, "modules_to_not_convert"):
            _correct_qwen_fp8_skip_modules(SimpleNamespace())
        with self.assertRaisesRegex(RuntimeError, "list of module names"):
            _correct_qwen_fp8_skip_modules(
                {"modules_to_not_convert": "model.layers.0.mlp.gate"}
            )


class QwenFp8LoadingInfoTests(unittest.TestCase):
    def test_accepts_loading_info_without_rejected_scales(self):
        _validate_qwen_fp8_loading_info(
            {
                "unexpected_keys": {"model.unrelated_buffer"},
                "missing_keys": set(),
                "mismatched_keys": set(),
            }
        )

    def test_rejects_scale_keys_in_every_error_category(self):
        for category in ("unexpected_keys", "missing_keys", "mismatched_keys"):
            with self.subTest(category=category):
                entry = "model.layers.0.mlp.gate_proj.weight_scale_inv"
                if category == "mismatched_keys":
                    entry = (entry, (1,), (2,))
                with self.assertRaisesRegex(RuntimeError, category):
                    _validate_qwen_fp8_loading_info({category: {entry}})

    def test_rejects_malformed_loading_info(self):
        with self.assertRaisesRegex(RuntimeError, "malformed loading information"):
            _validate_qwen_fp8_loading_info(None)
        with self.assertRaisesRegex(RuntimeError, "malformed missing_keys"):
            _validate_qwen_fp8_loading_info({"missing_keys": "not-a-list"})


class QwenFp8LoaderTests(unittest.TestCase):
    def test_input_device_comes_from_token_embeddings(self):
        wrapper = LLM.__new__(LLM)
        wrapper.model_name = "test"
        wrapper.model = MagicMock()
        wrapper.model.get_input_embeddings.return_value = SimpleNamespace(
            weight=SimpleNamespace(device="cuda:3")
        )

        self.assertEqual(wrapper.input_device, "cuda:3")

    def test_specialized_loader_uses_fp8_safe_arguments(self):
        wrapper = LLM.__new__(LLM)
        wrapper.model_name = "qwen_fp8"
        config = SimpleNamespace(
            quantization_config={
                "modules_to_not_convert": [
                    "model.layers.0.mlp.gate",
                    "model.visual",
                ]
            }
        )
        loaded_model = MagicMock()
        loaded_model.model = SimpleNamespace(visual=object())
        loading_info = {
            "unexpected_keys": [],
            "missing_keys": [],
            "mismatched_keys": [],
        }

        with (
            patch("models.core_models.llm.torch.cuda.is_available", return_value=True),
            patch("models.core_models.llm.torch.cuda.empty_cache") as empty_cache,
            patch("models.core_models.llm.gc.collect") as collect,
            patch("transformers.AutoConfig.from_pretrained", return_value=config) as load_config,
            patch("models.core_models.llm.AutoTokenizer.from_pretrained") as load_tokenizer,
            patch(
                "transformers.Qwen3_5ForConditionalGeneration.from_pretrained",
                return_value=(loaded_model, loading_info),
            ) as load_model,
        ):
            wrapper._load_qwen_fp8_model("Qwen/Qwen3.8-27B-FP8")

        load_config.assert_called_once_with(
            "Qwen/Qwen3.8-27B-FP8",
            trust_remote_code=True,
        )
        load_tokenizer.assert_called_once_with(
            "Qwen/Qwen3.8-27B-FP8",
            trust_remote_code=True,
        )
        load_model.assert_called_once_with(
            "Qwen/Qwen3.8-27B-FP8",
            config=config,
            device_map="auto",
            dtype="auto",
            output_loading_info=True,
            trust_remote_code=True,
            allow_all_kernels=True,
        )
        self.assertEqual(
            config.quantization_config["modules_to_not_convert"],
            ["model.visual"],
        )
        loaded_model.eval.assert_called_once_with()
        self.assertIsNone(loaded_model.model.visual)
        collect.assert_called_once_with()
        empty_cache.assert_called_once_with()

    def test_specialized_loader_requires_cuda_before_checkpoint_access(self):
        wrapper = LLM.__new__(LLM)
        wrapper.model_name = "qwen_fp8"
        with (
            patch("models.core_models.llm.torch.cuda.is_available", return_value=False),
            patch("models.core_models.llm.AutoTokenizer.from_pretrained") as load_tokenizer,
        ):
            with self.assertRaisesRegex(RuntimeError, "requires a CUDA GPU"):
                wrapper._load_qwen_fp8_model("Qwen/Qwen3.8-27B-FP8")
        load_tokenizer.assert_not_called()


class QwenFp8AdapterTests(unittest.TestCase):
    @patch("models.adapters.qwen_adapter.LLM")
    def test_fp8_adapter_selects_checkpoint_and_enables_thinking(self, llm_class):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "rendered prompt"
        llm_class.return_value = SimpleNamespace(tokenizer=tokenizer)

        adapter = QwenFp8Adapter()
        rendered = adapter.render_prompt([{"role": "user", "content": "question"}])

        llm_class.assert_called_once_with(
            model_name="qwen_fp8",
            attention_implementation=None,
        )
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "question"}],
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
            enable_thinking=True,
        )
        self.assertEqual(rendered, "rendered prompt")

    @patch("models.adapters.qwen_adapter.LLM")
    def test_existing_qwen_prompt_behavior_is_unchanged(self, llm_class):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "rendered prompt"
        llm_class.return_value = SimpleNamespace(tokenizer=tokenizer)

        adapter = QwenAdapter()
        adapter.render_prompt([{"role": "user", "content": "question"}])

        llm_class.assert_called_once_with(
            model_name="qwen",
            attention_implementation="flash_attention_2",
        )
        self.assertNotIn(
            "enable_thinking",
            tokenizer.apply_chat_template.call_args.kwargs,
        )


if __name__ == "__main__":
    unittest.main()
