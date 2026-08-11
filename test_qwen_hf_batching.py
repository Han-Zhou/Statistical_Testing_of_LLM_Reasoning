import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from transformers import BatchEncoding
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from config import GenerationConfig
from models.adapters.qwen_adapter import QwenAdapter, QwenScorer
from models.core_models.llm import LLM


class _GenerationTokenizerStub:
    eos_token_id = 0
    all_special_tokens: list[str] = []

    decoded = {
        10: "prompt<think>hidden 0</think>\n",
        11: "prompt<think>hidden 1</think>\n",
        20: "phase 2 sample 0",
        21: "phase 2 sample 1",
        30: "<think>hidden 0</think>\nvisible sample 0",
        31: "<think>hidden 1</think>\nvisible sample 1",
    }

    def __call__(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return SimpleNamespace(input_ids=[1])

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
        return "".join(self.decoded[int(token_id)] for token_id in ids)


class _PaddingTokenizerStub:
    eos_token_id = 0

    def __init__(self):
        self.prompt_lengths = {"short": 2, "long": 4}

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(input_ids=[1] * self.prompt_lengths[text])

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
        return ",".join(str(int(token_id)) for token_id in ids)


class _TinyQwenTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "right"

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [3 + (ord(char) % 100) for char in text]

    def __call__(
        self,
        texts: str | list[str],
        return_tensors: str | None = None,
        padding: bool = False,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **kwargs,
    ):
        del return_tensors, add_special_tokens, kwargs
        rows = [self._encode(texts)] if isinstance(texts, str) else [
            self._encode(text) for text in texts
        ]
        if return_offsets_mapping:
            return {
                "offset_mapping": [
                    (index, index + 1) for index in range(len(rows[0]))
                ]
            }

        width = max(len(row) for row in rows)
        input_ids = []
        attention_mask = []
        for row in rows:
            pad_length = width - len(row) if padding else 0
            if self.padding_side == "left":
                input_ids.append([self.pad_token_id] * pad_length + row)
                attention_mask.append([0] * pad_length + [1] * len(row))
            else:
                input_ids.append(row + [self.pad_token_id] * pad_length)
                attention_mask.append([1] * len(row) + [0] * pad_length)
        return BatchEncoding(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        )

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "x" * len(token_ids)


class QwenHfBatchingTest(unittest.TestCase):
    def test_qwen_is_enabled_by_experimental_batch_flag(self):
        config = GenerationConfig.__new__(GenerationConfig)
        config.model = "qwen"
        config.experimental_llama_batch = True

        self.assertTrue(config.batching_enabled)

    def test_scorer_batches_full_prompts_without_shared_cache(self):
        logits = torch.arange(24).reshape(2, 12)
        model = SimpleNamespace(
            forward_batch_last_logits=Mock(return_value=logits)
        )
        scorer = QwenScorer(model)

        results = scorer.forward_batch_confidence(
            ["prompt a", "prompt b"],
            shared_cache=object(),
        )

        model.forward_batch_last_logits.assert_called_once_with(
            ["prompt a", "prompt b"]
        )
        self.assertEqual(len(results), 2)
        torch.testing.assert_close(results[0], logits[0])
        torch.testing.assert_close(results[1], logits[1])

    def test_serial_scorers_return_scores_and_debug_info(self):
        class _ConfidenceTokenizer:
            def __call__(self, text, add_special_tokens=False):
                del add_special_tokens
                if text == " True":
                    return SimpleNamespace(input_ids=[1])
                if text == " False":
                    return SimpleNamespace(input_ids=[2])
                return SimpleNamespace(input_ids=[3])

        logits = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12)
        model = SimpleNamespace(
            tokenizer=_ConfidenceTokenizer(),
            align_cache=Mock(return_value=None),
            forward=Mock(return_value=SimpleNamespace(logits=logits)),
        )
        scorer = QwenScorer(model)

        indirect_scores, indirect_debug = scorer.forward_indirect(
            "indirect prompt",
            whole_cache=object(),
        )
        verbal_scores, verbal_debug = scorer.forward_verbal(
            "verbal prompt",
            whole_cache=object(),
        )

        self.assertEqual(set(indirect_scores), {"True", "False"})
        self.assertEqual(indirect_debug, {})
        self.assertEqual(verbal_debug, {})
        self.assertEqual(len(verbal_scores), 101)

    def test_generate_batch_helper_batches_all_three_qwen_phases(self):
        phase_1 = SimpleNamespace(
            sequences=torch.tensor([[10, 0], [11, 0]])
        )
        phase_2 = SimpleNamespace(
            sequences=torch.tensor([[20, 0], [21, 0]])
        )
        phase_3 = SimpleNamespace(
            sequences=torch.tensor([[30, 0], [31, 0]])
        )
        forwarded = ["forwarded 0", "forwarded 1"]

        adapter = QwenAdapter.__new__(QwenAdapter)
        adapter.model = SimpleNamespace(
            tokenizer=_GenerationTokenizerStub(),
            generate_multi_sequence=Mock(return_value=phase_1),
            generate_batch=Mock(side_effect=[phase_2, phase_3]),
            forward_batch=Mock(return_value=forwarded),
        )

        outputs = adapter.generate_batch_helper(
            prompt="prompt",
            max_tokens=128,
            cache=object(),
            temperature=0.7,
            num_sequences=2,
        )

        self.assertEqual(outputs, forwarded)
        adapter.model.generate_multi_sequence.assert_called_once_with(
            prompt="prompt",
            max_tokens=128,
            cache=None,
            temperature=0.7,
            num_sequences=2,
            stop_strings=["</think>"],
        )

        phase_2_call, phase_3_call = adapter.model.generate_batch.call_args_list
        self.assertEqual(
            phase_2_call.kwargs["prompts"],
            [
                "prompt<think>hidden 0</think>\nLet's think step by step. \nStep 1: ",
                "prompt<think>hidden 1</think>\nLet's think step by step. \nStep 1: ",
            ],
        )
        self.assertEqual(
            phase_3_call.kwargs["prompts"],
            [
                "phase 2 sample 0\nThe answer is \\boxed{",
                "phase 2 sample 1\nThe answer is \\boxed{",
            ],
        )
        adapter.model.forward_batch.assert_called_once_with(
            ["visible sample 0", "visible sample 1"],
            return_llm_output=True,
            return_cache=False,
        )

    def test_forward_pass_batch_helper_ignores_qwen_hybrid_cache(self):
        forwarded = ["output a", "output b"]
        adapter = QwenAdapter.__new__(QwenAdapter)
        adapter.model = SimpleNamespace(
            forward_batch=Mock(return_value=forwarded)
        )

        outputs = adapter.forward_pass_batch_helper(
            ["prompt a", "prompt b"],
            cache=object(),
            cache_seq_len=42,
        )

        self.assertEqual(outputs, forwarded)
        adapter.model.forward_batch.assert_called_once_with(
            ["prompt a", "prompt b"],
            return_llm_output=True,
            return_cache=False,
        )

    def test_public_batch_entry_points_do_not_try_to_align_hybrid_cache(self):
        adapter = QwenAdapter.__new__(QwenAdapter)
        adapter.render_prompt = Mock(
            side_effect=lambda messages: messages[0]["content"]
        )
        adapter.align_cache = Mock(
            side_effect=AssertionError("Qwen batch cache must not be aligned")
        )
        adapter.generate_batch_helper = Mock(return_value=["generated"])
        adapter.forward_pass_batch_helper = Mock(return_value=["forwarded"])
        adapter.process_generation_output = Mock(side_effect=lambda output: output)

        generated = adapter.generate_batch(
            [{"role": "user", "content": "generation prompt"}],
            max_tokens=64,
            cache=object(),
            temperature=0.8,
            num_sequences=1,
        )
        forwarded = adapter.forward_pass_batch(
            [[{"role": "assistant", "content": "forward prompt"}]],
            cache=object(),
        )

        self.assertEqual(generated, ["generated"])
        self.assertEqual(forwarded, ["forwarded"])
        adapter.align_cache.assert_not_called()
        adapter.generate_batch_helper.assert_called_once_with(
            prompt="generation prompt",
            max_tokens=64,
            cache=None,
            temperature=0.8,
            num_sequences=1,
        )
        adapter.forward_pass_batch_helper.assert_called_once_with(
            ["forward prompt"]
        )

    def test_left_padding_and_trailing_eos_are_removed_before_decode(self):
        adapter = QwenAdapter.__new__(QwenAdapter)
        adapter.model = SimpleNamespace(tokenizer=_PaddingTokenizerStub())
        sequences = torch.tensor(
            [
                [0, 0, 10, 11, 0, 0],
                [20, 21, 22, 23, 0, 0],
            ]
        )

        texts = adapter._decode_left_padded_sequences(
            sequences,
            ["short", "long"],
        )

        self.assertEqual(texts, ["10,11", "20,21,22,23"])

    def test_cacheless_batch_output_does_not_trigger_serial_cache_forward(self):
        adapter = QwenAdapter.__new__(QwenAdapter)
        adapter.model = SimpleNamespace(
            forward=Mock(side_effect=AssertionError("unexpected serial forward"))
        )
        text = "question<|im_start|>assistant\nStep 1: reasoning"
        cot_start = text.index("<|im_start|>assistant") + len(
            "<|im_start|>assistant"
        )

        result = adapter._extract_cot(
            output_text=text,
            output_tokens=[],
            offset_mappings=[],
            cache=None,
            sequence_ids=torch.tensor([], dtype=torch.long),
            cot_start_idx=cot_start,
            answer_span=None,
        )

        self.assertIsNone(result[4])
        self.assertIsNone(result[5])
        adapter.model.forward.assert_not_called()


class QwenCoreBatchingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        config = Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            layer_types=["linear_attention", "full_attention"],
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        model = Qwen3_5ForCausalLM(config).eval()

        cls.llm = LLM.__new__(LLM)
        cls.llm.model_name = "tiny-qwen"
        cls.llm.model = model
        cls.llm.tokenizer = _TinyQwenTokenizer()
        cls.llm.debug_nocache = False

    def test_cacheless_left_padded_forward_matches_serial_qwen(self):
        prompts = ["abcd", "xy"]

        batch_outputs = self.llm.forward_batch(
            prompts,
            return_llm_output=True,
            return_cache=False,
        )
        batch_last_logits = self.llm.forward_batch_last_logits(prompts)

        for index, (prompt, batch_output) in enumerate(
            zip(prompts, batch_outputs)
        ):
            inputs = self.llm.tokenizer(prompt, return_tensors="pt")
            with torch.inference_mode():
                serial_logits = self.llm.model(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                ).logits

            self.assertIsNone(batch_output.outputs.past_key_values)
            torch.testing.assert_close(
                batch_output.outputs.logits,
                serial_logits,
                rtol=1e-5,
                atol=1e-5,
            )
            torch.testing.assert_close(
                batch_last_logits[index],
                serial_logits[0, -1],
                rtol=1e-5,
                atol=1e-5,
            )


if __name__ == "__main__":
    unittest.main()
