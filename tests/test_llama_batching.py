import copy
import unittest

import torch
from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM

from models.core_models.llm import LLM


class _CharTokenizer:
    """Minimal tokenizer for CPU-only batching tests."""

    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [2 + (ord(char) % 100) for char in text]

    def __call__(
        self,
        texts: str | list[str],
        return_tensors: str | None = None,
        padding: bool = False,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        del return_tensors, add_special_tokens, kwargs
        rows = [self._encode(texts)] if isinstance(texts, str) else [
            self._encode(text) for text in texts
        ]
        width = max(len(row) for row in rows)
        input_ids = []
        attention_mask = []
        for row in rows:
            pad_len = width - len(row) if padding else 0
            if self.padding_side == "left":
                input_ids.append([self.pad_token_id] * pad_len + row)
                attention_mask.append([0] * pad_len + [1] * len(row))
            else:
                input_ids.append(row + [self.pad_token_id] * pad_len)
                attention_mask.append([1] * len(row) + [0] * pad_len)
        return BatchEncoding(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        )


class LlamaCachedBatchingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            pad_token_id=0,
            eos_token_id=1,
        )
        model = LlamaForCausalLM(config).eval()

        cls.tokenizer = _CharTokenizer()
        cls.llm = LLM.__new__(LLM)
        cls.llm.model_name = "tiny-llama"
        cls.llm.model = model
        cls.llm.tokenizer = cls.tokenizer
        cls.llm.debug_nocache = False

    def test_unequal_cached_deltas_match_serial_and_return_compact_caches(self):
        prefix = "abc"
        deltas = ["de", "fghi"]
        prefix_inputs = self.tokenizer(prefix, return_tensors="pt")

        with torch.inference_mode():
            prefix_cache = self.llm.model(
                **prefix_inputs,
                use_cache=True,
                return_dict=True,
            ).past_key_values

            batch_outputs = self.llm.forward_batch_with_cache(
                delta_texts=deltas,
                cache=copy.deepcopy(prefix_cache),
                cache_seq_len=prefix_inputs.input_ids.shape[1],
                return_llm_output=True,
            )

            for delta, batch_output in zip(deltas, batch_outputs):
                full_inputs = self.tokenizer(prefix + delta, return_tensors="pt")
                serial_logits = self.llm.model(
                    **full_inputs,
                    use_cache=False,
                    return_dict=True,
                ).logits[:, len(prefix):, :]

                torch.testing.assert_close(
                    batch_output.logits,
                    serial_logits,
                    rtol=1e-5,
                    atol=1e-5,
                )
                self.assertEqual(
                    batch_output.past_key_values.get_seq_length(),
                    len(prefix) + len(delta),
                )

                # Reusing the compacted cache for one more token must also
                # produce the same logits as an uncached serial forward pass.
                next_inputs = self.tokenizer("z", return_tensors="pt")
                cached_length = len(prefix) + len(delta)
                cached_next_logits = self.llm.model(
                    input_ids=next_inputs.input_ids,
                    attention_mask=torch.ones(1, cached_length + 1, dtype=torch.long),
                    position_ids=torch.tensor([[cached_length]], dtype=torch.long),
                    past_key_values=copy.deepcopy(batch_output.past_key_values),
                    use_cache=False,
                    return_dict=True,
                ).logits[:, -1, :]
                serial_next_inputs = self.tokenizer(
                    prefix + delta + "z",
                    return_tensors="pt",
                )
                serial_next_logits = self.llm.model(
                    **serial_next_inputs,
                    use_cache=False,
                    return_dict=True,
                ).logits[:, -1, :]
                torch.testing.assert_close(
                    cached_next_logits,
                    serial_next_logits,
                    rtol=1e-5,
                    atol=1e-5,
                )


if __name__ == "__main__":
    unittest.main()
