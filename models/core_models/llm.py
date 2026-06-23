import logging
import copy
from types import SimpleNamespace
import torch
from typing import Optional, Tuple
from dotenv import load_dotenv


from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, StopStringCriteria
from transformers.utils import ModelOutput
from transformers.cache_utils import DynamicCache
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
except ImportError:
    # See domain/data.py for context — transformers >=5.5.1 dropped this class.
    Qwen3_5DynamicCache = DynamicCache

from domain.data import LLMOutput, KVCache, CacheBundle
from models.core_models.registry import MODEL_HF_REGISTRY

load_dotenv()
logger = logging.getLogger(__name__)

class LLM():

    def __init__(self, model_name: str, attention_implementation: str, debug_nocache: bool = False):
        self.model_name = model_name
        self.attention_implementation = attention_implementation
        self.debug_nocache = debug_nocache
        self._load_model(attention_implementation)


    def _load_model(self, attention_implementation: str):
        # Load the model based on the model name
        logger.info(f"Loading LLM model: {self.model_name}")
        actual_model_name = MODEL_HF_REGISTRY.get(self.model_name)
        if not actual_model_name:
            raise ValueError(f"Model {self.model_name} not found in registry.")
    
        self.tokenizer = AutoTokenizer.from_pretrained(actual_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            actual_model_name,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation=attention_implementation,
        )
        logger.info(f"Model {self.model_name} loaded successfully.")


    
    def generate(
            self, 
            prompt: str, 
            max_tokens: int, 
            cache: Optional[Tuple], 
            temperature: float,
            stop_strings: list[str] | None = None
        ) -> LLMOutput:
        # Generate text based on the prompt
        # prompt SHOULD ALREADY HAVE chat template applied
        logger.info(f"Generating text with model {self.model_name}")

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        temp = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)
        # breakpoint()


        if stop_strings:
            stop_criteria = StoppingCriteriaList([
                StopStringCriteria(tokenizer=self.tokenizer, stop_strings=stop_strings),
            ])
        else:
            stop_criteria = None
        with torch.inference_mode():
            if cache is not None:
                outputs = self.model.generate(
                    **inputs,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict_in_generate=True,
                    max_new_tokens=max_tokens,
                    do_sample=(temperature > 0.0),
                    temperature=temperature if temperature > 0.0 else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                    output_logits=True,
                    output_scores=True,
                    stopping_criteria=stop_criteria,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    return_dict_in_generate=True,
                    max_new_tokens=max_tokens,
                    do_sample=(temperature > 0.0),
                    temperature=temperature if temperature > 0.0 else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                    output_logits=True,
                    output_scores=True,
                    stopping_criteria=stop_criteria,
                )

        output_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        # offsets_mapping requires a fast tokenizer
        full_offsets = self.tokenizer(
            output_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )["offset_mapping"]

        return LLMOutput(outputs=outputs, offset_mappings=full_offsets)
    


    def forward(
        self,
        prompt: str,
        cache: Optional[Tuple] = None,
        return_llm_output: bool = False,
        output_hidden_states: bool = False,
    ) -> ModelOutput | LLMOutput:
        """
        Single forward pass.
        - return_llm_output=False: returns the raw ModelOutput (used for confidence scoring).
        - return_llm_output=True: behaves like a generate with 0 new tokens, attaches
          `sequences` and offset mappings, and returns LLMOutput (used by stepbootstrap sampling).
        """
        logger.info(f"Forward pass with model {self.model_name}")

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        temp = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)
        # breakpoint()
        
        
        full_input_ids = inputs.input_ids

        # When a cache is provided, only feed the delta tokens. model.__call__()
        # (unlike model.generate()) does not slice input_ids against past_key_values,
        # so passing the full prompt would double-count the cached prefix.
        if cache is not None:
            cache_len = cache.get_seq_length()
            model_input_ids = full_input_ids[:, cache_len:]
        else:
            model_input_ids = full_input_ids

        with torch.inference_mode():
            outputs = self.model(
                input_ids=model_input_ids,
                past_key_values=cache,
                use_cache=return_llm_output or cache is not None,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        if not return_llm_output:
            return outputs

        # Attach the full input_ids as 'sequences' to mimic model.generate() API.
        # After this forward pass, the cache covers exactly full_input_ids.
        outputs.sequences = full_input_ids

        output_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        full_offsets = self.tokenizer(
            output_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )["offset_mapping"]

        return LLMOutput(outputs=outputs, offset_mappings=full_offsets)



    def align_cache(self, cache: Optional[CacheBundle], prompt_text: str) -> KVCache | None:
        """
        we return the subset of cache that is aligned with the prompt text
        """
        if self.debug_nocache:
            return None

        if cache is None:
            return None

        new_input_ids = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        lcp = cache.longest_common_prefix(new_input_ids)

        # here we need to take account of the cache type.
        if isinstance(cache.cache, DynamicCache):
            # cache used by llama: regular kv cache
            # crop the cache to the lcp and return
            aligned_cache = copy.deepcopy(cache.cache)
            aligned_cache.crop(lcp)
            return aligned_cache
        elif isinstance(cache.cache, Qwen3_5DynamicCache):
            # cache used by qwen: dynamic cache
            # check if the whole cache is aligned with the prompt text
            if lcp == cache.cache.get_seq_length():
                return cache.cache
            else:
                # the cache is not aligned with the prompt text, we can't use any since qwen does not support cropping
                logger.warning(f"Qwen Cache is NOT aligned with the prompt text, cache reuse will not be performed")
                return None
        else:
            raise ValueError(f"Unsupported cache type: {type(cache.cache)}")



    def generate_multi_sequence(
        self,
        prompt: str,
        max_tokens: int,
        cache: Optional[Tuple],
        temperature: float,
        num_sequences: int,
        stop_strings: list[str] | None = None,
    ) -> SimpleNamespace:
        """Generate num_sequences outputs from the same prompt in one call.
        Manually batches inputs (and cache if provided) to avoid HF's internal
        num_return_sequences expansion which conflicts with pre-expanded caches.
        Returns raw batched output (sequences [N, seq_len], logits tuple of [N, vocab], etc.)."""
        logger.info(f"Batched generate ({num_sequences} seqs) with model {self.model_name}")

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        if stop_strings:
            stop_criteria = StoppingCriteriaList([
                StopStringCriteria(tokenizer=self.tokenizer, stop_strings=stop_strings),
            ])
        else:
            stop_criteria = None

        # Expand inputs to batch=N manually (no num_return_sequences)
        batched_input_ids = inputs.input_ids.expand(num_sequences, -1)
        batched_attention_mask = inputs.attention_mask.expand(num_sequences, -1)

        with torch.inference_mode():
            kwargs = dict(
                input_ids=batched_input_ids,
                attention_mask=batched_attention_mask,
                return_dict_in_generate=True,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                output_logits=True,
                output_scores=True,
                stopping_criteria=stop_criteria,
            )
            if cache is not None:
                # Replicate cache along batch dim to match N
                batched_cache = DynamicCache()
                for layer_idx, layer in enumerate(cache.layers):
                    k_exp = layer.keys.expand(num_sequences, -1, -1, -1).contiguous()
                    v_exp = layer.values.expand(num_sequences, -1, -1, -1).contiguous()
                    batched_cache.update(k_exp, v_exp, layer_idx)
                kwargs["past_key_values"] = batched_cache
                kwargs["use_cache"] = True
            outputs = self.model.generate(**kwargs)

        return outputs


    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int,
        temperature: float,
        stop_strings: list[str] | None = None,
    ) -> SimpleNamespace:
        """Generate from N different prompts as a left-padded batch. No cache.
        Returns raw batched output (sequences [N, max_seq_len], logits tuple of [N, vocab])."""
        logger.info(f"Batched generate ({len(prompts)} prompts) with model {self.model_name}")

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)

        if stop_strings:
            stop_criteria = StoppingCriteriaList([
                StopStringCriteria(tokenizer=self.tokenizer, stop_strings=stop_strings),
            ])
        else:
            stop_criteria = None

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                return_dict_in_generate=True,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                output_logits=True,
                output_scores=True,
                stopping_criteria=stop_criteria,
            )

        self.tokenizer.padding_side = "right"
        return outputs


    def forward_batch(
        self,
        prompts: list[str],
        return_llm_output: bool = False,
    ) -> list:
        """Batched forward pass over N prompts. Left-pads, no cache.
        Returns list of N LLMOutput (if return_llm_output) or list of per-sequence ModelOutput-like namespaces."""
        logger.info(f"Batched forward ({len(prompts)} prompts) with model {self.model_name}")

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)
        # inputs.input_ids: [N, max_len], inputs.attention_mask: [N, max_len]

        with torch.inference_mode():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                use_cache=return_llm_output,
                return_dict=True,
            )
        # outputs.logits: [N, max_len, vocab]

        self.tokenizer.padding_side = "right"

        N = len(prompts)
        pad_token_id = self.tokenizer.pad_token_id
        results = []

        for i in range(N):
            # Find where real tokens start (skip left-padding)
            mask_i = inputs.attention_mask[i]
            pad_len = (mask_i == 0).sum().item()
            actual_len = mask_i.sum().item()

            # Extract per-sequence logits (only real positions)
            seq_logits = outputs.logits[i:i+1, pad_len:, :]  # [1, actual_len, vocab]

            # Extract per-sequence input_ids (unpadded)
            seq_ids = inputs.input_ids[i, pad_len:].unsqueeze(0)  # [1, actual_len]

            if return_llm_output:
                # Extract per-sequence cache
                if outputs.past_key_values is not None:
                    per_seq_cache = DynamicCache()
                    for layer_idx, layer in enumerate(outputs.past_key_values.layers):
                        key = layer.keys[i:i+1, :, pad_len:, :].contiguous()
                        value = layer.values[i:i+1, :, pad_len:, :].contiguous()
                        per_seq_cache.update(key, value, layer_idx)
                else:
                    per_seq_cache = None

                # Build output namespace mimicking single-sequence forward
                seq_output = SimpleNamespace(
                    logits=seq_logits,
                    past_key_values=per_seq_cache,
                    sequences=seq_ids,
                )

                output_text = self.tokenizer.decode(seq_ids[0], skip_special_tokens=False)
                full_offsets = self.tokenizer(
                    output_text,
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                )["offset_mapping"]

                results.append(LLMOutput(outputs=seq_output, offset_mappings=full_offsets))
            else:
                results.append(SimpleNamespace(logits=seq_logits))

        return results


    def forward_batch_last_logits(self, prompts: list[str]) -> torch.Tensor:
        """Batched forward pass returning only last-token logits for each prompt.
        Returns: [N, vocab_size] tensor."""
        logger.info(f"Batched forward last-logits ({len(prompts)} prompts) with model {self.model_name}")

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                use_cache=False,
                return_dict=True,
            )

        self.tokenizer.padding_side = "right"
        # With left-padding, the last position is always the last real token for all sequences
        return outputs.logits[:, -1, :]  # [N, vocab]


    def forward_batch_with_cache(
        self,
        delta_texts: list[str],
        cache: 'DynamicCache',
        cache_seq_len: int,
        return_llm_output: bool = False,
    ) -> list:
        """Batched forward pass over N delta suffixes sharing a common KV cache prefix.
        The cache is replicated along the batch dim. Only delta tokens are fed.

        Args:
            delta_texts: N strings representing the suffix beyond the cached prefix.
            cache: DynamicCache covering the shared prefix (batch=1).
            cache_seq_len: number of tokens in the cached prefix.
            return_llm_output: if True, returns list[LLMOutput]; otherwise list[SimpleNamespace(logits=...)].
        """
        logger.info(f"Batched forward with cache ({len(delta_texts)} deltas, cache_seq_len={cache_seq_len}) with model {self.model_name}")

        N = len(delta_texts)

        # Tokenize deltas with left-padding so they align on the right
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        delta_inputs = self.tokenizer(
            delta_texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)
        # delta_inputs.input_ids: [N, max_delta_len]
        # delta_inputs.attention_mask: [N, max_delta_len] (0 for padding, 1 for real)

        max_delta_len = delta_inputs.input_ids.shape[1]

        # Build full attention mask: [N, cache_seq_len + max_delta_len]
        # The cache prefix is always attended to (all 1s), delta has left-padding (0s then 1s)
        cache_mask = torch.ones(N, cache_seq_len, dtype=torch.long, device=self.model.device)
        full_attention_mask = torch.cat([cache_mask, delta_inputs.attention_mask], dim=1)

        # Replicate cache along batch dim: each layer's key/value [1, heads, seq, dim] -> [N, heads, seq, dim]
        batched_cache = DynamicCache()
        for layer_idx, layer in enumerate(cache.layers):
            key = layer.keys.expand(N, -1, -1, -1)   # [N, heads, cache_seq_len, head_dim]
            value = layer.values.expand(N, -1, -1, -1)
            batched_cache.update(key, value, layer_idx)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=delta_inputs.input_ids,
                attention_mask=full_attention_mask,
                past_key_values=batched_cache,
                use_cache=return_llm_output,
                return_dict=True,
            )
        # outputs.logits: [N, max_delta_len, vocab]

        self.tokenizer.padding_side = "right"

        results = []
        for i in range(N):
            mask_i = delta_inputs.attention_mask[i]
            pad_len = (mask_i == 0).sum().item()

            if return_llm_output:
                # Logits for real delta tokens only
                seq_logits = outputs.logits[i:i+1, pad_len:, :]  # [1, actual_delta_len, vocab]

                # Full sequence = prefix ids + delta ids (unpadded)
                # We need the prefix ids from the original tokenization
                delta_ids = delta_inputs.input_ids[i, pad_len:]  # [actual_delta_len]

                # Extract per-sequence cache if available
                if outputs.past_key_values is not None:
                    per_seq_cache = DynamicCache()
                    for layer_idx, layer in enumerate(outputs.past_key_values.layers):
                        # Skip the padding positions in the new part
                        key = layer.keys[i:i+1, :, :cache_seq_len + (max_delta_len - pad_len), :].contiguous()
                        value = layer.values[i:i+1, :, :cache_seq_len + (max_delta_len - pad_len), :].contiguous()
                        per_seq_cache.update(key, value, layer_idx)
                else:
                    per_seq_cache = None

                seq_output = SimpleNamespace(
                    logits=seq_logits,
                    past_key_values=per_seq_cache,
                    sequences=delta_ids.unsqueeze(0),  # placeholder — caller must prepend prefix ids if needed
                )
                results.append(seq_output)
            else:
                seq_logits = outputs.logits[i:i+1, pad_len:, :]
                results.append(SimpleNamespace(logits=seq_logits))

        return results


    def forward_batch_last_logits_with_cache(
        self,
        delta_texts: list[str],
        cache: 'DynamicCache',
        cache_seq_len: int,
    ) -> torch.Tensor:
        """Batched forward pass with shared cache, returning only last-token logits.
        Returns: [N, vocab_size] tensor.

        This is the confidence-scoring fast path: reuse the shared prefix cache
        and only compute over the (short) unique suffix per sample."""
        logger.info(f"Batched forward last-logits with cache ({len(delta_texts)} deltas, cache_seq_len={cache_seq_len})")

        N = len(delta_texts)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        delta_inputs = self.tokenizer(
            delta_texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)

        max_delta_len = delta_inputs.input_ids.shape[1]

        # Full attention mask: prefix (all 1s) + delta (with left-pad 0s)
        cache_mask = torch.ones(N, cache_seq_len, dtype=torch.long, device=self.model.device)
        full_attention_mask = torch.cat([cache_mask, delta_inputs.attention_mask], dim=1)

        # Replicate cache along batch dim (no copy — expand shares memory)
        batched_cache = DynamicCache()
        for layer_idx, layer in enumerate(cache.layers):
            key = layer.keys.expand(N, -1, -1, -1)
            value = layer.values.expand(N, -1, -1, -1)
            batched_cache.update(key, value, layer_idx)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=delta_inputs.input_ids,
                attention_mask=full_attention_mask,
                past_key_values=batched_cache,
                use_cache=False,
                return_dict=True,
            )

        self.tokenizer.padding_side = "right"
        # Last position = last real token for each row
        return outputs.logits[:, -1, :]  # [N, vocab]

