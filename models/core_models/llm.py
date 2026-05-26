import logging
import copy
import torch
from typing import Optional, Tuple
from dotenv import load_dotenv


from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, StopStringCriteria
from transformers.utils import ModelOutput

from domain.data import LLMOutput, KVCache, CacheBundle
from models.core_models.registry import MODEL_HF_REGISTRY

load_dotenv()
logger = logging.getLogger(__name__)

class LLM():

    def __init__(self, model_name: str, attention_implementation: str):
        self.model_name = model_name
        self.attention_implementation = attention_implementation
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
            # stop_strings: list[str], 
            temperature: float
        ) -> LLMOutput:
        # Generate text based on the prompt
        # prompt SHOULD ALREADY HAVE chat template applied
        logger.info(f"Generating text with model {self.model_name}")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # if stop_strings:
        #     stop_criteria = StoppingCriteriaList([
        #         StopStringCriteria(tokenizer=self.tokenizer, stop_strings=stop_strings),
        #     ])
        # else:
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
    


    def forward(self, prompt: str, cache: Optional[Tuple] = None, output_hidden_states: bool = False) -> ModelOutput:
        """
        Used for confidence scoring
        """
        logger.info(f"Forward pass with model {self.model_name}")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
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
                use_cache=cache is not None,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        return outputs


    def forward_pass(
        self,
        prompt: str,
        cache: Optional[Tuple] = None,
    ) -> LLMOutput:
        """
        different than forward function:
        - this is basically a generate with 0 tokens generated
        - for usage by stepbootstrap sampling
        """
        logger.info(f"Forward pass with model {self.model_name}")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
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
                use_cache=True,
                return_dict=True,
            )

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
        if cache is None:
            return None

        new_input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids[0]
        lcp = cache.longest_common_prefix(new_input_ids)

        aligned_cache = copy.deepcopy(cache.cache)
        aligned_cache.crop(lcp)

        return aligned_cache

