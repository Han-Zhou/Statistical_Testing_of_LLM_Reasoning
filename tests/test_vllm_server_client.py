import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from models.core_models import vllm_llm
from models.core_models.vllm_llm import VLLM_LLM, VLLMServerClient


def _choice(prompt_ids, token_id, text):
    return SimpleNamespace(
        text=text,
        token_ids=[token_id],
        prompt_token_ids=prompt_ids,
        logprobs=SimpleNamespace(
            top_logprobs=[{f"token_id:{token_id}": -0.1}],
            token_logprobs=[-0.1],
        ),
        prompt_logprobs=[
            None,
            {str(prompt_ids[-1]): SimpleNamespace(logprob=-0.2)},
        ],
    )


class VllmServerClientTest(unittest.TestCase):
    def test_remote_mode_skips_local_engine_construction(self):
        with (
            patch.object(vllm_llm.AutoTokenizer, "from_pretrained", return_value=Mock()),
            patch.object(vllm_llm, "LLM") as local_engine,
        ):
            model = VLLM_LLM("qwen", base_url="http://127.0.0.1:8000")

        local_engine.assert_not_called()
        self.assertIsInstance(model.model, VLLMServerClient)

    def test_normalizes_base_url_and_groups_batched_n_outputs(self):
        choices = [
            _choice([1, 2], 10, "a0"),
            _choice([1, 2], 11, "a1"),
            _choice([3, 4], 12, "b0"),
            _choice([3, 4], 13, "b1"),
        ]
        completions = Mock()
        completions.create.return_value = SimpleNamespace(choices=choices)
        fake_client = SimpleNamespace(completions=completions)
        client = VLLMServerClient(
            "http://127.0.0.1:8000/",
            "Qwen/Qwen3.5-27B",
            client=fake_client,
        )

        outputs = client.generate(
            ["prompt a", "prompt b"],
            SimpleNamespace(
                max_tokens=8,
                temperature=0.7,
                stop=["STOP"],
                n=2,
                logprobs=20,
                prompt_logprobs=None,
                include_stop_str_in_output=False,
                skip_special_tokens=False,
            ),
        )

        self.assertEqual(client.base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual([output.prompt for output in outputs], ["prompt a", "prompt b"])
        self.assertEqual([output.prompt_token_ids for output in outputs], [[1, 2], [3, 4]])
        self.assertEqual([item.text for item in outputs[0].outputs], ["a0", "a1"])
        self.assertEqual(outputs[1].outputs[0].logprobs[0][12].logprob, -0.1)

        request = completions.create.call_args.kwargs
        self.assertEqual(request["model"], "Qwen/Qwen3.5-27B")
        self.assertEqual(request["prompt"], ["prompt a", "prompt b"])
        self.assertTrue(request["extra_body"]["return_token_ids"])
        self.assertTrue(request["extra_body"]["return_tokens_as_token_ids"])

    def test_converts_prompt_logprobs_for_forward_requests(self):
        completions = Mock()
        completions.create.return_value = SimpleNamespace(
            choices=[_choice([1, 2], 10, "a")]
        )
        client = VLLMServerClient(
            "http://127.0.0.1:8000/v1",
            "Qwen/Qwen3.5-27B",
            client=SimpleNamespace(completions=completions),
        )

        outputs = client.generate(
            ["prompt"],
            SimpleNamespace(
                max_tokens=1,
                temperature=0.0,
                stop=None,
                n=1,
                logprobs=None,
                prompt_logprobs=20,
                include_stop_str_in_output=False,
                skip_special_tokens=False,
            ),
        )

        self.assertEqual(outputs[0].prompt_logprobs[1][2].logprob, -0.2)
        request = completions.create.call_args.kwargs
        self.assertEqual(request["extra_body"]["prompt_logprobs"], 20)


if __name__ == "__main__":
    unittest.main()
