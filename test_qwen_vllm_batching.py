import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from confidence.confidence_engine import ConfidenceEngine
from confidence.indirect import IndirectConfidenceMethod
from confidence.verbal import VerbalConfidenceMethod
from models.adapters.qwen_vllm_adapter import QwenVllmAdapter, QwenVllmScorer


class _TokenizerStub:
    true_id = 1
    false_id = 2
    answer_id = 3

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == " True":
            token_id = self.true_id
        elif text == " False":
            token_id = self.false_id
        else:
            token_id = 10 + int(text)
        return SimpleNamespace(input_ids=[token_id])

    def decode(self, token_ids):
        if token_ids[0] == self.answer_id:
            return "A"
        return str(token_ids[0])


class QwenVllmBatchingTest(unittest.TestCase):
    def test_generate_batch_uses_one_multi_sequence_helper_call(self):
        adapter = QwenVllmAdapter.__new__(QwenVllmAdapter)
        adapter.render_prompt = Mock(return_value="rendered prompt")
        adapter.generate_helper = Mock(
            return_value=(["clean 0", "clean 1"], ["raw 0", "raw 1"])
        )
        adapter._process_forward_output = Mock(
            side_effect=lambda text, output: (text, output)
        )

        outputs = adapter.generate_batch(
            messages=[{"role": "user", "content": "question"}],
            max_tokens=128,
            temperature=0.7,
            num_sequences=2,
        )

        self.assertEqual(outputs, [("clean 0", "raw 0"), ("clean 1", "raw 1")])
        adapter.generate_helper.assert_called_once_with(
            "rendered prompt", 128, None, 0.7, n=2
        )

    def test_forward_pass_batch_preserves_prompt_order(self):
        adapter = QwenVllmAdapter.__new__(QwenVllmAdapter)
        adapter.render_prompt = Mock(
            side_effect=lambda messages: messages[0]["content"]
        )
        adapter.model = SimpleNamespace(
            forward=Mock(return_value=["raw a", "raw b"])
        )
        adapter._process_forward_output = Mock(
            side_effect=lambda prompt, output: (prompt, output)
        )

        outputs = adapter.forward_pass_batch(
            [
                [{"role": "assistant", "content": "prompt a"}],
                [{"role": "assistant", "content": "prompt b"}],
            ]
        )

        self.assertEqual(outputs, [("prompt a", "raw a"), ("prompt b", "raw b")])
        adapter.model.forward.assert_called_once_with(
            prompts=["prompt a", "prompt b"],
            return_llm_output=False,
        )

    def test_scorer_submits_all_confidence_prompts_together(self):
        generated = [
            SimpleNamespace(
                outputs=[SimpleNamespace(logprobs=[{1: SimpleNamespace(logprob=-0.1)}])]
            ),
            SimpleNamespace(
                outputs=[SimpleNamespace(logprobs=[{2: SimpleNamespace(logprob=-0.2)}])]
            ),
        ]
        generate = Mock(return_value=generated)
        scorer = QwenVllmScorer(
            SimpleNamespace(model=SimpleNamespace(generate=generate))
        )

        scores = scorer.forward_batch_confidence(["prompt a", "prompt b"])

        self.assertEqual(len(scores), 2)
        self.assertEqual(scores[0][1].logprob, -0.1)
        self.assertEqual(generate.call_args.args[0], ["prompt a", "prompt b"])

    def test_confidence_batch_accepts_sparse_vllm_logprobs_and_list_probs(self):
        tokenizer = _TokenizerStub()
        verbal_scores = {
            10 + score: SimpleNamespace(logprob=-abs(75 - score) / 10)
            for score in range(101)
        }
        scorer = SimpleNamespace(
            model=SimpleNamespace(tokenizer=tokenizer),
            forward_batch_confidence=Mock(
                side_effect=[
                    [
                        {
                            tokenizer.true_id: SimpleNamespace(logprob=-0.1),
                            tokenizer.false_id: SimpleNamespace(logprob=-2.0),
                        }
                    ],
                    [verbal_scores],
                ]
            ),
        )
        engine = ConfidenceEngine.__new__(ConfidenceEngine)
        engine.confidence_config = SimpleNamespace(
            cuda_sync_for_timing=False,
            debug_top20=True,
        )
        engine.model_scorer = scorer
        engine.indirect_confidence_method = IndirectConfidenceMethod(scorer)
        engine.verbal_confidence_method = VerbalConfidenceMethod(scorer)

        parsed_output = SimpleNamespace(
            final_answer="A",
            text_question="question",
            text_cot=" reasoning",
            input_messages=None,
            answer_token_probs=[
                {tokenizer.answer_id: 0.75, 4: 0.25}
            ],
            answer_token_ids=["A"],
            answer_token_score_probs=None,
        )

        [(scores, _timings)] = engine.compute_confidence_batch([parsed_output])

        self.assertEqual(scores.answer_probabilities, [{"A": 0.75}])
        self.assertAlmostEqual(
            scores.answer_entropy[0]["A"],
            -(0.75 * math.log(0.75) + 0.25 * math.log(0.25)),
        )
        self.assertGreater(scores.indirect_probabilities["True"], 0.5)
        self.assertEqual(max(scores.verbconf_probabilities, key=scores.verbconf_probabilities.get), 75)
        self.assertEqual(scorer.forward_batch_confidence.call_count, 2)


if __name__ == "__main__":
    unittest.main()
