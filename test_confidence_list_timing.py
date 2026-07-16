import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from openai.types.chat.chat_completion_token_logprob import TopLogprob

from confidence.confidence_engine import ConfidenceEngine


class _ConfidenceMethodStub:
    def compute_confidence(self, parsed_output):
        return {}, {}


class _TokenizerStub:
    TOKENS = {1: "A", 2: "B", 3: "C"}

    def decode(self, token_ids):
        return self.TOKENS[token_ids[0]]


class ConfidenceListTimingTest(unittest.TestCase):
    def _engine(self, timestamps, *, debug_top20=True):
        engine = ConfidenceEngine.__new__(ConfidenceEngine)
        engine.confidence_config = SimpleNamespace(
            cuda_sync_for_timing=False,
            debug_top20=debug_top20,
        )
        engine.model_scorer = SimpleNamespace(
            model=SimpleNamespace(tokenizer=_TokenizerStub())
        )
        engine.indirect_confidence_method = _ConfidenceMethodStub()
        engine.verbal_confidence_method = _ConfidenceMethodStub()
        engine.time_stamp = Mock(side_effect=timestamps)
        return engine

    @staticmethod
    def _parsed_output(answer_token_probs, answer_token_ids):
        return SimpleNamespace(
            answer_token_probs=answer_token_probs,
            answer_token_ids=answer_token_ids,
        )

    def test_openai_probabilities_and_timings_accumulate(self):
        raw_probs = [
            [
                TopLogprob(token="A", logprob=math.log(0.8)),
                TopLogprob(token="B", logprob=math.log(0.2)),
            ],
            [
                TopLogprob(token="B", logprob=math.log(0.6)),
                TopLogprob(token="C", logprob=math.log(0.4)),
            ],
        ]
        engine = self._engine(
            [1.0, 1.1, 1.3, 2.0, 2.2, 2.5, 10.0, 11.0, 13.0]
        )

        scores, timings = engine._compute_confidence_list_probs(
            self._parsed_output(raw_probs, ["A", "B"])
        )

        self.assertAlmostEqual(scores.answer_probabilities[0]["A"], 0.8)
        self.assertAlmostEqual(scores.answer_probabilities[1]["B"], 0.6)
        expected_entropies = [
            -(0.8 * math.log(0.8) + 0.2 * math.log(0.2)),
            -(0.6 * math.log(0.6) + 0.4 * math.log(0.4)),
        ]
        self.assertAlmostEqual(scores.answer_entropy[0]["A"], expected_entropies[0])
        self.assertAlmostEqual(scores.answer_entropy[1]["B"], expected_entropies[1])
        self.assertAlmostEqual(timings.answer_prob_time, 0.3)
        self.assertAlmostEqual(timings.answer_ent_time, 0.5)
        self.assertEqual(timings.answer_score_prob_time, 0.0)
        self.assertEqual(timings.answer_score_ent_time, 0.0)

    def test_vllm_probabilities_use_the_same_timing_path(self):
        engine = self._engine([3.0, 3.25, 3.75, 10.0, 11.0, 12.0])

        scores, timings = engine._compute_confidence_list_probs(
            self._parsed_output([{1: 0.7, 2: 0.3}], ["A"])
        )

        self.assertEqual(scores.answer_probabilities, [{"A": 0.7}])
        self.assertAlmostEqual(
            scores.answer_entropy[0]["A"],
            -(0.7 * math.log(0.7) + 0.3 * math.log(0.3)),
        )
        self.assertEqual(
            scores.debug["answer_top20_probabilities"],
            [{"A": 0.7, "B": 0.3}],
        )
        self.assertAlmostEqual(timings.answer_prob_time, 0.25)
        self.assertAlmostEqual(timings.answer_ent_time, 0.5)

    def test_empty_answer_tokens_keep_zero_timings(self):
        engine = self._engine([5.0, 6.0, 7.0], debug_top20=False)

        scores, timings = engine._compute_confidence_list_probs(
            self._parsed_output([], [])
        )

        self.assertEqual(scores.answer_probabilities, [])
        self.assertEqual(scores.answer_entropy, [])
        self.assertEqual(timings.answer_prob_time, 0.0)
        self.assertEqual(timings.answer_ent_time, 0.0)
        self.assertEqual(timings.answer_score_prob_time, 0.0)
        self.assertEqual(timings.answer_score_ent_time, 0.0)


if __name__ == "__main__":
    unittest.main()
