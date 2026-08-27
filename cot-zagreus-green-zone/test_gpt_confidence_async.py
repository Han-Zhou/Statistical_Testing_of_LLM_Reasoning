import asyncio
import time
import unittest
from types import SimpleNamespace

from confidence.confidence_engine import ConfidenceEngine


class _ConcurrencyTracker:
    def __init__(self):
        self.active = 0
        self.maximum = 0

    async def run(self):
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1


class _AsyncConfidenceMethodStub:
    def __init__(self, name, tracker):
        self.name = name
        self.tracker = tracker

    async def compute_confidence_async(self, parsed_output):
        await self.tracker.run()
        return {self.name: parsed_output.index}, {}


class GptConfidenceAsyncTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _output(index):
        return SimpleNamespace(
            index=index,
            answer_token_probs=[],
            answer_token_ids=[],
        )

    async def test_batch_limits_requests_and_preserves_order(self):
        tracker = _ConcurrencyTracker()
        engine = ConfidenceEngine.__new__(ConfidenceEngine)
        engine.confidence_config = SimpleNamespace(
            cuda_sync_for_timing=False,
            debug_top20=False,
        )
        engine.indirect_confidence_method = _AsyncConfidenceMethodStub(
            "indirect", tracker
        )
        engine.verbal_confidence_method = _AsyncConfidenceMethodStub(
            "verbal", tracker
        )
        engine.time_stamp = time.perf_counter

        results = await engine.compute_confidence_batch_async(
            [self._output(i) for i in range(6)],
            concurrency=2,
        )

        self.assertEqual(tracker.maximum, 2)
        self.assertEqual(
            [scores.indirect_probabilities["indirect"] for scores, _ in results],
            list(range(6)),
        )
        self.assertEqual(
            [scores.verbconf_probabilities["verbal"] for scores, _ in results],
            list(range(6)),
        )
        self.assertTrue(all(timing.indirect_time > 0 for _, timing in results))
        self.assertTrue(all(timing.verbconf_time > 0 for _, timing in results))

    async def test_rejects_invalid_concurrency(self):
        engine = ConfidenceEngine.__new__(ConfidenceEngine)
        with self.assertRaisesRegex(ValueError, "concurrency must be at least 1"):
            await engine.compute_confidence_batch_async([], concurrency=0)


if __name__ == "__main__":
    unittest.main()
