import unittest
from types import SimpleNamespace

import numpy as np

from pipeline.sampling.stepbootstrap_sampling import StepBootstrapSampling, _resample_steps


class _StubRng:
    def choice(self, population_size, size, replace):
        self.call = (population_size, size, replace)
        return np.array([3, 1, 1, 0])


class StepBootstrapResamplingTest(unittest.TestCase):
    def test_resampling_preserves_original_order_and_final_step(self):
        steps = ["step 0", "step 1", "step 2", "step 3", "final step"]
        rng = _StubRng()

        result = _resample_steps(steps, rng)

        self.assertEqual(rng.call, (4, 4, True))
        self.assertEqual(
            result,
            ["step 0", "step 1", "step 1", "step 3", "final step"],
        )

    def test_single_step_is_returned_unchanged(self):
        result = _resample_steps(["final step"], np.random.default_rng(0))

        self.assertEqual(result, ["final step"])


class StepBootstrapPromptTest(unittest.TestCase):
    def test_local_backends_teacher_force_reference_answer(self):
        for backend in ("hf", "vllm"):
            with self.subTest(backend=backend):
                sampling = StepBootstrapSampling.__new__(StepBootstrapSampling)
                sampling.generation_config = SimpleNamespace(
                    backend=backend,
                    prompt_type=1,
                )
                sampling.context = SimpleNamespace(reference_vanilla_final_answer="C")
                messages = [{"role": "assistant", "content": "prefill"}]

                result = sampling._add_assistant_message_to_messages(
                    messages,
                    "Step 1: resampled reasoning.",
                )

                self.assertIn(
                    "Therefore the final answer is \\boxed{C}.",
                    result[-1]["content"],
                )
                self.assertEqual(messages[-1]["content"], "prefill")

    def test_api_backend_leaves_answer_open_for_generation(self):
        sampling = StepBootstrapSampling.__new__(StepBootstrapSampling)
        sampling.generation_config = SimpleNamespace(backend="api", prompt_type=1)
        sampling.context = SimpleNamespace(reference_vanilla_final_answer="C")

        result = sampling._add_assistant_message_to_messages(
            [{"role": "assistant", "content": "prefill"}],
            "Step 1: resampled reasoning.",
        )

        self.assertTrue(result[-1]["content"].endswith("\\boxed{"))
        self.assertNotIn("\\boxed{C}", result[-1]["content"])


if __name__ == "__main__":
    unittest.main()
