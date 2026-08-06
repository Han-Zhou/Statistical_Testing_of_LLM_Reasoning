import unittest

import numpy as np

from pipeline.sampling.stepbootstrap_sampling import _resample_steps


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


if __name__ == "__main__":
    unittest.main()
