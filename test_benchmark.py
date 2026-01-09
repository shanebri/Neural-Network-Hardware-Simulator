import unittest

from analysis.benchmark import measure_forward_time
from models.mlp import SimpleMLP


class TestBenchmark(unittest.TestCase):
    def test_measure_forward_time_cpu(self):
        # basic cpu timing sanity check
        model = SimpleMLP()
        result = measure_forward_time(
            model,
            input_shape=(784,),
            batch_size=1,
            device="cpu",
            runs=2,
            warmup=1,
        )
        self.assertGreater(result.mean_s, 0.0)
        self.assertGreater(result.samples_per_s, 0.0)
        self.assertEqual(result.batch_size, 1)


if __name__ == "__main__":
    unittest.main()
