import unittest

from analysis.api import (
    estimate_model_runtime,
    normalize_estimate_result,
    normalized_schemas,
    run_normalized_pipeline,
)
from analysis.hardware_estimates import compute_flop_rate_gpu
from models.hardware_specs import gpu_specs
from models.mlp import SimpleMLP


class TestPublicApi(unittest.TestCase):
    def test_estimate_model_runtime_cpu(self):
        # sanity check for cpu path and normalized payload
        model = SimpleMLP()
        result = estimate_model_runtime(
            model,
            batch_size=1,
            hardware="cpu",
            include_bias=True,
            include_activations=False,
        )

        self.assertEqual(result.macs, 109184)
        self.assertEqual(result.breakdown["bias_ops"], 202)
        self.assertEqual(result.flops, 218570)
        self.assertGreater(result.est_runtime, 0.0)
        normalized = normalize_estimate_result(result)
        self.assertEqual(normalized["kind"], "estimate")
        self.assertIn("version", normalized)

    def test_estimate_model_runtime_gpu(self):
        # ensure gpu path uses the correct flop-rate calculation
        model = SimpleMLP()
        result = estimate_model_runtime(
            model,
            batch_size=2,
            hardware="gpu",
            precision="fp32",
        )
        expected_rate = compute_flop_rate_gpu(gpu_specs, precision="fp32")
        self.assertAlmostEqual(result.flop_rate, expected_rate)

    def test_reject_invalid_hardware(self):
        # invalid hardware keys should be rejected
        model = SimpleMLP()
        with self.assertRaises(ValueError):
            estimate_model_runtime(model, batch_size=1, hardware="tpu")

    def test_reject_precision_on_cpu(self):
        # precision is only configurable for gpu estimates
        model = SimpleMLP()
        with self.assertRaises(ValueError):
            estimate_model_runtime(model, batch_size=1, hardware="cpu", precision="fp16")

    def test_normalized_schemas(self):
        # schema helper should expose each expected section
        schemas = normalized_schemas()
        self.assertIn("estimate", schemas)
        self.assertIn("measurement", schemas)
        self.assertIn("comparison", schemas)
        self.assertIn("version", schemas)

    def test_run_normalized_pipeline(self):
        # end-to-end normalized payload should include all sections
        model = SimpleMLP()
        payload = run_normalized_pipeline(
            model,
            input_shape=(784,),
            batch_size=1,
            hardware="cpu",
            device="cpu",
            runs=2,
            warmup=1,
        )
        self.assertIn("estimate", payload)
        self.assertIn("measurement", payload)
        self.assertIn("comparison", payload)


if __name__ == "__main__":
    unittest.main()
