"""CPU-only tests for owned Mamba-2 generator sampling admission.

No pretrained checkpoint, network access, external service or GPU required.
"""
import inspect
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import mamba2


class FakeModel:
    def __init__(self, values, vocab_size=None):
        self.args = SimpleNamespace(chunk_size=64, n_layer=0)
        if vocab_size is not None:
            self.args.vocab_size = vocab_size
        self.device = "cpu"
        self.logits = torch.tensor(values, dtype=torch.float64)
        self.calls = 0

    def __call__(self, token_ids, cache):
        self.calls += 1
        return self.logits.reshape(1, 1, -1).clone(), cache


class SamplingAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.values = [math.log(p) for p in (0.55, 0.25, 0.15, 0.05)]

    def observe(self, values=None, prompt=None, vocab_size=None, **overrides):
        model = FakeModel(self.values if values is None else values, vocab_size)
        probs = []
        def capture(distribution, num_samples):
            probs.append(distribution.detach().clone())
            return torch.tensor([int(distribution.argmax())])
        options = dict(max_new_length=1, temperature=1.0, top_k=0,
                       top_p=1.0, eos_token_id=-1)
        options.update(overrides)
        if prompt is None:
            prompt = torch.tensor([1], dtype=torch.long)
        with patch.object(mamba2.torch, "multinomial", side_effect=capture):
            result = list(mamba2.Mamba2LMHeadModel.generate(model, prompt, **options))
        return model, probs, result

    def test_source_uses_requested_top_p(self):
        code = inspect.getsource(mamba2.Mamba2LMHeadModel.generate)
        self.assertIn("sorted_indices_to_remove = cum_probs > top_p", code)
        self.assertNotIn("sorted_indices_to_remove = cum_probs > 0.5", code)

    def test_distinct_top_p_thresholds(self):
        _, a, _ = self.observe(top_p=0.2)
        _, b, _ = self.observe(top_p=0.9)
        self.assertEqual(int(torch.count_nonzero(a[0])), 1)
        self.assertEqual(int(torch.count_nonzero(b[0])), 3)

    def test_nucleus_support_is_monotonic(self):
        sizes = []
        for p in (0.01, 0.2, 0.5, 0.7, 0.8, 0.9, 0.99, 1.0):
            _, probs, _ = self.observe(top_p=p)
            sizes.append(int(torch.count_nonzero(probs[0])))
        self.assertEqual(sizes, sorted(sizes))

    def test_top_k_combines_with_top_p(self):
        _, probs, _ = self.observe(top_k=2, top_p=0.9)
        self.assertEqual(int(torch.count_nonzero(probs[0])), 2)
        self.assertAlmostEqual(float(probs[0].sum()), 1.0)

    def test_top_k_exact_cardinality_with_tied_logits(self):
        # Regression: comparing against the kth *value* keeps all ties.
        for k in (1, 2, 3, 4):
            with self.subTest(top_k=k):
                _, distributions, _ = self.observe(
                    values=[0.0, 0.0, 0.0, 0.0], top_k=k, top_p=1.0
                )
                self.assertEqual(int(torch.count_nonzero(distributions[0])), k)
                self.assertAlmostEqual(float(distributions[0].sum()), 1.0)

    def test_invalid_top_p_rejected_before_model(self):
        for p in (0, -0.1, 1.001, float("nan"), float("inf"), True, None, "0.8"):
            model = FakeModel(self.values)
            with self.subTest(p=p), self.assertRaises(ValueError):
                list(mamba2.Mamba2LMHeadModel.generate(model, torch.tensor([1]), top_p=p))
            self.assertEqual(model.calls, 0)

    def test_invalid_top_k_rejected_before_model(self):
        for k in (-1, 1.1, True, None, "4", 10**100):
            model = FakeModel(self.values)
            with self.subTest(k=k), self.assertRaises(ValueError):
                list(mamba2.Mamba2LMHeadModel.generate(model, torch.tensor([1]), top_k=k))
            self.assertEqual(model.calls, 0)

    def test_top_k_above_declared_vocab_rejected_before_model(self):
        model = FakeModel(self.values, vocab_size=4)
        with self.assertRaisesRegex(ValueError, "top_k exceeds declared"):
            list(mamba2.Mamba2LMHeadModel.generate(model, torch.tensor([1]), top_k=5))
        self.assertEqual(model.calls, 0)

    def test_top_k_above_observed_vocab_rejected(self):
        with self.assertRaisesRegex(ValueError, "top_k exceeds model"):
            self.observe(top_k=5)

    def test_invalid_temperature_rejected_before_model(self):
        for t in (0, -1, 0.0, float("nan"), float("inf"), True, None, "0.4", 10**2000):
            model = FakeModel(self.values)
            with self.subTest(t=str(t)[:30]), self.assertRaises(ValueError):
                list(mamba2.Mamba2LMHeadModel.generate(model, torch.tensor([1]), temperature=t))
            self.assertEqual(model.calls, 0)

    def test_temperature_changes_distribution(self):
        _, cold, _ = self.observe(temperature=0.5)
        _, normal, _ = self.observe(temperature=1.0)
        self.assertGreater(float(cold[0][0]), float(normal[0][0]))

    def test_invalid_max_length_rejected_before_model(self):
        for n in (-1, 4097, True, None, 0.4, "5"):
            model = FakeModel(self.values)
            with self.subTest(n=n), self.assertRaises(ValueError):
                list(mamba2.Mamba2LMHeadModel.generate(model, torch.tensor([1]), max_new_length=n))
            self.assertEqual(model.calls, 0)

    def test_zero_new_length_no_model_call(self):
        model, probs, outputs = self.observe(max_new_length=0)
        self.assertEqual((model.calls, len(probs), len(outputs)), (0, 0, 0))

    def test_invalid_prompt_rejected_before_model(self):
        for ids in (None, [], torch.tensor([]), torch.tensor([1.0]),
                    torch.tensor([[1]]), torch.tensor([1], dtype=torch.int32)):
            model = FakeModel(self.values)
            with self.subTest(ids=str(ids)[:30]), self.assertRaises(ValueError):
                list(mamba2.Mamba2LMHeadModel.generate(model, ids))
            self.assertEqual(model.calls, 0)

    def test_negative_and_oversized_prompt_rejected_before_model(self):
        for ids in (torch.tensor([-1]), torch.zeros(8193, dtype=torch.long)):
            model = FakeModel(self.values)
            with self.assertRaises(ValueError):
                list(mamba2.Mamba2LMHeadModel.generate(model, ids))
            self.assertEqual(model.calls, 0)

    def test_prompt_token_exceeds_declared_vocab(self):
        model = FakeModel(self.values, vocab_size=4)
        with self.assertRaisesRegex(ValueError, "exceeds declared vocabulary"):
            list(mamba2.Mamba2LMHeadModel.generate(model, torch.tensor([4]), top_k=0))
        self.assertEqual(model.calls, 0)

    def test_nonfinite_model_logits_rejected(self):
        for x in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(x=x), self.assertRaisesRegex(ValueError, "model logits"):
                self.observe(values=[x, 0.0])

    def test_scaled_logits_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError, "temperature scaling"):
            self.observe(values=[1e308, 1e308], temperature=1e-300)

    def test_empty_logits_rejected(self):
        with self.assertRaisesRegex(ValueError, "model logits"):
            self.observe(values=[])

    def test_normalized_parameter_grid_100_combinations(self):
        count = 0
        for k in (0, 1, 2, 3, 4):
            for p in (0.01, 0.3, 0.55, 0.85, 1.0):
                for t in (0.1, 0.5, 1.0, 2.0):
                    _, probs, outputs = self.observe(top_k=k, top_p=p, temperature=t)
                    self.assertEqual(len(outputs), 1)
                    self.assertTrue(torch.isfinite(probs[0]).all().item())
                    self.assertAlmostEqual(float(probs[0].sum()), 1.0, places=8)
                    self.assertGreater(int(torch.count_nonzero(probs[0])), 0)
                    count += 1
        self.assertEqual(count, 100)

    def test_full_synthetic_two_layer_generation(self):
        torch.manual_seed(91202)
        cfg = mamba2.Mamba2Config(d_model=64, n_layer=2, d_state=32,
                                   headdim=32, chunk_size=64, vocab_size=128)
        model = mamba2.Mamba2LMHeadModel(cfg, device="cpu").eval()
        with torch.no_grad():
            for layer in model.backbone.layers:
                layer.mixer.dt_bias.zero_()
                layer.mixer.A_log.zero_()
                layer.mixer.D.fill_(1.0)
        ids = torch.arange(66, dtype=torch.long) % cfg.vocab_size
        with torch.inference_mode():
            output = list(model.generate(ids, max_new_length=3, top_k=20,
                                         top_p=0.8, eos_token_id=-1))
        self.assertEqual(len(output), 3)
        self.assertTrue(all(0 <= token < cfg.vocab_size and len(cache) == 2
                            for token, cache in output))


if __name__ == "__main__":
    unittest.main()
