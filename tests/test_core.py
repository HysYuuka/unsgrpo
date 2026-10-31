import itertools
import json
import unittest

import numpy as np
import torch

from unsgrpo.core import (unit_rows, recurrence_original, recurrence_loo_video,
                          midrank_percentile, parse_scores, reward_parts,
                          group_advantages, grpo_loss)


class FeatureTests(unittest.TestCase):
    def test_mean_cosine_exact_nonrenormalized(self):
        query = np.array([[3., 4.], [1., 0.]])
        bank = np.array([[0., 2.], [6., 0.], [-3., 0.]])
        expected = (unit_rows(query) @ unit_rows(bank).T).mean(axis=1)
        np.testing.assert_allclose(recurrence_original(query, bank), expected)
        self.assertLess(abs(expected[0]), .5)

    def test_invalid_features_fail(self):
        for x in ([[0., 0.]], [[float('nan'), 0.]], [], [1., 2.]):
            with self.subTest(x=x), self.assertRaises(ValueError):
                unit_rows(x)

    def test_background_recurrence_counterexample(self):
        bank = np.array([[1., 0.]] * 9 + [[0., 1.]])
        np.testing.assert_allclose(recurrence_original(np.eye(2), bank), [.9, .1])

    def test_loo_excludes_whole_video_and_equal_video_weights(self):
        features = {'a': np.array([[1., 0.]] * 50), 'b': np.array([[0., 1.]]),
                    'c': np.array([[1., 0.]] * 100)}
        values, metadata = recurrence_loo_video('a', features, {'a': 0, 'b': 0, 'c': 0})
        np.testing.assert_allclose(values, .5)
        self.assertEqual(metadata, {'peers': ['b', 'c'], 'global_fallback': False})

    def test_singleton_fallback_is_visible(self):
        features = {'a': np.eye(2), 'b': np.array([[1., 0.]])}
        values, metadata = recurrence_loo_video('a', features, {'a': 0, 'b': 1})
        np.testing.assert_allclose(values, [1., 0.])
        self.assertTrue(metadata['global_fallback'])
        with self.assertRaises(ValueError):
            recurrence_loo_video('a', features, {'a': 0, 'b': 1}, global_fallback=False)

    def test_loo_rejects_unknown_or_mismatched_bank(self):
        with self.assertRaises(ValueError):
            recurrence_loo_video('a', {'a': np.eye(2)}, {'a': 0, 'test': 0})


class PercentileAndRewardTests(unittest.TestCase):
    def test_midrank_hand_example_and_global_mean(self):
        q = midrank_percentile([3, 1, 1, 8])
        np.testing.assert_allclose(q, [.625, .25, .25, .875])
        self.assertEqual(q.mean(), .5)
        np.testing.assert_array_equal(midrank_percentile([2] * 7), [.5] * 7)

    def test_midrank_only_exact_ties(self):
        q = midrank_percentile([.1, .1 + 1e-12])
        np.testing.assert_allclose(q, [.25, .75])

    def test_midrank_rejects_nonfinite_and_empty(self):
        for x in ([], [np.inf], [[1.]]):
            with self.assertRaises(ValueError):
                midrank_percentile(x)

    def test_strict_json_rejects_invalid_forms(self):
        bad = ['{"scores":[NaN]}', '{"scores":[true]}', '{"scores":[1.0]}',
               '{"scores":[101]}', '{"scores":[-1]}', '{"scores":[]}',
               '{"scores":[1],"extra":0}', '{"scores":[1],"scores":[2]}',
               '```json\n{"scores":[1]}\n```', '{"scores":[1]} extra',
               '{"scores":[1]', '[1]', '{"scores":[Infinity]}']
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_scores(text, 1)

    def test_parser_preserves_integer_extremes(self):
        np.testing.assert_array_equal(parse_scores('{"scores":[0,100]}', 2), [0., 1.])

    def test_parser_rejects_invalid_dimensions(self):
        for count, maximum in ((0, 100), (1, 0), (True, 100)):
            with self.assertRaises(ValueError):
                parse_scores('{"scores":[1]}', count, maximum)

    def test_reward_hand_values_and_reversal(self):
        best = reward_parts('{"scores":[0,100]}', [0, 1])
        reverse = reward_parts('{"scores":[100,0]}', [0, 1])
        constant = reward_parts('{"scores":[50,50]}', [0, 1])
        self.assertEqual((best.total, best.ranking, best.calibration), (1., 1., 1.))
        self.assertEqual((reverse.total, reverse.ranking, reverse.calibration), (0., 0., 0.))
        self.assertAlmostEqual(constant.total, .55)
        self.assertEqual(constant.ranking, .5)

    def test_weighted_rank_and_ignored_target_ties(self):
        q = [.1, .1, .9]
        result = reward_parts('{"scores":[80,20,90]}', q)
        self.assertEqual(result.ranking, 1.)
        no_order = reward_parts('{"scores":[80,20,90]}', [.5, .5, .5])
        self.assertEqual(no_order.ranking, .5)

    def test_invalid_completion_distinct_from_dataset_error(self):
        result = reward_parts('truncated', [.2, .8])
        self.assertEqual(result.total, -1.)
        self.assertFalse(result.valid)
        for q in ([], [np.nan], [-.1], [1.1]):
            with self.assertRaises(ValueError):
                reward_parts('truncated', q)

    def test_calibration_only_reward(self):
        x = reward_parts('{"scores":[0,100]}', [.25, .75], rank_weight=0)
        self.assertEqual(x.total, .9375)

    def test_rewards_bounded_for_many_discrete_outputs(self):
        q = [.2, .5, .8]
        for numbers in itertools.product((0, 50, 100), repeat=3):
            r = reward_parts(json.dumps({'scores': numbers}), q)
            self.assertTrue(0 <= r.total <= 1)
            self.assertTrue(r.valid)

    def test_quantization_error_bound_and_created_ties(self):
        q = midrank_percentile(np.arange(250))
        quantized = np.rint(100 * q) / 100
        self.assertLessEqual(np.max(abs(quantized - q)), .005 + 1e-15)
        self.assertLess(len(np.unique(quantized)), len(q))

    def test_population_standard_deviation_and_constant_group(self):
        a = group_advantages(np.array([[0., 2.], [3., 3.]]), eps=0)
        np.testing.assert_array_equal(a, [[-1., 1.], [0., 0.]])
        with self.assertRaises(ValueError):
            group_advantages(np.array([[1.]]))


class GrpoTests(unittest.TestCase):
    def make_inputs(self):
        new = torch.zeros(4, 3, requires_grad=True)
        old = torch.zeros(4, 3, requires_grad=True)
        ref = torch.zeros(4, 3, requires_grad=True)
        rewards = torch.tensor([0., 1., .25, .75], requires_grad=True)
        mask = torch.tensor([[1, 1, 0]] * 4)
        return new, old, ref, rewards, mask

    def test_old_ref_reward_detached_and_masked_current_grad(self):
        new, old, ref, rewards, mask = self.make_inputs()
        loss = grpo_loss(new, old, ref, rewards, mask)
        self.assertAlmostEqual(loss.item(), 0., places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(new.grad).all())
        self.assertGreater(new.grad[0, 0].item(), 0)
        self.assertLess(new.grad[1, 0].item(), 0)
        self.assertTrue(torch.all(new.grad[:, 2] == 0))
        for stopped in (old, ref, rewards):
            self.assertIsNone(stopped.grad)

    def test_completion_lengths_normalized_per_sequence(self):
        new, old, ref, rewards, _ = self.make_inputs()
        mask = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]])
        grpo_loss(new, old, ref, rewards, mask, beta=0).backward()
        self.assertAlmostEqual(new.grad[0].sum().abs().item() / new.grad[1].sum().abs().item(), 1., places=6)
        self.assertAlmostEqual(new.grad[0, 0].abs().item() / new.grad[1, 0].abs().item(), 2., places=6)

    def test_constant_reward_zero_signal_but_possible_kl(self):
        new = torch.full((4, 2), -.5, requires_grad=True)
        old, ref = new.detach().clone(), torch.zeros(4, 2)
        loss = grpo_loss(new, old, ref, torch.ones(4), torch.ones(4, 2))
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.any(new.grad != 0))

    def test_constant_reward_and_matching_reference_has_zero_grad(self):
        new, old, ref, _, mask = self.make_inputs()
        grpo_loss(new, old, ref, torch.ones(4), mask).backward()
        self.assertTrue(torch.all(new.grad == 0))

    def test_invalid_mask_reward_or_empty_completion_rejected(self):
        args = self.make_inputs()
        for bad_mask in (torch.zeros(4, 3), torch.full((4, 3), .5)):
            with self.assertRaises(ValueError):
                grpo_loss(*args[:-1], bad_mask)
        bad_rewards = torch.tensor([0., 1., float('nan'), .5])
        with self.assertRaises(ValueError):
            grpo_loss(*args[:3], bad_rewards, args[4])

    def test_nonfinite_log_probability_rejected(self):
        new, old, ref, rewards, mask = self.make_inputs()
        with torch.no_grad():
            new[0, 0] = float('inf')
        with self.assertRaises(FloatingPointError):
            grpo_loss(new, old, ref, rewards, mask)


if __name__ == '__main__':
    unittest.main()
