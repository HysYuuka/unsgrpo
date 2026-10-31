import copy
import json
import unittest

import numpy as np

from unsgrpo.data import (clip_intervals, sample_targets, window_indices,
                          split_by_source_group, audit_split, build_window_record,
                          audit_window_record, validate_rollout_groups,
                          aggregate_windows, match_prediction_rows)


def example(task='grpo'):
    bounds = clip_intervals(19.25)
    return build_window_record(video_id='video01', source_group='original01', split='train',
                               duration=19.25, bounds=bounds,
                               frame_paths=[[f'/frames/{i}_{j}.jpg' for j in range(4)] for i in range(len(bounds))],
                               sampled_pts=sample_targets(bounds), overview_path='/frames/overview.jpg',
                               clip_ids=list(range(2, 10)),
                               pseudo_scores=None if task == 'eval' else np.linspace(.05, .95, len(bounds)), task=task)


class TimelineTests(unittest.TestCase):
    def test_nonintegral_end_and_midpoint_sampling(self):
        bounds = clip_intervals(4.25)
        np.testing.assert_array_equal(bounds, [[0, 2], [2, 4], [4, 4.25]])
        np.testing.assert_allclose(sample_targets(bounds)[-1], [4.03125, 4.09375, 4.15625, 4.21875])

    def test_short_video_no_fake_clips(self):
        bounds = clip_intervals(.1)
        self.assertEqual(len(bounds), 1)
        self.assertEqual(window_indices(len(bounds)), [[0]])
        self.assertTrue(np.all(sample_targets(bounds) < .1))

    def test_windows_cover_entire_video(self):
        for n in range(1, 65):
            windows = window_indices(n)
            self.assertEqual(set(sum(windows, [])), set(range(n)))
            self.assertTrue(all(1 <= len(w) <= 8 for w in windows))
            self.assertEqual(len(windows), len({tuple(w) for w in windows}))
        self.assertEqual(window_indices(13), [list(range(8)), list(range(4, 12)), list(range(5, 13))])

    def test_invalid_timeline_and_window_parameters_fail(self):
        for value in (0, -1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                clip_intervals(value)
        for args in ((0,), (12, 8, 9), (12, 8, 0)):
            with self.assertRaises(ValueError):
                window_indices(*args)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'video_id': f'v{i}', 'source_group': f'g{i // 2}', 'sha256': f'hash{i}'} for i in range(20)]

    def test_source_group_partition_seeded_and_input_order_independent(self):
        result = split_by_source_group(self.rows)
        expected = {r['video_id']: r['split'] for r in result}
        self.assertEqual(expected, {r['video_id']: r['split'] for r in split_by_source_group(list(reversed(self.rows)))})
        self.assertEqual(audit_split(result)['counts'], {'train': 16, 'test': 4, 'dev': 0})
        self.assertTrue(all('split' not in row for row in self.rows))

    def test_source_crossing_and_duplicate_source_assets_fail(self):
        rows = split_by_source_group(self.rows)
        rows[0]['split'] = 'test' if rows[1]['split'] == 'train' else 'train'
        with self.assertRaises(ValueError):
            audit_split(rows)
        assets = copy.deepcopy(self.rows)
        assets[5]['sha256'] = assets[0]['sha256']
        with self.assertRaises(ValueError):
            split_by_source_group(assets)

    def test_duplicate_id_and_single_source_fail(self):
        with self.assertRaises(ValueError):
            split_by_source_group([self.rows[0], self.rows[0]])
        with self.assertRaises(ValueError):
            split_by_source_group(self.rows[:2])

    def test_categories_and_titles_do_not_affect_split(self):
        annotated = [dict(r, category='invisible', title='not used') for r in self.rows]
        self.assertEqual([r['split'] for r in split_by_source_group(annotated)],
                         [r['split'] for r in split_by_source_group(self.rows)])


class PromptIdentityTests(unittest.TestCase):
    def test_grpo_target_only_top_level_and_order_preserved(self):
        row = example()
        self.assertEqual([m['role'] for m in row['messages']], ['system', 'user'])
        self.assertEqual(len(row['videos'][0]), 32)
        self.assertEqual(row['videos'][0][0], '/frames/2_0.jpg')
        np.testing.assert_allclose(row['pseudo_scores'], np.linspace(.05, .95, 10)[2:])
        self.assertNotIn('pseudo_scores', str(row['messages']))
        self.assertIn('2: [4.000000, 6.000000)', row['messages'][1]['content'])

    def test_sft_only_assistant_answer_holds_quantized_targets(self):
        row = example('sft')
        self.assertEqual(row['messages'][-1]['role'], 'assistant')
        answer = json.loads(row['messages'][-1]['content'])['scores']
        self.assertEqual(answer, np.rint(np.asarray(row['pseudo_scores']) * 100).astype(int).tolist())
        self.assertEqual(audit_window_record(row)['n_frames'], 32)

    def test_eval_inputs_contain_neither_pseudo_targets_nor_answers(self):
        row = example('eval')
        self.assertNotIn('pseudo_scores', row)
        self.assertEqual(len(row['messages']), 2)

    def test_injected_prompt_and_assistant_pseudo_answer_fail(self):
        row = example()
        row['messages'][1]['content'] += '\npseudo_scores: [.1, .2]'
        with self.assertRaises(ValueError):
            audit_window_record(row)
        row = example()
        row['messages'].append({'role': 'assistant', 'content': '{"scores":[1]}'})
        with self.assertRaises(ValueError):
            audit_window_record(row)

    def test_training_split_and_forbidden_human_metadata_fail(self):
        row = example()
        row['split'] = 'test'
        with self.assertRaises(ValueError):
            audit_window_record(row)
        row = example()
        row['title'] = 'a label clue'
        with self.assertRaises(ValueError):
            audit_window_record(row)

    def test_media_or_clip_reassignment_fail(self):
        row = example()
        row['videos'][0][0] = '/wrong/video.jpg'
        with self.assertRaises(ValueError):
            audit_window_record(row)
        row = example()
        row['request_id'] = 'other'
        with self.assertRaises(ValueError):
            audit_window_record(row)

    def test_rollout_content_binding_and_tensor_evidence(self):
        rows = [copy.deepcopy(example()) for _ in range(4)]
        self.assertEqual(validate_rollout_groups(rows)['n_groups'], 1)
        with self.assertRaises(ValueError):
            validate_rollout_groups(rows, require_tensor_hash=True)
        for row in rows:
            row['media_tensor_sha256'] = 'identical-processed-tensor-hash'
        self.assertEqual(validate_rollout_groups(rows, require_tensor_hash=True)['n_completions'], 4)
        rows[2]['media_tensor_sha256'] = 'different'
        with self.assertRaises(ValueError):
            validate_rollout_groups(rows, require_tensor_hash=True)

    def test_rollout_same_lengths_but_different_q_rejected(self):
        rows = [copy.deepcopy(example()) for _ in range(4)]
        rows[1]['pseudo_scores'][0] += .01
        with self.assertRaises(ValueError):
            validate_rollout_groups(rows)

    def test_q_consistent_between_overlapping_windows(self):
        args = dict(video_id='v', source_group='s', split='train', duration=20., bounds=clip_intervals(20),
                    frame_paths=[[f'{i}/{j}' for j in range(4)] for i in range(10)],
                    sampled_pts=sample_targets(clip_intervals(20)), overview_path='overview',
                    pseudo_scores=np.linspace(.05, .95, 10))
        left = build_window_record(**args, clip_ids=list(range(8)))
        right = build_window_record(**args, clip_ids=list(range(2, 10)))
        self.assertEqual(left['pseudo_scores'][2:], right['pseudo_scores'][:-2])


class PredictionTests(unittest.TestCase):
    def test_invalid_window_counts_in_denominator(self):
        scores, invalid = aggregate_windows(3, [([0, 1], '{"scores":[100,100]}'), ([1, 2], 'truncated')])
        np.testing.assert_array_equal(scores, [1., .5, 0.])
        self.assertEqual(invalid, .5)

    def test_missing_clip_and_invalid_indices_fail(self):
        for rows in ([([0, 1], '{"scores":[50,50]}')], [([0, 0, 2], '{"scores":[50,50,50]}')],
                     [([0, 1, 3], '{"scores":[50,50,50]}')]):
            with self.assertRaises(ValueError):
                aggregate_windows(3, rows)

    def test_prediction_join_uses_id_and_rejects_append_duplicates(self):
        examples = [{'request_id': 'a'}, {'request_id': 'b'}]
        predictions = [{'request_id': 'b', 'response': 'B'}, {'request_id': 'a', 'response': 'A'}]
        self.assertEqual([r['response'] for r in match_prediction_rows(examples, predictions)], ['A', 'B'])
        with self.assertRaises(ValueError):
            match_prediction_rows(examples, predictions + predictions[:1])
        with self.assertRaises(ValueError):
            match_prediction_rows(examples, predictions[:1])
        with self.assertRaises(ValueError):
            match_prediction_rows(examples, predictions + [{'request_id': 'c', 'response': 'C'}])


if __name__ == '__main__':
    unittest.main()
