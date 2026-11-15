import copy
import unittest

from app.contracts import visible
from app.conversion import (
    AGGREGATION_VERSION,
    FEATURES,
    MODEL_VERSION,
    trial_conversion,
)


class ConversionTests(unittest.TestCase):
    def setUp(self):
        from app.domain import seed_dataset
        self.data = seed_dataset()
        self.as_of = '2026-09-30T23:59:59Z'
        self.snapshot = visible(self.data, self.as_of)

    def run_check(self, snapshot=None, **options):
        payload = {'as_of': self.as_of, 'list_version': 'sales-list-test'}
        payload.update(options)
        return trial_conversion(self.snapshot if snapshot is None else snapshot, payload)

    def test_training_unit_is_company_not_session_or_employee(self):
        result = self.run_check()
        self.assertEqual(result['training']['unit'], 'account')
        self.assertEqual(len(result['training']['samples']), result['training']['accounts'])
        self.assertEqual(result['training']['accounts'],
                         result['training']['positive'] + result['training']['negative'])

    def test_immature_trials_are_not_labelled(self):
        result = self.run_check()
        excluded = result['training']['excluded']
        self.assertEqual(excluded.get('observation_not_matured'), 2)
        labelled = {row['account_id'] for row in result['training']['samples']}
        self.assertNotIn('a0-0', labelled)
        self.assertNotIn('a12-0', labelled)

    def test_open_trials_are_scored_before_they_end(self):
        result = self.run_check()
        candidates = {item['account_id']: item for item in result['ranking'] + result['limited']}
        self.assertEqual(set(candidates), {'a0-0', 'a12-0'})
        for item in candidates.values():
            self.assertEqual(item['trial']['status'], 'in_trial')
            self.assertGreater(item['trial']['days_remaining'], 0)
            self.assertIsInstance(item['score'], float)
            if item['applicability'] == 'limited':
                self.assertTrue(item['applicability_reasons'])
        self.assertEqual(sum(result['counts'].values()), len(candidates))
        ordered = [item['score'] for item in result['limited']]
        self.assertEqual(ordered, sorted(ordered, reverse=True))
        if result['scope']['signal_strength']['level'] != 'usable':
            self.assertFalse(result['ranking'])
            self.assertTrue(all(item['applicability'] == 'limited' for item in result['limited']))
        else:
            self.assertTrue(all(item['rank'] == position
                                for position, item in enumerate(result['ranking'], start=1)))

    def test_post_contract_session_is_excluded_from_training_features(self):
        result = self.run_check()
        sample = next(row for row in result['training']['samples'] if row['account_id'] == 'a1-0')
        self.assertEqual(sample['post_cutoff_sessions_excluded'], 1)
        self.assertEqual(sample['sessions_used'], 12)

    def test_backfilled_session_is_excluded_from_open_scoring(self):
        early = visible(self.data, '2026-09-20T00:00:00Z')
        result = trial_conversion(early, {'as_of': '2026-09-20T00:00:00Z',
                                          'list_version': 'sales-list-test'})
        candidate = next(item for item in result['ranking'] + result['limited']
                         if item['account_id'] == 'a0-0')
        self.assertNotIn('late-session', candidate['evidence']['session_ids'])
        self.assertLess(candidate['evidence']['session_count'], 13)

    def test_conversion_rejects_session_received_after_cutoff(self):
        snapshot = copy.deepcopy(self.snapshot)
        late = copy.deepcopy(next(row for row in snapshot['sessions']
                                  if row['account_id'] == 'a12-0'))
        late['session_id'] = 'a12-0-late'
        late['received_at'] = '2026-10-01T00:00:00Z'
        snapshot['sessions'].append(late)
        result = trial_conversion(snapshot, {'as_of': self.as_of,
                                             'list_version': 'sales-list-test'})
        candidate = next(item for item in result['ranking'] + result['limited']
                         if item['account_id'] == 'a12-0')
        self.assertNotIn('a12-0-late', candidate['evidence']['session_ids'])
        self.assertGreaterEqual(candidate['evidence']['backfilled_sessions_excluded'], 1)

    def test_withdrawn_employees_never_enter_features(self):
        result = self.run_check()
        self.assertGreater(result['training']['withdrawn_sessions_excluded'], 0)
        evidence = set()
        for item in result['ranking'] + result['limited']:
            evidence.update(item['evidence']['session_ids'])
        for row in self.snapshot['sessions']:
            if row['employee_hash'] == 'e3-2':
                self.assertNotIn(row['session_id'], evidence)

    def test_rescoring_after_withdrawal_drops_withdrawn_evidence(self):
        withdrawn = copy.deepcopy(self.snapshot)
        sessions = [row for row in withdrawn['sessions'] if row['employee_hash'] != 'e3-2']
        withdrawn['sessions'] = sessions
        rescored = self.run_check(withdrawn)
        evidence = {session_id for item in rescored['ranking'] + rescored['limited']
                    for session_id in item['evidence']['session_ids']}
        for row in self.snapshot['sessions']:
            if row['employee_hash'] == 'e3-2':
                self.assertNotIn(row['session_id'], evidence)

    def test_group_and_shared_employee_do_not_cross_validation_folds(self):
        result = self.run_check()
        validation = result['validation']
        self.assertGreaterEqual(validation['folds'], 2)
        self.assertIsNotNone(validation['auc'])
        self.assertIn('employee_hash', validation['grouping'])

    def test_versions_must_match(self):
        with self.assertRaises(ValueError):
            self.run_check(model_version='mdl-wrong')
        with self.assertRaises(ValueError):
            self.run_check(aggregation_version='agg-wrong')
        result = self.run_check()
        self.assertEqual(result['versions']['model'], MODEL_VERSION)
        self.assertEqual(result['versions']['aggregation'], AGGREGATION_VERSION)

    def test_list_digest_binds_ranking_to_list_contents(self):
        result = self.run_check()
        digest = result['list']['digest']
        self.assertEqual(self.run_check(list_digest=digest)['list']['digest'], digest)
        with self.assertRaises(ValueError):
            self.run_check(list_digest='deadbeef')

    def test_session_revision_arriving_later_does_not_rewrite_the_past(self):
        snapshot = copy.deepcopy(self.snapshot)
        original = next(row for row in snapshot['sessions'] if row['session_id'] == 'a0-0-s5')
        revision = copy.deepcopy(original)
        revision['revision'] = 1
        revision['received_at'] = '2026-10-05T00:00:00Z'
        revision['features'] = {'active_minutes': 900, 'exports': 9, 'invites': 9, 'plan': 'basic'}
        snapshot['sessions'].append(revision)
        result = trial_conversion(snapshot, {'as_of': self.as_of,
                                             'list_version': 'sales-list-test'})
        candidate = next(item for item in result['ranking'] + result['limited']
                         if item['account_id'] == 'a0-0')
        self.assertIn('a0-0-s5', candidate['evidence']['session_ids'])
        self.assertLess(candidate['evidence']['session_count'], 14)
        self.assertLess(candidate['explanation']['top_patterns'][0]['value'], 900)

    def test_explanation_is_company_level_with_limits(self):
        result = self.run_check()
        for item in result['ranking'] + result['limited']:
            explanation = item['explanation']
            self.assertTrue(explanation['top_patterns'])
            self.assertTrue(explanation['limitations'])
            self.assertTrue(any('个人' in note for note in explanation['limitations']))
            for pattern in explanation['top_patterns']:
                self.assertIn(pattern['feature'], FEATURES)
                self.assertEqual(pattern['feature'], pattern['feature'].lower())
                self.assertIn(pattern['direction'], {'支持', '不支持', '中性'})
                if pattern['separation'] == 'overlapping':
                    self.assertEqual(pattern['direction'], '中性')

    def test_weak_validation_is_surfaced_not_hidden(self):
        result = self.run_check()
        strength = result['scope']['signal_strength']
        if result['validation']['auc'] is not None and result['validation']['auc'] < 0.6:
            self.assertEqual(strength['level'], 'unreliable')
            self.assertIn('不足以支持', strength['note'])

    def test_open_candidate_with_known_signed_outcome_is_not_ranked(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot['outcomes'].append({'account_id': 'a12-0', 'signed_at': '2026-09-20',
                                     'received_at': '2026-09-21T00:00:00Z'})
        result = trial_conversion(snapshot, {'as_of': self.as_of,
                                             'list_version': 'sales-list-test'})
        ids = {item['account_id'] for item in result['ranking'] + result['limited']}
        self.assertNotIn('a12-0', ids)

    def test_samples_do_not_mutate_snapshot(self):
        before = copy.deepcopy(self.snapshot)
        self.run_check()
        self.assertEqual(self.snapshot, before)

    def test_late_arrival_is_invisible_to_earlier_request(self):
        early = visible(self.data, '2026-09-10T23:59:59Z')
        result = trial_conversion(early, {'as_of': '2026-09-10T23:59:59Z',
                                          'list_version': 'sales-list-test'})
        candidate = next(item for item in result['ranking'] + result['limited']
                         if item['account_id'] == 'a0-0')
        self.assertNotIn('late-session', candidate['evidence']['session_ids'])
