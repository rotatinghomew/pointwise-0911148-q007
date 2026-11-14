import tempfile
import unittest
from collections import Counter
from datetime import datetime
from pathlib import Path

from app.contracts import visible
from app.domain import OPERATIONS, seed_dataset
from app.repository import digest
from app.service import Service


class BusinessContractTests(unittest.TestCase):
    def setUp(self):
        self.data = seed_dataset()
        self.operation = next(iter(OPERATIONS))

    def run_check(self):
        return OPERATIONS[self.operation](self.data, {})

    def test_sessions_join_to_company_and_withdrawals_remove_evidence(self):
        ids = {row['account_id'] for row in self.data['accounts']}
        self.assertTrue(all(row['account_id'] in ids for row in self.data['sessions']))
        result = self.run_check()
        self.assertLess(result['sessions'], len(self.data['sessions']))
        self.assertEqual(result['sessions'], sum(result['account_sessions'].values()))

    def test_late_registration_is_not_visible_early(self):
        early = visible(self.data, '2026-09-01')
        self.assertNotIn('late-session', {row['session_id'] for row in early['sessions']})
        self.assertTrue(all('observation_end' in row for row in self.data['accounts']))
        self.assertTrue(any(row['observation_end'] is None for row in self.data['accounts']))


    def test_fixture_timestamps_do_not_arrive_before_events(self):
        from app.contracts import instant
        for records in self.data.values():
            for row in records:
                if 'received_at' not in row:
                    continue
                for key in ('event_at', 'assigned_at', 'signed_at', 'paid_at', 'issued_at'):
                    if row.get(key):
                        self.assertLessEqual(instant(row[key]), instant(row['received_at']))

    def test_seed_and_live_source_have_one_contract(self):
        from app.contracts import load
        self.assertEqual(self.data, load('data/source.json'))
