from collections import Counter
from pathlib import Path

from app.contracts import latest, load
from app.conversion import trial_conversion
from app.repository import digest

TABLES = ['accounts', 'sessions', 'outcomes', 'withdrawals']

def bag_coverage(snapshot, options):
    accounts = {row['account_id']: row for row in snapshot['accounts']}
    withdrawn = {row['employee_hash'] for row in snapshot['withdrawals']}
    sessions = [row for row in snapshot['sessions'] if row['employee_hash'] not in withdrawn]
    if any(row['account_id'] not in accounts for row in sessions):
        raise ValueError('会话所属企业不存在')
    by_account = Counter(row['account_id'] for row in sessions)
    return {'groups': len({row['group_id'] for row in accounts.values()}), 'accounts': len(accounts),
            'sessions': len(sessions), 'employees': len({row['employee_hash'] for row in sessions}),
            'known_outcomes': len(snapshot['outcomes']), 'withdrawals': len(withdrawn),
            'account_sessions': dict(sorted(by_account.items()))}

def action(repository, request):
    raise ValueError('尚未登记该业务的可写操作')

OPERATIONS = {'bag_coverage': bag_coverage, 'trial_conversion': trial_conversion}


def seed_dataset():
    return load(Path(__file__).resolve().parents[1] / 'data/source.json')
