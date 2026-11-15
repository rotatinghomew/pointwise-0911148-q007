"""企业试用成交预测与公司级会话证据解释。

学习单元是账户（一次企业试用），而不是会话或员工。会话只用来构造该公司在
预测时点之前已经可见的聚合行为特征；公司成交标签不会被摊到个人或单条会话上。

几个不可妥协的口径：

* 特征只使用 ``event_at`` 与 ``received_at`` 都不晚于预测截止点的会话。补登的
  会话、合同签订后的使用、以及试用结束后才到达的记录都不会进入当时的特征。
* 只有试用已结束且观察窗口已成熟（``observation_end`` 已到且不早于
  ``trial_end``）的公司才有训练标签；``observation_end`` 为空表示尚未成熟。
* 验证按商业集团与员工连通分量分组，同一集团或同一员工关联的账户不会同时
  出现在训练折和验证折。同一集团的多个账户在训练时按账户数分摊权重。
* 撤回的员工会话在评分与解释前就被剔除；重新评分不会引用已撤回证据。
"""

import warnings
from collections import Counter, defaultdict

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.contracts import instant, latest
from app.repository import digest

OPERATION = 'trial_conversion'
SCHEMA = 'company-trial-conversion/v1'
ESTIMATOR = {
    'type': 'LogisticRegression',
    'C': 0.5,
    'max_iter': 1000,
    'random_state': 7,
    'solver': 'lbfgs',
}
FOLDS = 5
MIN_SESSIONS_FOR_RECENT = 3

# 会话聚合约定是版本化的业务口径。修改其中任何一条都必须被视为新的版本。
AGGREGATION_RULES = {
    'schema': 'company-trial-session-aggregation/v1',
    'unit': 'account',
    'join': 'sessions.account_id',
    'identity': 'latest revision by received_at',
    'visibility': 'received_at <= as_of',
    'feature_cutoff': 'min(as_of, trial_end, signed_at)',
    'session_inclusion': 'trial_start <= event_at <= feature_cutoff and received_at <= feature_cutoff',
    'withdrawn': 'sessions of a withdrawn employee_hash are never used',
    'immature_observation': 'accounts without matured observation_end never receive a label',
    'post_contract': 'sessions after signed_at are excluded',
    'backfill': 'sessions received after the feature cutoff are excluded',
    'employee_identity': 'same employee_hash across accounts is one person for grouping',
}
AGGREGATION_VERSION = 'agg-' + digest(AGGREGATION_RULES)[:12]

# 顺序属于模型契约，修改必须同时改变 FEATURES 或 ESTIMATOR。
FEATURES = (
    'sessions',
    'employees',
    'sessions_per_employee',
    'active_minutes_mean',
    'active_minutes_recent_mean',
    'active_minutes_growth',
    'exports_per_employee',
    'invites_per_employee',
    'team_plan_share',
    'span_days',
)
FEATURE_LABELS = {
    'sessions': '可见会话数',
    'employees': '参与员工数',
    'sessions_per_employee': '人均会话数',
    'active_minutes_mean': '平均单次活跃分钟',
    'active_minutes_recent_mean': '近期单次活跃分钟',
    'active_minutes_growth': '活跃度前后变化',
    'exports_per_employee': '人均导出次数',
    'invites_per_employee': '人均邀请次数',
    'team_plan_share': '团队版会话占比',
    'span_days': '会话覆盖天数',
}
MODEL_VERSION = 'mdl-' + digest({
    'schema': SCHEMA,
    'features': list(FEATURES),
    'aggregation': AGGREGATION_VERSION,
    'estimator': ESTIMATOR,
})[:12]

LIMITATIONS = (
    '结论来自公司级会话聚合，只描述与该次试用成交相关的模式，不代表任何个人促成了合同。',
    '成交标签只覆盖观察窗口已成熟的历史试用；未成熟试用不会被当作未成交样本。',
    '特征是试用期内的可见行为，未包含行业、合同金额、销售动作等未观测因素，不能解释因果。',
    '排序只在当前模型版本、会话聚合约定与名单版本三者一致时适用；换版本需重新评分。',
)


def _iso(value):
    return value if value is None else str(value)


def _withdrawn_employees(snapshot):
    return {row['employee_hash'] for row in latest(snapshot.get('withdrawals', []), ['employee_hash'])}


def _feature_cutoff(account, signed_at, as_of):
    points = [as_of, instant(account['trial_end'])]
    if signed_at:
        points.append(instant(signed_at))
    return min(points)


def _window(account, sessions, cutoff):
    """公司在该预测窗口内可见的会话：采用、因补登排除、因窗口后发生排除。

    会话身份取“截至窗口的版本”：窗口之后才到达的订正不得回填到当时。
    """
    start = instant(account['trial_start'])
    by_id = {}
    for row in sessions:
        received_at = instant(row['received_at']) if row.get('received_at') else instant(row['event_at'])
        if received_at > cutoff:
            continue
        previous = by_id.get(row['session_id'])
        position = (row.get('revision', 0), row.get('received_at', ''))
        if previous is None or position > (previous.get('revision', 0), previous.get('received_at', '')):
            by_id[row['session_id']] = row
    used, backfilled, post_cutoff = [], 0, 0
    for row in sessions:
        event_at = instant(row['event_at'])
        received_at = instant(row['received_at']) if row.get('received_at') else event_at
        if event_at < start:
            continue
        if event_at > cutoff:
            post_cutoff += 1
        elif received_at > cutoff:
            backfilled += 1
    for row in by_id.values():
        if start <= instant(row['event_at']) <= cutoff:
            used.append(row)
    return used, backfilled, post_cutoff


def _aggregate_features(rows):
    """把一家公司的会话压成一行公司级特征，绝不产生员工级样本。"""
    ordered = sorted(rows, key=lambda row: (row['event_at'], row['session_id']))
    values = []
    for row in ordered:
        features = row.get('features', {})
        values.append((
            float(features.get('active_minutes', 0)),
            float(features.get('exports', 0)),
            float(features.get('invites', 0)),
            str(features.get('plan', 'unknown')),
        ))
    count = len(values)
    employees = {row['employee_hash'] for row in ordered}
    minutes = [value[0] for value in values]
    exports = sum(value[1] for value in values)
    invites = sum(value[2] for value in values)
    teams = sum(1 for value in values if value[3] == 'team')
    if count >= MIN_SESSIONS_FOR_RECENT:
        recent = sum(minutes[-MIN_SESSIONS_FOR_RECENT:]) / MIN_SESSIONS_FOR_RECENT
    else:
        recent = sum(minutes) / count if count else 0.0
    if count >= 2:
        middle = count // 2
        first = sum(minutes[:middle]) / max(1, middle)
        second = sum(minutes[middle:]) / max(1, count - middle)
        growth = (second - first) / max(1.0, abs(first))
    else:
        growth = 0.0
    if count:
        span = (instant(ordered[-1]['event_at']) - instant(ordered[0]['event_at'])).total_seconds() / 86400
    else:
        span = 0.0
    people = max(1, len(employees))
    result = {
        'sessions': float(count),
        'employees': float(len(employees)),
        'sessions_per_employee': count / people,
        'active_minutes_mean': sum(minutes) / count if count else 0.0,
        'active_minutes_recent_mean': recent,
        'active_minutes_growth': growth,
        'exports_per_employee': exports / people,
        'invites_per_employee': invites / people,
        'team_plan_share': teams / count if count else 0.0,
        'span_days': span,
    }
    return {name: round(float(result[name]), 6) for name in FEATURES}


def _session_evidence(rows):
    ordered = sorted(rows, key=lambda row: (row['event_at'], row['session_id']))
    return {
        'session_count': len(ordered),
        'employee_count': len({row['employee_hash'] for row in ordered}),
        'first_session_at': _iso(ordered[0]['event_at']) if ordered else None,
        'last_session_at': _iso(ordered[-1]['event_at']) if ordered else None,
        'plans': dict(sorted(Counter(str(row.get('features', {}).get('plan', 'unknown'))
                                     for row in ordered).items())),
        'session_ids': [row['session_id'] for row in ordered],
    }


def _components(accounts, sessions):
    """把同一集团以及共享员工的账户连成一个分量，供分组验证使用。"""
    parent = {account['account_id']: account['account_id'] for account in accounts}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    by_group = defaultdict(list)
    by_employee = defaultdict(list)
    for account in accounts:
        by_group[account['group_id']].append(account['account_id'])
    for row in sessions:
        if row.get('account_id') in parent:
            by_employee[row['employee_hash']].append(row['account_id'])
    for members in list(by_group.values()) + list(by_employee.values()):
        for member in members[1:]:
            union(members[0], member)
    return {account_id: find(account_id) for account_id in parent}


def _training_rows(accounts, outcomes, sessions_by_account, as_of):
    rows, excluded = [], Counter()
    for account in accounts:
        outcome = outcomes.get(account['account_id'])
        mature = account.get('observation_end') is not None
        observed = mature and instant(account['observation_end']) <= as_of
        ended = instant(account['trial_end']) <= as_of
        if not mature:
            excluded['observation_not_matured'] += 1
            continue
        if not observed:
            excluded['observation_not_reached'] += 1
            continue
        if not ended:
            excluded['trial_not_ended'] += 1
            continue
        if instant(account['observation_end']) < instant(account['trial_end']):
            excluded['observation_shorter_than_trial'] += 1
            continue
        if outcome is None:
            excluded['outcome_missing'] += 1
            continue
        signed_at = outcome.get('signed_at')
        if signed_at and instant(signed_at) > instant(account['observation_end']):
            excluded['signed_after_observation'] += 1
            continue
        cutoff = _feature_cutoff(account, signed_at, as_of)
        used, backfilled, post_cutoff = _window(
            account, sessions_by_account.get(account['account_id'], []), cutoff)
        rows.append({
            'account': account,
            'label': 1 if signed_at else 0,
            'signed_at': signed_at,
            'cutoff': cutoff,
            'features': _aggregate_features(used),
            'evidence': _session_evidence(used),
            'backfilled_excluded': backfilled,
            'post_cutoff_excluded': post_cutoff,
        })
    return rows, excluded


def _open_candidates(accounts, outcomes, as_of):
    """试用尚未结束、且尚无已成熟成交结果的公司。"""
    candidates = []
    for account in accounts:
        if instant(account['trial_start']) > as_of:
            continue
        observation_end = account.get('observation_end')
        if observation_end is not None and instant(observation_end) <= as_of:
            continue
        outcome = outcomes.get(account['account_id'])
        if outcome and outcome.get('signed_at') and instant(outcome['signed_at']) <= as_of:
            continue
        if observation_end is None and as_of <= instant(account['trial_end']):
            status = 'in_trial'
        else:
            status = 'awaiting_observation'
        candidates.append((account, status))
    return candidates


def _matrix(rows):
    return np.asarray([[row['features'][name] for name in FEATURES] for row in rows], dtype=float)


def _model():
    return Pipeline([
        ('scale', StandardScaler()),
        ('logit', LogisticRegression(**{key: value for key, value in ESTIMATOR.items()
                                        if key != 'type'})),
    ])


def _fit(rows, weights):
    if not rows:
        raise ValueError('没有可用于训练的成熟公司样本')
    labels = np.asarray([row['label'] for row in rows], dtype=int)
    if len(set(labels.tolist())) < 2:
        raise ValueError('训练样本只有一种成交结果，无法建立排序模型')
    model = _model()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        model.fit(_matrix(rows), labels, logit__sample_weight=np.asarray(weights, dtype=float))
    return model


def _group_weights(rows, components):
    counts = Counter(components.get(row['account']['account_id']) for row in rows)
    return [1.0 / counts[components.get(row['account']['account_id'])] for row in rows]


def _grouped_folds(rows, components):
    """按分量分层分折：集团与共享员工始终落在同一折。"""
    bucket = defaultdict(list)
    for row in rows:
        bucket[(row['label'], components.get(row['account']['account_id']))].append(row)
    positive = sorted({key[1] for key in bucket if key[0] == 1})
    negative = sorted({key[1] for key in bucket if key[0] == 0})
    splits = min(FOLDS, len(positive), len(negative))
    if splits < 2:
        return []
    fold_of = {}
    for label, group_ids in ((1, positive), (0, negative)):
        for position, group_id in enumerate(group_ids):
            fold_of[group_id] = position % splits
    folds = []
    for fold in range(splits):
        test = [row for row in rows if fold_of[components.get(row['account']['account_id'])] == fold]
        train = [row for row in rows if fold_of[components.get(row['account']['account_id'])] != fold]
        if len({row['label'] for row in train}) < 2 or not test:
            continue
        folds.append((train, test))
    return folds


def _auc(labels, scores):
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = sum(1.0 if positive > negative else 0.5 if positive == negative else 0.0
               for positive in positives for negative in negatives)
    return wins / (len(positives) * len(negatives))


def _cross_validation(rows, components):
    folds = _grouped_folds(rows, components)
    if not folds:
        return {'folds': 0, 'note': '成熟公司样本不足以做分组交叉验证'}
    labels, scores = [], []
    for train, test in folds:
        model = _fit(train, _group_weights(train, components))
        for row in test:
            labels.append(row['label'])
            scores.append(float(model.predict_proba(_matrix([row]))[0][1]))
    predicted = [1 if score >= 0.5 else 0 for score in scores]
    return {
        'folds': len(folds),
        'observations': len(labels),
        'auc': None if _auc(labels, scores) is None else round(_auc(labels, scores), 6),
        'accuracy': round(sum(int(a == b) for a, b in zip(labels, predicted)) / len(labels), 6),
        'brier': round(sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels), 6),
        'grouping': 'group_id 与共享 employee_hash 的连通分量（同一分量不跨折）',
    }


def _training_median(rows, label, name):
    values = sorted(row['features'][name] for row in rows if row['label'] == label)
    if not values:
        return 0.0
    return values[len(values) // 2]


def list_payload(list_version, account_ids):
    """销售名单版本的规范内容；模型、聚合约定与名单三者绑定到该摘要。"""
    return {'version': list_version, 'account_ids': sorted(set(account_ids))}


def _quantile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[position]


def _class_separation(training, name):
    """两类样本在该特征上的可分性；区间重合说明该模式本身没有区分度。"""
    signed = [row['features'][name] for row in training if row['label'] == 1]
    unsigned = [row['features'][name] for row in training if row['label'] == 0]
    if not signed or not unsigned:
        return 'insufficient', 0.0
    gap = abs(_quantile(signed, 0.5) - _quantile(unsigned, 0.5))
    scale = max(1e-9, max(abs(_quantile(signed, 0.5)), abs(_quantile(unsigned, 0.5)), 1.0))
    overlap = not (_quantile(signed, 0.75) < _quantile(unsigned, 0.25)
                   or _quantile(unsigned, 0.75) < _quantile(signed, 0.25))
    if overlap:
        return 'overlapping', gap / scale
    return 'separating', gap / scale


def _explain(model, row, training):
    scaler = model.named_steps['scale']
    logit = model.named_steps['logit']
    scaled = scaler.transform(_matrix([row]))[0]
    contributions = logit.coef_[0] * scaled
    order = sorted(range(len(FEATURES)), key=lambda index: (-abs(contributions[index]), FEATURES[index]))
    top = []
    for index in order[:4]:
        name = FEATURES[index]
        value = row['features'][name]
        signed = _training_median(training, 1, name)
        unsigned = _training_median(training, 0, name)
        separation, gap = _class_separation(training, name)
        if contributions[index] > 0 and separation == 'separating':
            direction = '支持'
        elif contributions[index] < 0 and separation == 'separating':
            direction = '不支持'
        else:
            direction = '中性'
        if direction == '中性':
            statement = (f'{FEATURE_LABELS[name]}为 {value}；成交与未成交参考中位数分别为 '
                         f'{signed} 与 {unsigned}，两类分布接近，该模式不足以区分，'
                         f'不能作为跟进依据。')
        else:
            statement = (f'{FEATURE_LABELS[name]}为 {value}；成交参考中位数 {signed}，'
                         f'未成交参考中位数 {unsigned}，该模式{direction}本次预测。')
        top.append({
            'pattern': FEATURE_LABELS[name],
            'feature': name,
            'value': value,
            'signed_median': signed,
            'unsigned_median': unsigned,
            'direction': direction,
            'separation': separation,
            'median_gap_ratio': round(gap, 6),
            'magnitude': round(abs(float(contributions[index])), 6),
            'statement': statement,
        })
    signed_share = sum(1 for item in top if item['direction'] == '支持')
    neutral = sum(1 for item in top if item['direction'] == '中性')
    summary = (f'公司层面有 {row["features"]["sessions"]:.0f} 条可见会话、'
               f'{row["features"]["employees"]:.0f} 名参与员工；'
               f'前 {len(top)} 个模式中 {signed_share} 个支持成交倾向、'
               f'{neutral} 个在训练样本中不具区分度。')
    return {
        'basis': 'model_contribution',
        'summary': summary,
        'top_patterns': top,
        'weak_signals': [item['feature'] for item in top if item['direction'] == '中性'],
        'limitations': list(LIMITATIONS),
    }


def _applicability(row, training_rows, strength):
    reasons = []
    if row['features']['sessions'] <= 0:
        reasons.append('试用截止前没有可见会话证据')
    for name in FEATURES:
        low = min(item['features'][name] for item in training_rows)
        high = max(item['features'][name] for item in training_rows)
        value = row['features'][name]
        if not (low <= value <= high):
            reasons.append(f'{FEATURE_LABELS[name]}={value} 超出训练范围 [{low}, {high}]')
    plans = set(row['evidence']['plans'])
    training_plans = {plan for item in training_rows for plan in item['evidence']['plans']}
    unexpected = sorted(plans - training_plans)
    if unexpected:
        reasons.append('出现训练中未见过的试用等级：' + ','.join(unexpected))
    if training_rows and instant(row['account']['trial_start']) < min(
            instant(item['account']['trial_start']) for item in training_rows):
        reasons.append('试用开始时间早于训练样本范围')
    if strength['level'] != 'usable':
        reasons.append('整体验证' + strength['level'] + '：' + strength['note'])
    return ('supported' if not reasons else 'limited'), reasons


def _signal_strength(metrics):
    """把交叉验证结果翻译成销售能直接理解的适用范围提示。"""
    auc = metrics.get('auc')
    if auc is None:
        return {'level': 'insufficient', 'note': '分组交叉验证不足，排序只能作内部参考'}
    if auc >= 0.75:
        return {'level': 'usable', 'note': '分组交叉验证表现尚可，可在适用范围内跟进'}
    if auc >= 0.6:
        return {'level': 'weak', 'note': '分组交叉验证区分度有限，建议结合人工判断使用'}
    return {'level': 'unreliable',
            'note': ('分组交叉验证接近随机（AUC=' + str(round(auc, 3)) +
                     '），当前样本不足以支持任何公司的优先跟进排序')}


def _round(value, digits=6):
    return round(float(value), digits)


def trial_conversion(snapshot, options):
    if not isinstance(options, dict):
        raise ValueError('分析选项须为对象')
    list_version = options.get('list_version')
    if not isinstance(list_version, str) or not list_version.strip():
        raise ValueError('缺少销售名单版本 list_version')
    list_version = list_version.strip()
    if options.get('model_version') not in (None, MODEL_VERSION):
        raise ValueError('模型版本与会话聚合约定不一致')
    if options.get('aggregation_version') not in (None, AGGREGATION_VERSION):
        raise ValueError('会话聚合约定版本不一致')
    as_of = instant(options['as_of']) if options.get('as_of') else None
    if as_of is None:
        raise ValueError('缺少预测截止时间 as_of')

    accounts = latest(snapshot.get('accounts', []), ['account_id'])
    # 会话的身份版本必须在每个预测窗口内分别决定：截止后才到达的订正
    # 不能覆盖当时可见的旧版，否则等于使用了未来信息。
    sessions = snapshot.get('sessions', [])
    outcomes = {row['account_id']: row for row in latest(snapshot.get('outcomes', []), ['account_id'])}
    withdrawn = _withdrawn_employees(snapshot)
    withdrawn_sessions = sum(1 for row in sessions if row['employee_hash'] in withdrawn)
    sessions = [row for row in sessions if row['employee_hash'] not in withdrawn]

    account_ids = {account['account_id'] for account in accounts}
    if any(row.get('account_id') not in account_ids for row in sessions):
        raise ValueError('会话所属企业不存在')
    sessions_by_account = defaultdict(list)
    for row in sessions:
        sessions_by_account[row['account_id']].append(row)

    training_rows, excluded = _training_rows(accounts, outcomes, sessions_by_account, as_of)
    if not training_rows:
        raise ValueError('没有观察成熟的公司样本可用于训练')

    components = _components(accounts, [row for row in sessions
                                        if instant(row['received_at']) <= as_of])
    metrics = _cross_validation(training_rows, components)
    strength = _signal_strength(metrics)
    model = _fit(training_rows, _group_weights(training_rows, components))

    ranked, limited = [], []
    list_ids = []
    for account, status in _open_candidates(accounts, outcomes, as_of):
        cutoff = _feature_cutoff(account, None, as_of)
        used, backfilled, post_cutoff = _window(
            account, sessions_by_account.get(account['account_id'], []), cutoff)
        evidence = _session_evidence(used)
        row = {'account': account, 'label': None, 'signed_at': None, 'cutoff': cutoff,
               'features': _aggregate_features(used), 'evidence': evidence,
               'backfilled_excluded': backfilled, 'post_cutoff_excluded': post_cutoff}
        scope, reasons = _applicability(row, training_rows, strength)
        explanation = _explain(model, row, training_rows)
        identity = components.get(account['account_id'])
        identity_rows = [item for item in training_rows
                         if components.get(item['account']['account_id']) == identity]
        item = {
            'account_id': account['account_id'],
            'group_id': account['group_id'],
            'score': _round(float(model.predict_proba(_matrix([row]))[0][1])),
            'applicability': scope,
            'applicability_reasons': reasons,
            'trial': {
                'start': account['trial_start'],
                'end': account['trial_end'],
                'stage_days': max(0, (min(as_of, instant(account['trial_end']))
                                      - instant(account['trial_start'])).days),
                'days_remaining': max(0, (instant(account['trial_end']) - as_of).days),
                'status': status,
            },
            'evidence': {
                **evidence,
                'withdrawn_sessions_excluded': withdrawn_sessions,
                'backfilled_sessions_excluded': row['backfilled_excluded'],
                'post_cutoff_sessions_excluded': row['post_cutoff_excluded'],
            },
            'explanation': explanation,
            'scope_note': ('同集团/同员工历史账户 ' + str(len(identity_rows)) + ' 个用于训练'
                           if identity_rows else '训练集中没有同集团历史账户'),
            'evidence_refs': [{'session_id': session_id, 'account_id': account['account_id']}
                              for session_id in evidence['session_ids']],
        }
        list_ids.append(account['account_id'])
        if scope == 'supported':
            ranked.append(item)
        else:
            limited.append(item)

    ranked.sort(key=lambda item: (-item['score'], item['account_id']))
    limited.sort(key=lambda item: (-item['score'], item['account_id']))
    for position, item in enumerate(ranked, start=1):
        item['rank'] = position
    list_ids.sort()
    list_digest = digest(list_payload(list_version, list_ids))
    if options.get('list_digest') not in (None, list_digest):
        raise ValueError('销售名单版本与实际名单内容不一致，请使用正在使用的名单版本重新评分')
    if options.get('list_accounts') is not None and sorted(set(options['list_accounts'])) != list_ids:
        raise ValueError('销售名单账户与当前可见试用不一致，请使用正在使用的名单版本重新评分')

    return {
        'operation': OPERATION,
        'as_of': options['as_of'],
        'versions': {
            'model': MODEL_VERSION,
            'aggregation': AGGREGATION_VERSION,
            'list': list_version,
            'consistent': True,
        },
        'aggregation_rules': AGGREGATION_RULES,
        'list': {
            'version': list_version,
            'digest': list_digest,
            'account_ids': list_ids,
        },
        'training': {
            'unit': 'account',
            'accounts': len(training_rows),
            'groups': len({components.get(row['account']['account_id']) for row in training_rows}),
            'positive': sum(1 for row in training_rows if row['label'] == 1),
            'negative': sum(1 for row in training_rows if row['label'] == 0),
            'excluded': dict(sorted(excluded.items())),
            'withdrawn_sessions_excluded': withdrawn_sessions,
            'weights': '同一集团/员工分量按账户数分摊，避免多账户集团重复计权',
            'samples': [
                {
                    'account_id': row['account']['account_id'],
                    'group_id': row['account']['group_id'],
                    'signed': bool(row['label']),
                    'feature_cutoff': _iso(row['cutoff'].isoformat()),
                    'sessions_used': row['evidence']['session_count'],
                    'backfilled_sessions_excluded': row['backfilled_excluded'],
                    'post_cutoff_sessions_excluded': row['post_cutoff_excluded'],
                }
                for row in sorted(training_rows, key=lambda item: item['account']['account_id'])
            ],
        },
        'validation': metrics,
        'scope': {
            'eligible_status': ['in_trial', 'awaiting_observation'],
            'training_labels': '仅 observation_end 已成熟且不早于 trial_end 的公司',
            'feature_window': AGGREGATION_RULES['feature_cutoff'],
            'not_individual_causation': True,
            'signal_strength': strength,
            'notes': list(LIMITATIONS),
        },
        'ranking': ranked,
        'limited': limited,
        'counts': {'ranked': len(ranked), 'limited': len(limited)},
    }
