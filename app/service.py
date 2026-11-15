from app.contracts import instant, load, require_tables, visible
from app.domain import OPERATIONS, TABLES, action
from app.repository import Repository


class Service:
    def __init__(self, source, database):
        self.source = source
        self.repository = Repository(database)

    def run(self, request):
        if not isinstance(request, dict):
            raise ValueError('分析请求须为对象')
        if not isinstance(request.get('request_id'), str) or not request['request_id'].strip():
            raise ValueError('缺少请求编号')
        operation = request.get('operation')
        if operation not in OPERATIONS:
            raise ValueError('未知分析类型')
        instant(request['as_of'])
        previous = self.repository.previous(request['request_id'], request)
        if previous is not None:
            return previous
        dataset = load(self.source)
        require_tables(dataset, TABLES)
        snapshot = visible(dataset, request['as_of'])
        options = request.get('options') or {}
        if not isinstance(options, dict):
            raise ValueError('分析选项须为对象')
        options = dict(options)
        if 'as_of' in options and options['as_of'] != request['as_of']:
            raise ValueError('分析选项的 as_of 与请求截止时间不一致')
        options['as_of'] = request['as_of']
        result = OPERATIONS[operation](snapshot, options)
        if operation == 'trial_conversion':
            version = result['list']['version']
            entry = {'version': version, 'account_ids': result['list']['account_ids']}
            self.repository.save_list(version, entry)
        return self.repository.remember(request['request_id'], request, snapshot, result)

    def apply(self, request):
        return action(self.repository, request)

    def describe(self):
        return {'operations': sorted(OPERATIONS), 'tables': TABLES}
