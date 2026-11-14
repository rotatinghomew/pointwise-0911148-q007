# 业务记录约定

accounts 包含 account_id、group_id、trial_start、trial_end，group_id 是不能拆分的商业集团。sessions 以 session_id 和 account_id 记录所属企业及 employee_hash、event_at、received_at 和 features，若同一员工跨账户活动，仅可作为相关性提示，不能当两个人。
outcomes 的 signed_at 与 received_at 分别为签约时间和得知时间，null 表示尚无合同信息，不代表永不成交。withdrawals 指明员工记录何时撤回，当前评分不得再使用该员工的特征。
features 中的 active_minutes、exports、invites 是当次会话计数；plan 是当时试用等级。现有 bag_coverage 输出集团、公司、员工和会话的覆盖关系以及观察成熟状态。

## 共同版本语义
所有示例记录均为本地合成，不包含真实客户资料。received_at 不晚于观察截止的记录才可见；缺省该字段的配置为静态配置。revision 表示同一业务身份的订正次数，不是新样本。时间不带时区时按 UTC 解读。
分析请求会保存可见数据的内容摘要及完整内容；重复 request_id 与原请求完全相同时返回原结果，即使源文件后来改变。其他请求内容复用该编号返回冲突。新观察须使用新编号。


accounts.observation_end 为本批样例可观测成交的截止日；null 表示尚未成熟。outcomes 为每个企业提供截至该批数据的结果行，signed_at=null 表示截止时未见签约，不等同永久未成交。sessions.account_id 显式关联所属试用企业。


## 快照样例字段补充

以上字段说明对应当前初始快照中的本地合成记录；示例不存在的历史字段不视为隐藏接口要求。
