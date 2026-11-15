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


## 已有能力与字段边界

sessions.account_id 是所属试用企业，不从会话编号推断关联；employee_hash 跨账户相同表示同一员工。accounts.observation_end 是目前确认能够观测成交的截止日，只有在试用结束且结果观测充分时才可判定未成交；outcomes 缺行仅表示尚无签约记录。bag_coverage 报告撤回后会话覆盖，不含训练、销售排序或行为解释。

## 企业成交预测与行为解释（trial_conversion）

`trial_conversion` 以账户（一次企业试用）为学习单元，会话只用于构造公司级聚合特征，不产生员工级样本，也不把公司成交标签摊到个人或单条会话。

**训练标签。** 只有 `observation_end` 不为空、截至 `as_of` 已到、且不早于 `trial_end` 的公司才进入训练。`observation_end` 为空表示观察尚未成熟，不会被当成未成交。签约时间晚于观察截止、以及尚在试用中的公司也不进入训练。

**特征窗口。** 每家公司的特征截止点为 `min(as_of, trial_end, signed_at)`。只采用 `event_at` 落在 `trial_start` 到该截止点之间、且 `received_at` 不晚于该截止点的会话。由此排除：
- 补登会话：事件发生时点已知、但到截止点后才收到的记录；
- 合同签订后的使用；
- 试用结束后才发生的会话；
- 在同一会话编号上、截止点之后才到达的订正（会话身份版本按预测窗口分别取，不把未来订正回填到当时）。

**撤回。** 所有撤回员工的会话在训练、评分与解释前统一剔除；重新评分不会引用已撤回证据。撤回记录本身也按 `received_at` 受观察截止约束。

**验证。** 按“同集团 + 共享 employee_hash”的连通分量做分组交叉验证，同一集团或同一员工关联的多个账户不会同时进入训练折与验证折，避免同集团多账户和重复员工制造偏高的验证成绩。训练时同一分量内的账户按账户数分摊权重。输出 AUC、准确率与 Brier 分数，并给出 `signal_strength`：AUC 低于 0.6 时明确标注为不可靠，且不会有公司被判为 `supported`。

**排序与适用范围。** 只对 `as_of` 时试用已开始、尚未成熟、且尚无已知签约结果的账户评分；试用未结束即可给出排序。每个账户带 `applicability`（`supported`/`limited`）与原因：没有可见会话、特征超出训练范围、出现训练中未见过的试用等级、或整体验证强度不足时均为 `limited`。

**解释。** 解释逐公司给出贡献最大的若干会话聚合模式，并标注每个模式在训练样本中的两类中位数与可分性；两类分布重合的模式标为“中性”，不会说成支持或反对。解释明确限定为公司层面相关性，不声称某个员工促成了合同。

**版本一致性。** 模型版本由特征集合、估计器参数与聚合约定摘要生成；聚合约定是列明的规则字典并按内容取摘要。请求可携带 `model_version`、`aggregation_version`、`list_digest` 或 `list_accounts`，与当前计算不一致时拒绝，提示换用正在使用的名单版本重新评分。分析结果返回 `versions`、`aggregation_rules` 与名单摘要；`options.list_version` 必填，服务会登记该名单版本对应的账户内容，同一版本号对应不同名单会被拒绝。

**时点与留存。** 分析仍受 `as_of` 与 `received_at` 约束，重复 `request_id` 返回原结果；撤回或补登后重新评分须使用新的 `request_id`。

