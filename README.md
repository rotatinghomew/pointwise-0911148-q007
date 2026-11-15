# 企业试用信号研究台

本项目用于企业试用信号研究台的业务资料核对与分析留存。提供 `bag_coverage` 覆盖核对与 `trial_conversion` 企业成交预测/行为解释，业务定义见 `docs/data-contract.md`。示例数据在本地生成，分析不访问远程业务系统。

## 运行环境
Python 3.11；数值库固定在 `requirements.txt`。Dockerfile 通过不可变镜像摘要锁定已验证的 Python 3.11.14，代码兼容控制台 Python 3.11.2。依赖安装属于环境准备，不属于服务启动；已经准备过的环境不再联网下载。

```sh
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 -m app demo
python3 -m app score --as-of 2026-09-30T23:59:59Z --list-version sales-list-2026-09
python3 -m app serve
```

服务默认监听本机 8080，`GET /health` 检查存活，`GET /v1/catalog` 查看当前分析类型。`POST /v1/analyses` 接收 `request_id`、`operation`、`as_of` 和可选 `options` 对象，返回保存后的分析；`GET /v1/analyses/{request_id}` 打开历史记录。数据文件由 `--source` 指定，SQLite 由 `--database` 指定，参数放在子命令前。`POST /v1/actions` 调用领域操作，已支持的动作见领域模块；没有配置动作的项目会明确拒绝。

## 成交预测与行为解释
`trial_conversion` 以公司（一次试用）为学习单元，用公司完整的可见试用历史训练；公司标签不会摊到员工或单条会话。排序可在试用结束前给出，并附每组会话证据的适用范围。要点：
- 训练只用观察窗口已成熟的公司；未成熟试用不会被当成未成交。
- 特征截止于 `min(as_of, trial_end, signed_at)`，补登会话、签约后使用、试用后会话都会被排除。
- 验证按集团与共享员工分组，避免多账户集团与重复员工虚高成绩。
- 撤回员工会话不入模、不入证据；重新评分须用新 `request_id`。
- `options.list_version` 必填，模型、聚合约定与名单版本三者不一致时拒绝评分。

请求示例：

```json
{"request_id": "conversion-2026-09", "operation": "trial_conversion",
 "as_of": "2026-09-30T23:59:59Z", "options": {"list_version": "sales-list-2026-09"}}
```

## 目录职责
`app/domain.py` 保存当前业务核对与动作，`app/conversion.py` 实现企业成交预测与解释，`app/contracts.py` 处理可见版本，`app/repository.py` 维护输入、分析记录与销售名单版本，`app/service.py` 协调操作和留存，`app/httpd.py` 提供本地接口。`tests` 覆盖业务口径、重复请求、并发回放及真实 HTTP 往返。

## 快速查看
`python3 -m app demo` 从本地数据执行当前覆盖核对并完整打印结果，`python3 -m app score` 输出成交预测排序与解释，一般数秒内完成。清理 `var` 可从干净状态重新运行，历史正式记录应另外留存。`demo` 展示的是覆盖核对能力，不代表已完成其他分析。

## 容器与离线复现
首次在运行前执行 `docker build -t local-analysis .` 准备锁定依赖。
之后 `docker run --rm --network none local-analysis python3 -m unittest discover -s tests -v` 检查领域契约和真实 HTTP 工作流，`docker run --rm --network none local-analysis` 从全新容器输出当前业务核对。
容器不访问业务网络，不需要外部数据库或第二个服务；`requirements.lock` 与正式依赖入口 `requirements.txt` 保留相同精确版本。
当前测试和演示的执行上限各为80秒；新增分析应提供90秒以内可实际展示完整结果的入口，依赖准备在演示前完成。
