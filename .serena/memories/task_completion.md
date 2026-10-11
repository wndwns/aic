# 完成检查

- 后端变更：先执行 `python -m compileall backend`，再用 `python backend/server.py` 启动并检查受影响 API 或 `http://127.0.0.1:8000/docs`。
- 前端变更：在后端运行时打开受影响页面；确认 `frontend/app.js` 的 API 路径与 `backend/server.py` 路由一致。
- 数据或模型变更：验证相关 JSON 可加载、模型端点可返回；涉及训练时调用 `POST /api/model/train` 并检查对应状态或结果 API。
- 只运行与改动范围匹配的脚本；项目没有定义统一全量测试、lint 或格式化命令。
# 2026-10-11 今日完成

- 从 GitHub `main@6ca4ba2` 恢复安全副本，核对 `backend/deep_api.py` 与 `deep_service.py` 的真实返回形状。
- 重建正式入口 `frontend/meteo.html` 为 Polar Operations 冰原运营台；只改前端和记录文件，保留 `/api/deep/*`、月份选择、刷新、风险排序、归因、趋势、注意力和月度气象报告。
- 页面形成“风险排序 → 证据回溯 → 人工核查 → 牧区处置建议”闭环。候选风险、未确认来源、NDVI/同比/气候历史缺口明确显示待核查或未接入，不用 0 填充。
- 行业指标状态区已预留 PR-AUC、Hit@3/5、Recall@K、事件命中率和提前量；当前接口没有 event_id 与事件回放结果，因此未伪造数值。
- 文档已同步：`README.md`、`项目全景说明（Codex读）.md`、`AGENTS.md`、`task_plan.md`，并新建 `docs/daily/2026-10-11.md`。
- 验收：`/api/health`、深度 status/risk/fields/attention 返回成功；1440x900、1280x800、390x844 无横向溢出、5 条风险队列可见、点击队列更新证据、无控制台错误。

恢复边界：原工作目录清空后，未提交且未进入 GitHub 的 Trae 产物无法从远端恢复。本记录只登记恢复副本中实际完成并验证的内容。
