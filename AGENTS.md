# ApexOracle super-repo 维护约束

- 本仓库只维护统一入口、固定 submodule gitlinks、资产 manifest、环境说明、bootstrap、quickstarts 和发布文档。
- 不复制 Core、DLM-Pretraining、MDLM、Evo-2 或 Generation 的实现代码。
- `.gitmodules` 只能加入已经存在、可公开 clone 且通过 module-level 验收的 repository；禁止加入浮动或失效 URL。
- 每个 active gitlink 必须在 `manifests/modules.lock.yaml` 记录完整 40-character commit，并由
  `python scripts/check_module_locks.py` 验证。
- checkpoint、embedding、dataset、raw output、cache 和 private assay data 不进入 Git；只在 asset manifest
  登记 URI、revision、SHA-256、许可和发布状态。
- 现有 ApexOracle legacy tree 已由 branch `legacy-monorepo` 与 annotated tag
  `legacy-monorepo-snapshot-2026-08-10` 保存。不得删除或移动这两个远程恢复点。
- 最终 Core 直接复用当前 `DragonDescentZerotsu/Synergy` repository，并在完整 history audit 后重命名为
  `DragonDescentZerotsu/ApexOracle-Core`；不得建立第二份 Core repository。
- 当前发布阶段和剩余 gate 记录在 `docs/RELEASE_STATUS.md`；每次新增 module gitlink 或资产时必须同步更新。
- 发布前运行 `python scripts/check_release_tree.py`、`python scripts/check_module_locks.py` 和
  `python -m pytest -q`；三个入口均通过后才允许更新默认分支。
