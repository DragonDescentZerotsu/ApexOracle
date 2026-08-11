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
- `docs/RELEASE_PROVENANCE.md` 必须区分 release gitlinks、科学实现验收 commits、恢复 refs 和外部资产
  revisions；文档-only module commit 不得被误写为重新验证过的科学实现。
- 论文 Code availability 已固定 Zenodo embedding dataset DOI `10.5281/zenodo.15612048`；README、
  `CITATION.cff`、`manifests/data_assets.yaml` 与 release provenance 必须保持一致，不得再写成没有 Zenodo record。
- README 只允许使用从 legacy history 恢复并由 SHA-256 固定的 `assets/ApexOracle_1.png` 与
  `assets/upenn.png` 两个视觉资产；其他 root binary/data 文件仍由 `python scripts/check_release_tree.py`
  拒绝，不能借 README 美化放宽发布边界。
- 发布前运行 `python scripts/check_release_tree.py`、`python scripts/check_module_locks.py` 和
  `python -m pytest -q`；三个入口均通过后才允许更新默认分支。
- 完整 source archive canonical 入口为 `python scripts/build_source_archive.py --output PATH.tar.gz`；它只展开
  root `HEAD` 与 `manifests/modules.lock.yaml` 的五个固定 commits，输出 archive、JSON manifest 和 SHA-256。
  `--plan-only` 只核验 locks。归档不得包含 `.git`、checkpoint、embedding、dataset、cache 或 raw outputs。
  归档本身用 `python scripts/check_source_archive.py ARCHIVE.tar.gz` 验收；依赖 Git refs 的 root checkers 只用于
  recursive clone，不能在刻意移除 `.git` 的解压目录中运行。
