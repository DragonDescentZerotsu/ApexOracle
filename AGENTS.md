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
- `v0.2.3` 只收口 downstream reporting/candidate scorer 的误导命名：canonical profile 为
  `fixed_epsilon_non_pad`，本地资产固定 `t=1e-3`，不是精确 clean `t=0`。Core/MDLM/Generation gitlinks
  分别固定到 `1973d2d3cc6b27202a3960c363c207dd030f74e7`、
  `931e3dc09bfc2501809c03dbd016741406950f5f`、`67b593e1a623af3af80c64e263bde527d73d89ef`；checkpoint
  bytes/SHA 与 Generation sampler 权重均未改变。
- **2026-08-10 post-`v0.2.3` reviewer reproducibility batch：** Core `main` 已新增 compact paper strain
  mapping，固定 commit `8751c80cb86c3382a9fc3c8689666e992c0ee7a9`；root `manifests/data_assets.yaml`
  必须登记其 730,151-byte file、SHA-256 `51db55fe...d8f4` 和 1,766 labels/1,769 routes/92,322 routed rows
  scope。Reviewer code/data 回复的 canonical working draft 为 `docs/REVIEWER_CODE_RESPONSE_DRAFT.md`；其中
  `DONE` 与 `OPEN` 必须按 public immutable asset 分开，禁止把 prediction capsule、exact runtime/RAM/VRAM
  或未发布 model-ready tables 提前写成完成。全 checkpoint 上传不是 release gate；固定政策见
  `docs/REPRODUCIBILITY_SCOPE.md`。
- README 只允许使用从 legacy history 恢复并由 SHA-256 固定的 `assets/ApexOracle_1.png` 与
  `assets/upenn.png` 两个视觉资产；其他 root binary/data 文件仍由 `python scripts/check_release_tree.py`
  拒绝，不能借 README 美化放宽发布边界。
- 发布前运行 `python scripts/check_release_tree.py`、`python scripts/check_module_locks.py` 和
  `python scripts/check_repository_bloat.py`、`python -m pytest -q`；四个入口均通过后才允许更新默认分支。
- Repository anti-bloat policy 固定在 `manifests/repository_size_policy.json`，解释与当前基线见
  `docs/REPOSITORY_HYGIENE.md`。任何新 tracked file 默认不得超过 1 MiB；只有精确路径、窄 size cap 和
  明确科学理由的 allowlist 才允许例外。六棵 active trees 中任意 >=20 KiB exact duplicate、checkpoint/cache
  suffix、generated cache/build path、repo file-count/total-byte 超限都会使 CI 失败。Paper model-ready tables、
  sample predictions、embeddings 和 checkpoints 必须外置到 Zenodo/Hugging Face，Git 只保留 compact manifest、
  exporter、split IDs、hash 和 recomputation code。
- Paper-data capsule 的 machine-readable staging ledger 固定为 `manifests/paper_data_capsule_plan.json`，解释见
  `docs/PAPER_DATA_CAPSULE_PLAN.md`。Classification `random_state=42` folds 与 2026 fixed MIC reconstruction 可作为
  exact frozen assets；2025 strain-wise MIC membership 未恢复，synergy seed-0 仅与 archived counts 一致，二者
  不得写成 exact historical split。任何 model-ready table 必须先按 source/private-public status 分区并完成
  redistribution record，再进入唯一一份外部 Zenodo capsule；不得复制到 Core 或 root Git。
- 完整 source archive canonical 入口为 `python scripts/build_source_archive.py --output PATH.tar.gz`；它只展开
  root `HEAD` 与 `manifests/modules.lock.yaml` 的五个固定 commits，输出 archive、JSON manifest 和 SHA-256。
  `--plan-only` 只核验 locks。归档不得包含 `.git`、checkpoint、embedding、dataset、cache 或 raw outputs。
  归档本身用 `python scripts/check_source_archive.py ARCHIVE.tar.gz` 验收；依赖 Git refs 的 root checkers 只用于
  recursive clone，不能在刻意移除 `.git` 的解压目录中运行。
