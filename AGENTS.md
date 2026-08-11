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
  mapping，并在后续 paper genome list batch 推进到 commit
  `1ab309c3275f21e3f4e7346e8d0340894a7507cc`；root `manifests/data_assets.yaml`
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
- **2026-08-11 六仓库文件系统复核：** anti-bloat schema v2 额外拒绝同一仓库内 >=1 KiB exact duplicate 和
  未登记的顶层目录，并报告每个顶层路径的 file/byte 分布、>=500-line source review candidates 与 80% soft
  limit alerts。当前六棵发布树均无 nonignored untracked file、无仓库内重复；root/Core/MDLM file-count 与 Evo2
  bytes 进入软警戒但未越界，处理原则固定在 `docs/REPOSITORY_HYGIENE.md`。DLM-Pretraining 与 downstream
  MDLM 之间少量相同 upstream runtime 文件只作信息报告，因两者必须独立 clone/install，不得为表面去重建立
  cross-submodule import。Core 的 16 个 tracked experiment directories 均有 README；本机旧 producer 中的
  unfinished reviewer files 不得混入 clean public gitlink。该复核同时发现 MDLM fixed-epsilon scorer 改名
  manifest 未进入 generated lineage ledger；MDLM `26e414b` 已重建四份 ledger outputs，176/176 tracked
  code/config assets 覆盖且 118 tests 通过，不涉及模型或接口变更。
- Paper-data capsule 的 machine-readable staging ledger 固定为 `manifests/paper_data_capsule_plan.json`，解释见
  `docs/PAPER_DATA_CAPSULE_PLAN.md`。Classification `random_state=42` folds 与 2026 fixed MIC reconstruction 可作为
  exact frozen assets；2025 strain-wise MIC membership 未恢复，synergy seed-0 仅与 archived counts 一致，二者
  不得写成 exact historical split。任何 model-ready table 必须先按 source/private-public status 分区并完成
  redistribution record，再进入唯一一份外部 Zenodo capsule；不得复制到 Core 或 root Git。
- **2026-08-11 classification capsule 发布：** 不新建第二个 Zenodo 项目；所有 paper-data 补充均作为现有
  concept DOI `10.5281/zenodo.15612047`（首个 version DOI `10.5281/zenodo.15612048`）的新版本发布。
  Canonical builder 为 `python scripts/build_classification_capsule.py --source-root PATH --output OUTSIDE_GIT`，
  source ledger 为 `manifests/classification_capsule_sources.json`，独立指标入口为
  `scripts/recompute_classification_metrics.py`。同一系列的 v2.0.0 已公开为 version DOI
  `10.5281/zenodo.21882300`：classification archive 为 1,317,912 bytes、SHA-256
  `6d053c68ef21afd37d0c7bb76d555c55073513db3785238ace0a7055ea203f68`；三个 exact eligible folds、九份
  normalized predictions、shared-row AUPRC/AUROC、内部 `SHA256SUMS`、无绝对路径、authenticated draft
  与 unauthenticated public fresh-download 均已通过。该版本同时以 canonical filename 发布 9,168,011,488-byte
  fixed-`t=1e-3` all-peptide MIC candidate scorer；本地 SHA-256 为
  `c0d7c2be49ef179a25a19dcd9c54c592c282b6961e51aff60e95fabc13786802`，Zenodo 与本地 MD5 均为
  `c65712310d86091128b5591b234cc401`。该大权重未完成一次完整 public fresh-download SHA-256，文档不得把
  server MD5/local SHA 校验升级为该结论。Zenodo 文件清单和边界固定在
  `manifests/zenodo_release_21882300.json`；旧误导性 checkpoint filename 不得重新出现。
- **2026-08-11 论文 Evo-2 基因组清单：** 对外统一称为“paper genome list / 论文基因组清单”。Core
  `1ab309c3275f21e3f4e7346e8d0340894a7507cc` 发布 563-row CSV 与相邻 JSON manifest；MIC、
  classification、synergy 使用数分别为 563/2/100，CSV size/SHA-256 为 171,749 bytes /
  `64323cab44a4a287b0b63e6e60bd7b0270557d5f0ce5715acb651aeb98b1f860`。清单仅包含保守来源标签、
  当前 filename-matched FASTA 身份、embedding hash 和任务使用标记，不含 sequence、tensor、assay label、
  molecule 或 private row；`current_fasta_*` 不得写成已证明的原始 producer input。
- **2026-08-11 fixed MIC reconstruction capsule：** Canonical builder 为
  `python scripts/build_mic_reconstruction_capsule.py --source-root PATH --output OUTSIDE_GIT --version-doi DOI`，
  独立 standard-library checker 为 `scripts/recompute_mic_reconstruction_metrics.py CAPSULE_ROOT`。公开边界固定
  为 2026 post-paper fixed-split reconstruction，不得写成未恢复的 2025 exact membership。Capsule 包含 3 groups ×
  7 members、86,358-row ensemble、normalized labels/predictions、hashed molecule identity、strain ID、route、
  `MIC <= 16 µM` boolean、frozen metrics/bootstrap、fixed membership 和 member registry；不得包含 molecule
  structure/token、exact MIC、source row ID、embedding、checkpoint、optimizer state、private source table 或作者
  绝对路径。每个 member 的 checkpoint hash 若历史 metadata 未记录，必须标为 `not_recorded`，禁止推断。
  该 capsule 已作为现有 Zenodo concept DOI 的 v3.0.0 发布，version DOI 为
  `10.5281/zenodo.21883545`；archive 为 40,177,188 bytes、MD5
  `bbf7e3a1ab36b1bc029163a9e8d3ad30`、SHA-256
  `25e74abde1f01be57e83b22f6bd1633634284e74257d71f3c71864f7c4b9eebc`。Authenticated draft 与
  unauthenticated public download 均匹配，公共下载解包后 30 hashes、21-member ensemble means 和 48 metric
  rows 独立重算通过。Release manifest 为 `manifests/zenodo_release_21883545.json`。
- **2026-08-11 synergy replay capsule：** Core canonical replay 为
  `PYTHONHASHSEED=0 PYTHONPATH=src python scripts/reproduce/replay_synergy_checkpoints.py --asset-root PATH --output-dir OUTSIDE_GIT --device cuda:0 --local-files-only`；root builder/checker 为
  `scripts/build_synergy_replay_capsule.py` 与 `scripts/recompute_synergy_metrics.py`。完整 1 base + 21 member
  hashes 已从 bytes 重算，三折 AUROC/AUPRC 四位小数逐项匹配旧日志，第二次 replay 的四个 CSV
  byte-identical。公开边界始终为 high-confidence seed-0 candidate，不得写成 proven exact 2025 membership；
  公开表不得包含 exact FICI、raw molecule ID/structure、embedding、checkpoint、optimizer state 或绝对路径。
  Capsule 已作为同一 Zenodo concept DOI 的 v4.0.0 发布，version DOI `10.5281/zenodo.21883954`；archive
  205,983 bytes、MD5 `08407d97ab8aee3ea6130e47452aaefb`、SHA-256
  `a40ec811b179782ffd9d2429c2d3d262df0149c3594a286d0f0c666d9c58d70c`。Authenticated draft 与
  unauthenticated public download 一致，fresh extraction 的 7 hashes、7-member means、2,371 rows 与三折
  AUROC/AUPRC 均独立重算通过。Release manifest 为 `manifests/zenodo_release_21883954.json`。
- **2026-08-11 model-ready 数据与 quickstart benchmark 收口：** 同一 Zenodo concept DOI 的 v5.0.0 已公开，
  version DOI 为 `10.5281/zenodo.21891064`。Canonical builder 是
  `python scripts/build_model_ready_capsule.py --source-root PATH --core-root modules/core --output OUTSIDE_GIT`；
  archive 为 3,743,537 bytes，MD5 `e403f6836dd2dccfd2eb8b62addbaad1`，SHA-256
  `ae0c76febd4e0b4d43fd68c8bf3ddfa27fc2251011f88c5f693d9aa27be95901`。公开内容为 105,237 条
  DBAASP-derived MIC、49,330 条 classification、4,285 条 synergy source rows、compact strain mapping 和
  563-row paper genome list；15,718 条 private in-house MIC rows 已全部排除。两次 build byte-identical，
  authenticated draft/unauthenticated public download、内部 hash、row count 和作者路径检查通过。Fresh-cache
  benchmark 已固定在 `manifests/quickstart_benchmarks_2026-08-11.json`：MIC compute 7.27 s / 5.77 GiB peak RSS；
  Generation 256-step compute 40.34 s / 12,281 MiB peak GPU memory。`manifests/paper_result_registry.json` 对未
  记录的历史 checkpoint hash 明确标 `not_recorded`，不得推断。公开链接审计已核验 root/五模块 commits、
  三个 HF model revisions、HF dataset、Zenodo v1--v5 和 concept DOI 均 HTTP 200。
- 完整 source archive canonical 入口为 `python scripts/build_source_archive.py --output PATH.tar.gz`；它只展开
  root `HEAD` 与 `manifests/modules.lock.yaml` 的五个固定 commits，输出 archive、JSON manifest 和 SHA-256。
  `--plan-only` 只核验 locks。归档不得包含 `.git`、checkpoint、embedding、dataset、cache 或 raw outputs。
  归档本身用 `python scripts/check_source_archive.py ARCHIVE.tar.gz` 验收；依赖 Git refs 的 root checkers 只用于
  recursive clone，不能在刻意移除 `.git` 的解压目录中运行。
