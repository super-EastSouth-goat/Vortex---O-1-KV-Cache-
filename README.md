# Vortex KV Cache Compression / Vortex KV 缓存压缩实验

**English | 中文 bilingual README**

Vortex is an experimental decode-time KV cache compression prototype for local LLM inference. It explores whether a model can selectively retain only the most useful generated tokens in its KV cache by combining Q-space geometry, local feature jumps, Chinese lexical priors, structural guards, and controlled ablation experiments.

Vortex 是一个面向本地大模型推理的 **生成期 KV Cache 动态压缩实验原型**。它尝试回答一个问题：在生成文本时，模型能否只保留“真正有用”的 token 到 KV cache 里，而把低价值 token 逻辑删除？Vortex 将 Q 空间几何探针、相邻特征跳跃、中文词法先验、结构保护和预算对照实验组合在一起，用于研究 decode-time KV cache compression。

---

## 1. Project Status / 项目状态

**Current version:** Vortex V5.6  
**Model used in experiments:** `google/gemma-2-2b-it`  
**Primary framework:** MLX / `mlx-lm`  
**Primary target:** Apple Silicon local inference experiments

V5.6 is best described as a **research prototype**, not a production-ready cache compressor. It provides a clean ablation harness and attribution pipeline for comparing Vortex against random and simplified baselines under equal KV retention budgets.

V5.6 目前应被视为 **研究原型**，不是生产级 KV 压缩器。它的主要价值是提供一套可解释的消融实验框架：在相同保留预算下，比较 Vortex、随机保留、几何探针、词法/逻辑 guard、去掉 jump、去掉 logic guard 等策略的行为。

---

## 2. Core Idea / 核心思想

During autoregressive decoding, each generated token is written into the KV cache. Standard inference keeps all generated tokens. Vortex asks whether every token deserves to remain in memory.

Vortex assigns each decode token a keep/drop decision using several signals:

1. **Global Q-space divergence**: how far the current token's Q vector is from the prompt centroid.
2. **Local Q-space jump**: how sharply the current token differs from the previous retained token.
3. **Jieba lexical prior**: whether recent token pieces form a valid Chinese lexical phrase.
4. **Logic and boundary guards**: whether the token participates in negation, contrast, causality, time boundaries, or structural punctuation.
5. **Post-hit local fuse**: when an important token is kept, keep a small number of following decode tokens to preserve short-range phrase continuity.
6. **Startup keep**: protect the first few generated tokens after prefill to avoid destabilizing the opening trajectory.

在自回归生成中，每个新生成的 token 都会进入 KV cache。标准推理会保留所有 token。Vortex 追问：**每个 token 都值得被记住吗？**

Vortex 通过多种信号判断当前 decode token 是否应该保留：

1. **全局 Q 空间偏离**：当前 token 的 Q 向量距离 prompt 语义中心有多远。
2. **局部 Q 空间跳跃**：当前 token 与上一个幸存 token 的 Q 方向差异有多大。
3. **Jieba 词法先验**：最近的 token 片段是否构成中文词汇或短语。
4. **逻辑与边界保护**：否定、转折、因果、时间边界、标点和引号等结构信息是否出现。
5. **Post-hit local fuse**：重要 token 命中后，保护后续极少数 token，避免短语被切断。
6. **Startup keep**：prefill 后前几个生成 token 绝对不杀，稳定生成开局。

---

## 3. How Vortex Works / Vortex 如何工作

### 3.1 Prompt centroid / Prompt 中心

During prefill, Vortex probes one transformer layer and computes a normalized centroid from the prompt's Q vectors:

```text
prompt_centroid = mean(normalized_Q(prompt_tokens))
```

在 prefill 阶段，Vortex 在指定 transformer 层读取 prompt token 的 Q 向量，并构造一个归一化后的 prompt centroid。

### 3.2 Decode-time score / Decode 阶段分数

For each decode token, Vortex computes:

```text
base_volume = (1 - cosine(q_current, prompt_centroid)) * 10000
```

A higher value means the current token is more geometrically different from the prompt's average Q-space direction.

每个 decode token 会得到一个几何分数：

```text
base_volume = (1 - cosine(q_current, prompt_centroid)) * 10000
```

分数越高，代表当前 token 在 Q 空间里越偏离 prompt 的平均方向，可能携带更强的新信息。

### 3.3 Local jump / 局部跳跃

Vortex also compares the current Q vector with the previous retained token's Q vector:

```text
local_jump = 1 - cosine(q_current, q_previous_retained)
```

If `local_jump` exceeds `JUMP_THRESHOLD`, Vortex treats the token as a potential semantic or logical discontinuity.

Vortex 还会比较当前 token 与上一个保留 token 的 Q 向量：

```text
local_jump = 1 - cosine(q_current, q_previous_retained)
```

如果该值超过 `JUMP_THRESHOLD`，Vortex 会认为这里可能存在语义转折、逻辑断层或叙事跳跃。

### 3.4 Lexical prior / 词法先验

V5.6 uses a rolling 2/3/4-token window and Jieba segmentation with `HMM=False` to detect Chinese lexical phrases. This is a soft prior: it helps protect meaningful Chinese units, but it is not enough by itself.

V5.6 使用 rolling 2/3/4-token window，并用 Jieba 在 `HMM=False` 下判断中文词法组合。它是一个 soft prior，用于保护中文实体和短语，但实验显示它不能单独支撑生成。

### 3.5 Logical eviction / 逻辑删除

If a token is dropped, Vortex applies a logical KV eviction by rolling back cache offsets:

```text
cache[layer].offset -= 1
```

This makes future decode steps logically ignore or overwrite that token position. This is **logical compression**, not guaranteed physical memory release.

如果 token 被判定为低价值，Vortex 会通过回退 cache offset 进行逻辑删除：

```text
cache[layer].offset -= 1
```

这会让后续 decode 在逻辑上忽略或覆盖该 token 的 KV 位置。注意：这只是 **逻辑压缩**，并不保证 MLX allocator 立即释放物理显存。

---

## 4. V5.6 Default Configuration / V5.6 默认配置

```python
MODEL_ID = "google/gemma-2-2b-it"
TARGET_LAYER = 12
VOLUME_THRESHOLD = 3550.0
JUMP_THRESHOLD = 0.22
FUSE_KEEP = 2
STARTUP_KEEP = 8
LEXICAL_NGRAM_MAX = 4
RANDOM_SEEDS = (1, 2, 3, 4, 5)
MAX_TOKENS = 150
BUDGET_TARGET_RATES = (0.80, 0.70, 0.60)
```

这些参数来自当前 V5.6 诊断实验，并不是最终最优超参。

---

## 5. Strategy Matrix / 策略矩阵

V5.6 evaluates the following families:

| Strategy | Description |
|---|---|
| `baseline` | Native generation; no eviction. |
| `random_p` | Random keep/drop using probability matched to Vortex retention. |
| `random_k` | Exact random keep count matched to a target budget. |
| `geo_no_jump` | Q-space volume only. |
| `geo` | Q-space volume + local jump. |
| `jieba_only` | Jieba lexical prior only. |
| `guard_only` | Jieba + logic/boundary guards, no Q probe. |
| `vortex_no_logic` | Full Vortex minus explicit logic guard. |
| `vortex_no_jump` | Full Vortex minus local Q-space jump. |
| `vortex` | Full Vortex: geometry + jump + lexical prior + guards + fuse. |
| `vortex@80/70/60` | Full Vortex under fixed budget quota. |
| `random_k@80/70/60` | Exact random budget baselines. |

V5.6 会评估自然策略与固定预算策略两类实验，尤其关注 `80% / 70% / 60%` 三档保留率下的同预算比较。

---

## 6. Metrics / 指标说明

Vortex reports both system metrics and quality metrics.

### System metrics / 系统指标

| Metric | Meaning |
|---|---|
| `kept_count` | Number of decode tokens logically retained. |
| `evicted_count` | Number of decode tokens logically evicted. |
| `retention_rate` | `kept_count / processed_decode_tokens`. |
| `cache_offset_mean` | Mean cache offset after generation. Approximates logical KV length. |
| `active_mb` / `peak_mb` / `cache_mb` | MLX memory telemetry. |
| `tokens_per_sec` | Decode speed. |

### Quality metrics / 质量指标

| Metric | Meaning |
|---|---|
| `output_entity_recall` | Whether key entities appear in final generated text. |
| `retained_entity_recall` | Whether key entities remain in retained KV stream. |
| `output_logic_recall` | String-level logic marker recall. |
| `output_logic_category_recall` | Category-level logic recall, e.g. negation / contrast / causality / temporal. |
| `baseline_similarity` | Bag/multiset similarity against baseline output. |
| `baseline_lcs` | Order-sensitive LCS ratio against baseline output. |
| `repeat_3gram_rate` / `repeat_4gram_rate` | Repetition diagnostics. |
| `reason_counts` | Priority decision attribution. |
| `feature_hit_counts` | Non-exclusive feature hit histogram. |

其中 `repeat_4gram_rate` 很重要，因为压缩后模型容易进入重复循环，仅靠 LCS 或 entity recall 可能看不出来。

---

## 7. Current Findings from V5.6 / V5.6 当前发现

V5.6 shows that the experiment harness is mature enough to perform fair budget comparisons. The key findings are:

1. **Exact Random-k alignment works.** Random-k now correctly matches the target keep count, including startup-protected tokens.
2. **Natural Vortex is stable but conservative.** Around 88% retention, Vortex preserves key entities and logic categories, but does not clearly dominate random baselines on LCS/F1.
3. **Jump is critical for the natural policy.** Removing local Q-space jump causes severe collapse and repetition.
4. **Logic guard is not yet independently proven on the current prompt.** `vortex_no_logic` can match or outperform full Vortex on some similarity metrics in this single prompt.
5. **Vortex@80 is the most promising compressed setting so far.** At 70% and 60%, repetition rises sharply under the current budget controller.
6. **Logical KV compression is visible, physical memory savings are not yet proven.** Cache offsets shrink, but short 150-token experiments do not yet show strong peak memory separation.

V5.6 显示实验框架已经成熟，可以进行公平预算比较。当前主要结论：

1. **Random-k 精确对齐已修复。** 含 `STARTUP_KEEP` 的预算也能严格对齐。
2. **自然 Vortex 稳定但偏保守。** 约 88% 保留率下实体和逻辑类别保持良好，但 LCS/F1 没有明显赢过随机。
3. **Jump 是自然策略主梁。** 去掉局部 Q-space jump 后，模型明显崩溃并出现重复。
4. **Logic guard 在当前单 prompt 上还没有独立证明价值。** 需要更多逻辑测试集。
5. **Vortex@80 是目前最有希望的压缩点。** @70/@60 开始出现明显重复。
6. **逻辑 KV 压缩成立，但物理显存收益尚未证明。** cache offset 下降，但短生成长度不足以证明真实显存下降。

---

## 8. Installation / 安装

This project currently targets Apple Silicon with MLX.

```bash
pip install mlx mlx-lm jieba
```

Depending on your environment, you may also need:

```bash
pip install pandas
```

本项目主要面向 Apple Silicon + MLX 环境。

---

## 9. Running the Experiment / 运行实验

```bash
python vortex_generate_v5_6.py
```

The script will:

1. Load `google/gemma-2-2b-it`.
2. Run baseline and ablation strategies.
3. Run random multi-seed baselines.
4. Run budget curves at 80%, 70%, and 60% retention.
5. Save JSON and CSV reports.

脚本会自动运行 baseline、消融组、随机多 seed 对照，以及 80/70/60 三档预算曲线，并输出 JSON/CSV 报告。

---

## 10. Output Files / 输出文件

Typical outputs:

```text
vortex_v5_6_reports.json
vortex_v5_6_reports.csv
```

The JSON file contains detailed per-run information, including generated text, retained stream, evicted stream, attribution counters, quality metrics, and memory telemetry.

JSON 文件包含每轮实验的详细结果，包括生成文本、保留流、删除流、归因统计、质量指标和显存遥测。

---

## 11. Known Limitations / 已知局限

1. **Logical eviction is not physical memory release.** `cache.offset -= 1` reduces effective KV length, but may not reduce peak allocated memory.
2. **Vortex is decode-time only.** Prompt prefill KV is not compressed in this version.
3. **Single-prompt results are not enough.** Current findings must be validated on a broader prompt suite.
4. **Budget controller can interact with model dynamics.** Fixed-budget results are diagnostic, not necessarily deployment-ready.
5. **Quality metrics are lightweight.** Future versions should add teacher-forced logprob gap and task-level evaluation.

已知局限：

1. **逻辑删除不等于物理释放显存。** `cache.offset -= 1` 主要缩短逻辑 KV 长度。
2. **当前是 Vortex-Decode，不压缩 prefill。** 长 prompt 的 prompt KV 仍然完整保留。
3. **单 prompt 不足以支撑论文结论。** 必须扩展多任务测试集。
4. **预算控制器可能影响模型轨迹。** 固定预算曲线是诊断工具，不一定是最终部署策略。
5. **质量指标仍偏轻量。** 后续应加入 teacher-forced logprob gap、任务级准确率和更长文本评估。

---

## 12. Roadmap / 后续路线

Recommended next steps:

1. Build a multi-prompt benchmark suite: narrative, logic reversal, temporal order, multi-entity tracking, code explanation, math word problems.
2. Increase `MAX_TOKENS` to 1024 / 2048 / 4096 to test physical memory behavior.
3. Add teacher-forced logprob gap against baseline continuation.
4. Tune Vortex@80 and reduce repetition in Vortex@70/@60.
5. Explore Vortex-Prefill for long prompt compression.
6. Replace heuristic thresholds with learned or percentile-calibrated thresholds.

建议后续工作：

1. 构建多 prompt 测试集：叙事、否定反转、时间顺序、多实体追踪、代码解释、数学文字题。
2. 将 `MAX_TOKENS` 提升到 1024 / 2048 / 4096，验证真实显存行为。
3. 加入 teacher-forced logprob gap。
4. 优化 Vortex@80，并降低 Vortex@70/@60 的重复率。
5. 探索 Vortex-Prefill，实现长 prompt KV 压缩。
6. 用分布分位数或学习到的门控替代手工阈值。

---

## 13. Conceptual Framing / 概念定位

Vortex can be described as:

> A lightweight inference-time conditional retention mechanism for decode KV cache, inspired by heavy-hitter behavior, lexical priors, and external memory ideas.

Vortex 可以这样定位：

> 一种推理期轻量级条件保留机制，用于动态裁剪 decode KV cache；它受 heavy hitter 行为、中文词法先验和外挂记忆思想启发。

It is **not** a trained memory module, and it is **not** equivalent to architectural external memory systems. It is an inference-time probing and retention policy.

它不是训练得到的记忆模块，也不等价于底层架构级外挂记忆系统。它是一个推理期探针与保留策略。

---

## 14. Disclaimer / 免责声明

This is an experimental research prototype. Results are prompt-dependent and model-dependent. Do not interpret current results as a proven general-purpose KV cache compression method.

这是一个实验性研究原型。当前结果依赖 prompt、模型和解码设置。不要将现阶段结果理解为已经被充分证明的通用 KV cache 压缩方法。

---

## 15. License / 许可证

Choose a license before publishing. Suggested options: MIT for open experimentation, or Apache-2.0 if you want explicit patent language.

发布前请自行选择许可证。若希望方便开放实验，可选 MIT；若希望包含更明确的专利条款，可选 Apache-2.0。
