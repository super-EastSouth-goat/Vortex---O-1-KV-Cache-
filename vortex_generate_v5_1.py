import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import jieba
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


# ========================================================
# 0. 模型与全局配置
# ========================================================

MODEL_ID = "google/gemma-2-2b-it"

print("🚀 启动 M4 统一内存... 载入 Vortex V5.1 内存收割者 (Gemma-2-2B)")
model, tokenizer = load(MODEL_ID)

# 探针层：沿用已有实验的第 12 层。
TARGET_LAYER = 12

# 几何体积阈值：base_volume = (1 - cos(q_t, prompt_centroid)) * 10000
VOLUME_THRESHOLD = 3200.0

# 局部 Q 特征跳跃阈值。
# 当当前 token 与上一个幸存 token 的 Q 方向发生明显突变时，认为可能出现逻辑断层/语义反转。
JUMP_THRESHOLD = 0.15
JUMP_BONUS = 1200.0

# Jieba 词法先验与逻辑词保护。
LEXICAL_BONUS = 1500.0
LOGIC_BONUS = 900.0

# 微型局部保险丝。
# 重要 token 命中后，保护它后续紧邻的 RECENT_KEEP - 1 个 token。
RECENT_KEEP = 4

# Random 对照组配置。
DEFAULT_RANDOM_KEEP_PROB = 0.60
RANDOM_SEED = 42

# 输出长度。
MAX_TOKENS = 150

# 是否输出逐 token 调试信息。False 时只打印生成文本和最后战报。
DEBUG_TOKENS = False

# 关键实体质量指标。
QUALITY_ENTITIES = ("李星辰", "舰队", "黑洞", "时间")

# 轻量逻辑质量指标。logic_recall 会以 Baseline 输出中实际出现的这些词为 target。
LOGIC_QUALITY_MARKERS = (
    "不", "没", "没有", "无", "未", "非", "并非", "不是", "不能", "不要", "无法",
    "但", "但是", "却", "然而", "不过", "否则",
    "因为", "所以", "因此", "导致", "如果", "若", "则", "除非",
    "之前", "之后", "以前", "以后", "直到", "正在", "已经", "曾经", "将要",
)

# 逻辑保护词：这些词在中文里经常决定否定、转折、因果、条件、时间边界。
LOGIC_GUARD_STRINGS = {
    "不", "没", "没有", "无", "未", "非", "并非", "不是", "不能", "不要", "无法",
    "但", "但是", "却", "然而", "不过", "而", "否则",
    "因", "因为", "所以", "因此", "导致", "如果", "若", "则", "除非",
    "前", "后", "之前", "之后", "以前", "以后", "直到", "正在", "已经", "曾经", "将要",
    "在", "由", "被", "把", "从", "向", "至", "与", "或", "并",
}

# 结构边界保护。不要把所有逗号都当神圣 token，这里只保护更强的段落/引语/句子边界。
BOUNDARY_GUARD_STRINGS = {
    "。", "！", "？", "：", "；", "\n", "\n\n", "“", "”", "《", "》", "（", "）", "(", ")",
}

# 初始化 Jieba。词法命中时会使用 HMM=False，避免“新词发现”伪装成词典命中。
jieba.initialize()


# ========================================================
# 1. MLX 显存工具
# ========================================================

def mlx_call(name: str, default=None):
    """兼容不同 MLX 版本：函数可能挂在 mx 或 mx.metal 下。"""
    owners = [mx]
    if hasattr(mx, "metal"):
        owners.append(mx.metal)

    for owner in owners:
        fn = getattr(owner, name, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                continue
    return default


def bytes_to_mb(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(value) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def reset_mlx_memory_trackers() -> None:
    mlx_call("clear_cache")
    mlx_call("reset_peak_memory")


def memory_snapshot_mb() -> Dict[str, float]:
    return {
        "active_mb": bytes_to_mb(mlx_call("get_active_memory", 0)),
        "peak_mb": bytes_to_mb(mlx_call("get_peak_memory", 0)),
        "cache_mb": bytes_to_mb(mlx_call("get_cache_memory", 0)),
    }


# ========================================================
# 2. Vortex 运行态
# ========================================================

@dataclass
class VortexRuntime:
    # 当前策略，用于让同一个 proxy 在 Geo/Vortex 下启用不同计算。
    active_strategy: str = "baseline"

    # 当前 token 的综合分数。
    current_score: float = 10.0
    base_volume: float = 10.0
    lexical_bonus: float = 0.0
    logic_bonus: float = 0.0
    jump_bonus: float = 0.0
    local_jump: float = 0.0

    # 探针状态。
    prompt_centroid: Optional[mx.array] = None
    current_q_tensor: Optional[mx.array] = None
    current_input_id: Optional[int] = None

    # 几何参考：上一个真正幸存进 KV 的 token。
    prev_kept_q: Optional[mx.array] = None
    prev_kept_token_id: Optional[int] = None

    # 词法参考：上一个实际输入 token，不管它有没有幸存。
    # 这个变量专门给 Jieba 判断真实相邻 token，避免和几何参考互相污染。
    prev_actual_token_id: Optional[int] = None

    # V5/V5.1 微型局部保险丝。
    recent_fuse_remaining: int = 0

    # 当前 token 是否命中各类先验。
    lexical_hit: bool = False
    logic_hit: bool = False
    boundary_hit: bool = False
    jump_hit: bool = False

    # 调试字段。
    combined_word: str = ""
    current_text: str = ""
    keep_reason: str = ""

    # 统计。
    evicted_count: int = 0
    kept_count: int = 0
    processed_decode_tokens: int = 0
    output_tokens: int = 0

    # 记录 decode 阶段被保留/斩杀的输入 token，供质量指标做“保留流”检查。
    kept_input_ids: List[int] = field(default_factory=list)
    evicted_input_ids: List[int] = field(default_factory=list)

    def reset(self, strategy: str = "baseline") -> None:
        self.active_strategy = strategy

        self.current_score = 10.0
        self.base_volume = 10.0
        self.lexical_bonus = 0.0
        self.logic_bonus = 0.0
        self.jump_bonus = 0.0
        self.local_jump = 0.0

        self.prompt_centroid = None
        self.current_q_tensor = None
        self.current_input_id = None

        self.prev_kept_q = None
        self.prev_kept_token_id = None
        self.prev_actual_token_id = None

        self.recent_fuse_remaining = 0

        self.lexical_hit = False
        self.logic_hit = False
        self.boundary_hit = False
        self.jump_hit = False

        self.combined_word = ""
        self.current_text = ""
        self.keep_reason = ""

        self.evicted_count = 0
        self.kept_count = 0
        self.processed_decode_tokens = 0
        self.output_tokens = 0

        self.kept_input_ids = []
        self.evicted_input_ids = []


STATE = VortexRuntime()


# ========================================================
# 3. 文本/词法先验与质量指标工具
# ========================================================

def decode_token(token_id: Optional[int]) -> str:
    if token_id is None:
        return ""
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return ""


def decode_ids(token_ids: List[int]) -> str:
    if not token_ids:
        return ""
    try:
        return tokenizer.decode([int(t) for t in token_ids])
    except Exception:
        return ""


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def normalize_piece(text: str) -> str:
    return text.replace("▁", " ").strip()


def compute_lexical_prior(prev_token_id: Optional[int], current_token_id: Optional[int]) -> Tuple[float, bool, str]:
    """
    Jieba 词法先验：
    - 使用真实相邻 token，而不是上一个幸存 token。
    - HMM=False，尽量让命中更接近“默认词典/词频路径”，避免新词发现过度乐观。
    """
    if prev_token_id is None or current_token_id is None:
        return 0.0, False, ""

    combined = normalize_piece(tokenizer.decode([int(prev_token_id), int(current_token_id)]))
    if len(combined) <= 1:
        return 0.0, False, combined
    if len(combined) > 12:
        return 0.0, False, combined
    if not contains_cjk(combined):
        return 0.0, False, combined

    try:
        words = list(jieba.cut(combined, cut_all=False, HMM=False))
    except TypeError:
        # 老版本 jieba 如果不支持 HMM 参数，就退回默认精确模式。
        words = list(jieba.cut(combined, cut_all=False))

    hit = len(words) == 1 and words[0] == combined
    return (LEXICAL_BONUS if hit else 0.0), hit, combined


def compute_logic_prior(current_text: str, combined_word: str) -> Tuple[float, bool, bool]:
    """保护否定、转折、因果、条件、时间边界，以及强结构边界。"""
    piece = normalize_piece(current_text)
    combo = normalize_piece(combined_word)

    logic_hit = piece in LOGIC_GUARD_STRINGS or combo in LOGIC_GUARD_STRINGS
    boundary_hit = piece in BOUNDARY_GUARD_STRINGS or combo in BOUNDARY_GUARD_STRINGS

    bonus = 0.0
    if logic_hit:
        bonus += LOGIC_BONUS
    if boundary_hit:
        # 边界 token 重要，但不要让它压过实体词。
        bonus += LOGIC_BONUS * 0.5

    return bonus, logic_hit, boundary_hit


def refresh_non_probe_token_state() -> None:
    """Baseline/Random 这类不挂探针策略使用：只刷新当前 token 文本，清空各类分数。"""
    STATE.current_text = decode_token(STATE.current_input_id)
    STATE.current_q_tensor = None

    STATE.current_score = 10.0
    STATE.base_volume = 10.0
    STATE.lexical_bonus = 0.0
    STATE.logic_bonus = 0.0
    STATE.jump_bonus = 0.0
    STATE.local_jump = 0.0

    STATE.lexical_hit = False
    STATE.logic_hit = False
    STATE.boundary_hit = False
    STATE.jump_hit = False
    STATE.combined_word = ""


def refresh_lexical_only_state() -> None:
    """
    Lexical-only 消融：完全不挂 Q 探针，只基于真实相邻 token 做 Jieba/逻辑保护。
    """
    refresh_non_probe_token_state()

    lexical_bonus, lexical_hit, combined_word = compute_lexical_prior(
        STATE.prev_actual_token_id,
        STATE.current_input_id,
    )
    STATE.lexical_bonus = lexical_bonus
    STATE.lexical_hit = lexical_hit
    STATE.combined_word = combined_word

    logic_bonus, logic_hit, boundary_hit = compute_logic_prior(STATE.current_text, combined_word)
    STATE.logic_bonus = logic_bonus
    STATE.logic_hit = logic_hit
    STATE.boundary_hit = boundary_hit

    STATE.current_score = STATE.lexical_bonus + STATE.logic_bonus


def similarity_units(text: str) -> List[str]:
    """用于 baseline_similarity 的轻量字符/词切分。中文按字，英文/数字按连续串。"""
    units: List[str] = []
    buf: List[str] = []

    def flush_buf() -> None:
        if buf:
            units.append("".join(buf).lower())
            buf.clear()

    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            flush_buf()
            units.append(ch)
        elif ch.isalnum():
            buf.append(ch)
        else:
            flush_buf()
    flush_buf()
    return units


def multiset_overlap_f1(candidate: str, baseline: str) -> float:
    cand_units = similarity_units(candidate)
    base_units = similarity_units(baseline)
    if not cand_units and not base_units:
        return 100.0
    if not cand_units or not base_units:
        return 0.0

    overlap = sum((Counter(cand_units) & Counter(base_units)).values())
    return 200.0 * overlap / (len(cand_units) + len(base_units))


def recall_pct(targets: List[str], haystack: str) -> Tuple[Optional[float], List[str]]:
    if not targets:
        return None, []
    hits = [x for x in targets if x and x in haystack]
    return 100.0 * len(hits) / len(targets), hits


def compute_quality_metrics(candidate_text: str, baseline_text: str, retained_decode_text: str) -> Dict[str, object]:
    """
    三个轻量质量指标：
    - entity_recall：固定关键实体在候选输出或保留 decode 流中的召回。
    - logic_recall：以 Baseline 输出中实际出现的逻辑 marker 为 target，检查候选是否保留/复现。
    - baseline_similarity：候选输出与 Baseline 输出的字符/词 multiset overlap F1。
    """
    quality_haystack = candidate_text + "\n" + retained_decode_text

    entity_recall, entity_hits = recall_pct(list(QUALITY_ENTITIES), quality_haystack)

    logic_targets = [m for m in LOGIC_QUALITY_MARKERS if m in baseline_text]
    logic_recall, logic_hits = recall_pct(logic_targets, quality_haystack)

    return {
        "entity_recall": entity_recall,
        "entity_hits": entity_hits,
        "logic_recall": logic_recall,
        "logic_hits": logic_hits,
        "logic_targets": logic_targets,
        "baseline_similarity": multiset_overlap_f1(candidate_text, baseline_text),
    }


def attach_quality_metrics(reports: List[Dict[str, object]]) -> None:
    baseline_reports = [r for r in reports if r.get("strategy") == "baseline"]
    if not baseline_reports:
        return

    baseline_text = str(baseline_reports[0].get("generated_text", ""))
    for report in reports:
        report.update(
            compute_quality_metrics(
                candidate_text=str(report.get("generated_text", "")),
                baseline_text=baseline_text,
                retained_decode_text=str(report.get("retained_decode_text", "")),
            )
        )


# ========================================================
# 4. Vortex 探针层代理
# ========================================================

class VortexLayerProxy:
    def __init__(self, original_layer):
        self.layer = original_layer

    def __getattr__(self, name):
        # 让外部代码访问 proxy 上不存在的属性时，自动转发给原始层。
        return getattr(self.layer, name)

    def __call__(self, x, mask=None, cache=None, **kwargs):
        # 先在第 TARGET_LAYER 层取 Q 向量做探针，不改变原始 forward 的输出。
        h = self.layer.input_layernorm(x)
        q_raw = self.layer.self_attn.q_proj(h)
        q_norm = q_raw / (mx.linalg.norm(q_raw, axis=-1, keepdims=True) + 1e-6)

        # Prefill：整段 prompt 进入，建立 prompt centroid。
        if x.shape[1] > 1:
            centroid = mx.mean(q_norm, axis=1)
            STATE.prompt_centroid = centroid / (mx.linalg.norm(centroid, axis=-1, keepdims=True) + 1e-6)
            mx.eval(STATE.prompt_centroid)

            STATE.current_q_tensor = None
            STATE.current_score = 10.0
            STATE.base_volume = 10.0
            STATE.lexical_bonus = 0.0
            STATE.logic_bonus = 0.0
            STATE.jump_bonus = 0.0
            STATE.local_jump = 0.0
            STATE.lexical_hit = False
            STATE.logic_hit = False
            STATE.boundary_hit = False
            STATE.jump_hit = False
            STATE.combined_word = ""
            STATE.current_text = decode_token(STATE.current_input_id)

            return self.layer(x, mask=mask, cache=cache, **kwargs)

        # Decode：单 token 进入，计算当前 token 的几何分数。
        q_current = q_norm[:, 0, :]
        STATE.current_q_tensor = q_current
        current_token_id = STATE.current_input_id
        STATE.current_text = decode_token(current_token_id)

        if STATE.prompt_centroid is None or current_token_id is None:
            STATE.current_score = 10.0
            STATE.base_volume = 10.0
            return self.layer(x, mask=mask, cache=cache, **kwargs)

        q_32 = q_current.astype(mx.float32)
        centroid_32 = STATE.prompt_centroid.astype(mx.float32)

        cos_sim_global = mx.sum(q_32 * centroid_32, axis=-1)
        base_volume = (1.0 - cos_sim_global) * 10000.0
        STATE.base_volume = float(base_volume.item())

        # 局部跳跃：当前 token 与上一个幸存 token 的 Q 方向差。
        STATE.local_jump = 0.0
        STATE.jump_bonus = 0.0
        STATE.jump_hit = False
        if STATE.prev_kept_q is not None:
            cos_sim_local = mx.sum(q_32 * STATE.prev_kept_q.astype(mx.float32), axis=-1)
            STATE.local_jump = float((1.0 - cos_sim_local).item())
            STATE.jump_hit = STATE.local_jump > JUMP_THRESHOLD
            if STATE.jump_hit:
                STATE.jump_bonus = JUMP_BONUS

        # Geo-only 消融保持纯净：不计算 Jieba/逻辑先验。
        if STATE.active_strategy == "geo":
            STATE.lexical_bonus = 0.0
            STATE.lexical_hit = False
            STATE.combined_word = ""
            STATE.logic_bonus = 0.0
            STATE.logic_hit = False
            STATE.boundary_hit = False
            STATE.current_score = STATE.base_volume + STATE.jump_bonus
            return self.layer(x, mask=mask, cache=cache, **kwargs)

        # Vortex 模式：几何 + Jieba + 逻辑保护。
        lexical_bonus, lexical_hit, combined_word = compute_lexical_prior(
            STATE.prev_actual_token_id,
            current_token_id,
        )
        STATE.lexical_bonus = lexical_bonus
        STATE.lexical_hit = lexical_hit
        STATE.combined_word = combined_word

        logic_bonus, logic_hit, boundary_hit = compute_logic_prior(STATE.current_text, combined_word)
        STATE.logic_bonus = logic_bonus
        STATE.logic_hit = logic_hit
        STATE.boundary_hit = boundary_hit

        STATE.current_score = STATE.base_volume + STATE.jump_bonus + STATE.lexical_bonus + STATE.logic_bonus

        return self.layer(x, mask=mask, cache=cache, **kwargs)


ORIGINAL_TARGET_LAYER = model.model.layers[TARGET_LAYER]
VORTEX_PROXY = VortexLayerProxy(ORIGINAL_TARGET_LAYER)


def set_probe_enabled(enabled: bool) -> None:
    model.model.layers[TARGET_LAYER] = VORTEX_PROXY if enabled else ORIGINAL_TARGET_LAYER


# ========================================================
# 5. KV cache 回退与策略决策
# ========================================================

def rollback_current_token_from_cache(cache, prompt_len: int) -> int:
    """
    当前 token 已经写入 KV cache；若判定为低价值，就把每一层 offset 回退一格。
    这是一种逻辑 eviction，不保证立即释放 allocator 已经申请的物理内存。
    """
    if cache is None:
        return 0

    changed = 0
    for layer_cache in cache:
        if layer_cache is None or not hasattr(layer_cache, "offset"):
            continue
        try:
            old_offset = layer_cache.offset
            if isinstance(old_offset, int):
                layer_cache.offset = max(prompt_len, old_offset - 1)
            else:
                layer_cache.offset -= 1
            changed += 1
        except Exception:
            pass
    return changed


def decide_keep(
    strategy: str,
    is_prefill: bool,
    rng: random.Random,
    random_keep_prob: float,
    random_keep_indices: Optional[Set[int]] = None,
    random_decision_budget: Optional[int] = None,
) -> Tuple[bool, bool, str]:
    """
    返回：
    - keep: 当前输入 token 是否保留在 KV cache。
    - important: 是否由核心规则命中，而不是仅由 recent fuse 保护。
    - reason: 调试原因。
    """
    if is_prefill:
        return True, False, "prefill"

    if strategy == "baseline":
        return True, False, "baseline"

    if strategy == "random_p":
        keep = rng.random() < random_keep_prob
        return keep, keep, "random_p_keep" if keep else "random_p_drop"

    if strategy == "random_k":
        # 当前 decode 决策序号：第一个非 prefill token 为 0。
        decision_index = STATE.processed_decode_tokens
        if random_decision_budget is not None and decision_index >= random_decision_budget:
            return False, False, "random_k_after_budget"
        keep = random_keep_indices is not None and decision_index in random_keep_indices
        return keep, keep, "random_k_keep" if keep else "random_k_drop"

    if strategy == "geo":
        important = STATE.base_volume >= VOLUME_THRESHOLD or STATE.jump_hit
        if STATE.jump_hit:
            return True, True, "geo_jump"
        if STATE.base_volume >= VOLUME_THRESHOLD:
            return True, True, "geo_volume"
        return False, False, "geo_low"

    if strategy == "lexical":
        important = STATE.lexical_hit or STATE.logic_hit or STATE.boundary_hit
        if STATE.lexical_hit:
            return True, True, "jieba"
        if STATE.logic_hit:
            return True, True, "logic"
        if STATE.boundary_hit:
            return True, True, "boundary"
        return False, False, "lexical_low"

    if strategy == "vortex":
        # 局部跳跃“破盾”：如果语义突然跳变，不继续沿用上一段的 recent fuse。
        # 但当前 token 本身会因为 jump_hit 被视为重要 token，从而重新点燃保险丝。
        if STATE.jump_hit:
            STATE.recent_fuse_remaining = 0

        recent_keep = STATE.recent_fuse_remaining > 0
        important = (
            STATE.current_score >= VOLUME_THRESHOLD
            or STATE.jump_hit
            or STATE.lexical_hit
            or STATE.logic_hit
            or STATE.boundary_hit
        )

        if important:
            if STATE.jump_hit:
                reason = "vortex_jump"
            elif STATE.lexical_hit:
                reason = "vortex_jieba"
            elif STATE.logic_hit:
                reason = "vortex_logic"
            elif STATE.boundary_hit:
                reason = "vortex_boundary"
            else:
                reason = "vortex_volume"
            return True, True, reason

        if recent_keep:
            return True, False, "recent_fuse"

        return False, False, "vortex_low"

    raise ValueError(f"未知策略: {strategy}")


def update_after_decision(is_prefill: bool, keep: bool, important: bool, strategy: str) -> None:
    """决策之后统一更新几何参考、词法参考、recent fuse 与统计。"""
    if not is_prefill:
        STATE.processed_decode_tokens += 1

        if keep:
            STATE.kept_count += 1
            if STATE.current_input_id is not None:
                STATE.kept_input_ids.append(int(STATE.current_input_id))
            if STATE.current_q_tensor is not None:
                STATE.prev_kept_q = STATE.current_q_tensor
                STATE.prev_kept_token_id = STATE.current_input_id
        else:
            STATE.evicted_count += 1
            if STATE.current_input_id is not None:
                STATE.evicted_input_ids.append(int(STATE.current_input_id))

        if strategy == "vortex":
            # 如果只是被保险丝保护，消耗一格保险丝。
            if keep and not important and STATE.recent_fuse_remaining > 0:
                STATE.recent_fuse_remaining -= 1

            # 只有核心命中才重新点燃局部保险丝。
            if important and keep:
                STATE.recent_fuse_remaining = max(STATE.recent_fuse_remaining, max(RECENT_KEEP - 1, 0))

    # 真实相邻 token 参考永远更新，不管当前 token 有没有幸存。
    # 这是 Jieba 词法系统和几何幸存系统解耦的关键。
    if STATE.current_input_id is not None:
        STATE.prev_actual_token_id = int(STATE.current_input_id)


# ========================================================
# 6. 生成主循环
# ========================================================

STRATEGY_NAMES = {
    "baseline": "Baseline 原生模式",
    "random_p": "Random-p 概率对齐对照",
    "random_k": "Random-k 数量精确对照",
    "geo": "Geo-only 几何探针对照",
    "lexical": "Lexical-only 词法先验对照",
    "vortex": "Vortex V5.1 大一统模式",
}


def strategy_needs_probe(strategy: str) -> bool:
    # V5.1 改动二：Lexical-only 不挂 Q 探针，保证消融干净。
    return strategy in {"geo", "vortex"}


def build_random_k_indices(
    rng: random.Random,
    target_decisions: Optional[int],
    target_keep_count: Optional[int],
) -> Tuple[Optional[Set[int]], Optional[int], Optional[int]]:
    if target_decisions is None or target_keep_count is None:
        return None, None, None

    decision_budget = max(0, int(target_decisions))
    keep_budget = max(0, min(int(target_keep_count), decision_budget))

    if decision_budget == 0:
        return set(), 0, 0

    keep_indices = set(rng.sample(range(decision_budget), keep_budget))
    return keep_indices, decision_budget, keep_budget


def generate_with_strategy(
    prompt_text: str,
    max_tokens: int = MAX_TOKENS,
    strategy: str = "vortex",
    random_keep_prob: float = DEFAULT_RANDOM_KEEP_PROB,
    random_keep_budget: Optional[int] = None,
    random_decision_budget: Optional[int] = None,
    seed: int = RANDOM_SEED,
    debug_tokens: bool = DEBUG_TOKENS,
) -> Dict[str, object]:
    if strategy not in STRATEGY_NAMES:
        raise ValueError(f"strategy 必须是 {list(STRATEGY_NAMES)} 之一")

    STATE.reset(strategy=strategy)
    set_probe_enabled(strategy_needs_probe(strategy))
    reset_mlx_memory_trackers()

    rng = random.Random(seed)
    random_keep_indices, resolved_decision_budget, resolved_keep_budget = build_random_k_indices(
        rng=rng,
        target_decisions=random_decision_budget,
        target_keep_count=random_keep_budget,
    )

    prompt_ids = tokenizer.encode(prompt_text)
    if not prompt_ids:
        raise ValueError("prompt 为空，无法生成。")

    prompt_len = len(prompt_ids)
    x = mx.array([prompt_ids])
    STATE.current_input_id = int(prompt_ids[-1])

    cache = make_prompt_cache(model)
    generated_ids: List[int] = []

    print(f"\n[{STRATEGY_NAMES[strategy]}] 正在生成...")
    if strategy == "random_p":
        print(f"🎲 Random-p 目标保留率: {random_keep_prob * 100:.1f}%")
    if strategy == "random_k":
        if resolved_decision_budget is None or resolved_keep_budget is None:
            raise ValueError("Random-k 需要 random_keep_budget 和 random_decision_budget")
        print(f"🎯 Random-k 精确保留预算: {resolved_keep_budget}/{resolved_decision_budget} 个 decode token")
    print("-" * 88)
    sys.stdout.flush()

    started_at = time.time()

    for step in range(max_tokens):
        is_prefill = step == 0 and x.shape[1] > 1

        logits = model(x, cache=cache)
        mx.eval(logits)

        # 非探针策略不会进入 VortexLayerProxy，因此需要在外层刷新当前 token 的状态。
        if strategy == "lexical":
            refresh_lexical_only_state()
        elif not strategy_needs_probe(strategy):
            refresh_non_probe_token_state()

        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        next_token_id = int(next_token.item())
        generated_ids.append(next_token_id)
        STATE.output_tokens += 1

        keep, important, reason = decide_keep(
            strategy=strategy,
            is_prefill=is_prefill,
            rng=rng,
            random_keep_prob=random_keep_prob,
            random_keep_indices=random_keep_indices,
            random_decision_budget=resolved_decision_budget,
        )
        STATE.keep_reason = reason

        if not keep and not is_prefill:
            rollback_current_token_from_cache(cache, prompt_len=prompt_len)

        update_after_decision(is_prefill, keep, important, strategy)

        word = decode_token(next_token_id)
        if debug_tokens:
            marker = "✅" if keep else "✂️"
            print(
                f"{marker}[out={word!r} | in={STATE.current_text!r} | "
                f"score={STATE.current_score:.1f} base={STATE.base_volume:.1f} "
                f"jump={STATE.local_jump:.3f} lex={STATE.lexical_hit} "
                f"logic={STATE.logic_hit} fuse={STATE.recent_fuse_remaining} "
                f"reason={reason}]",
                flush=True,
            )
        else:
            print(word, end="", flush=True)

        # 下一轮将刚刚生成的 token 作为输入；届时才会决定它是否进入 KV。
        x = mx.array([[next_token_id]])
        STATE.current_input_id = next_token_id

        if next_token_id == tokenizer.eos_token_id:
            break

    elapsed = max(time.time() - started_at, 1e-9)
    generated_text = decode_ids(generated_ids)
    retained_decode_text = decode_ids(STATE.kept_input_ids)
    evicted_decode_text = decode_ids(STATE.evicted_input_ids)
    mem = memory_snapshot_mb()

    if not debug_tokens:
        print()
    print("-" * 88)

    processed = STATE.processed_decode_tokens
    kept = STATE.kept_count
    evicted = STATE.evicted_count
    retention_rate = (kept / processed * 100.0) if processed > 0 else 100.0
    tokens_per_sec = STATE.output_tokens / elapsed

    print(
        f"📊 逻辑战报: 输出 {STATE.output_tokens} 个 token | "
        f"已进入 decode 决策 {processed} 个 | 保留 {kept} 个 | 斩杀 {evicted} 个 | "
        f"生成期逻辑保留率 {retention_rate:.1f}%"
    )
    print(
        f"💾 MLX 显存: active={mem['active_mb']:.2f} MB | "
        f"peak={mem['peak_mb']:.2f} MB | cache={mem['cache_mb']:.2f} MB"
    )
    print(f"⚡ 速度: {tokens_per_sec:.2f} tok/s")

    if strategy == "random_k" and resolved_keep_budget is not None:
        exact_flag = "OK" if kept == resolved_keep_budget else "EARLY_STOP/DRIFT"
        print(f"🎯 Random-k 对齐检查: actual_kept={kept}, target_kept={resolved_keep_budget}, status={exact_flag}")

    return {
        "strategy": strategy,
        "strategy_name": STRATEGY_NAMES[strategy],
        "generated_text": generated_text,
        "retained_decode_text": retained_decode_text,
        "evicted_decode_text": evicted_decode_text,
        "output_tokens": STATE.output_tokens,
        "processed_decode_tokens": processed,
        "kept_count": kept,
        "evicted_count": evicted,
        "retention_rate": retention_rate,
        "tokens_per_sec": tokens_per_sec,
        "random_keep_budget": resolved_keep_budget,
        "random_decision_budget": resolved_decision_budget,
        **mem,
    }


# ========================================================
# 7. 消融实验主入口
# ========================================================

def fmt_pct(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "n/a"


def print_summary(reports: List[Dict[str, object]]) -> None:
    print("\n" + "=" * 112)
    print("📌 Ablation Summary")
    print("=" * 112)
    print(
        f"{'策略':<26} | {'输出':>5} | {'决策':>5} | {'保留':>5} | {'斩杀':>5} | "
        f"{'保留率':>8} | {'Peak MB':>10} | {'tok/s':>8} | "
        f"{'Entity':>8} | {'Logic':>8} | {'BaseSim':>8}"
    )
    print("-" * 112)
    for r in reports:
        print(
            f"{str(r['strategy_name'])[:26]:<26} | "
            f"{int(r['output_tokens']):>5} | "
            f"{int(r['processed_decode_tokens']):>5} | "
            f"{int(r['kept_count']):>5} | "
            f"{int(r['evicted_count']):>5} | "
            f"{fmt_pct(r.get('retention_rate')):>8} | "
            f"{float(r['peak_mb']):>10.2f} | "
            f"{float(r['tokens_per_sec']):>8.2f} | "
            f"{fmt_pct(r.get('entity_recall')):>8} | "
            f"{fmt_pct(r.get('logic_recall')):>8} | "
            f"{fmt_pct(r.get('baseline_similarity')):>8}"
        )
    print("=" * 112)

    baseline = next((r for r in reports if r.get("strategy") == "baseline"), None)
    if baseline is not None:
        logic_targets = baseline.get("logic_targets", [])
        print(f"🔎 Entity targets: {', '.join(QUALITY_ENTITIES)}")
        print(f"🔎 Logic targets from Baseline: {', '.join(logic_targets) if logic_targets else 'n/a'}")


def build_test_prompt() -> str:
    raw_prompt = """
公元3024年，人类联邦的星际舰队在银河系边缘遭遇了一个无法解释的物理现象：一个巨大的黑洞正在吞噬周围的时间。
舰队指挥官李星辰站在舰桥上，看着眼前的全息投影，深吸了一口气，说道：
""".strip()

    messages = [{"role": "user", "content": raw_prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


if __name__ == "__main__":
    test_prompt = build_test_prompt()
    print(f"✅ 提示词已组装: {test_prompt[:60]!r}...")

    reports: List[Dict[str, object]] = []

    # A. 原生 baseline：不挂载 VortexLayerProxy，不做 eviction。
    reports.append(generate_with_strategy(test_prompt, max_tokens=MAX_TOKENS, strategy="baseline"))

    print("\n" + "=" * 88)

    # B. Geo-only：只看 Q 几何体积与局部跳跃；不计算 Jieba/逻辑先验。
    reports.append(generate_with_strategy(test_prompt, max_tokens=MAX_TOKENS, strategy="geo"))

    print("\n" + "=" * 88)

    # C. Lexical-only：不挂 Q 探针，只看 Jieba 词法先验与逻辑词保护。
    reports.append(generate_with_strategy(test_prompt, max_tokens=MAX_TOKENS, strategy="lexical"))

    print("\n" + "=" * 88)

    # D. Vortex V5.1：几何探针 + Jieba + 逻辑保护 + 微型 recent fuse。
    vortex_report = generate_with_strategy(test_prompt, max_tokens=MAX_TOKENS, strategy="vortex")
    reports.append(vortex_report)

    print("\n" + "=" * 88)

    # E. Random-p：概率保留率对齐 Vortex。
    random_keep_prob = max(0.0, min(1.0, float(vortex_report["retention_rate"]) / 100.0))
    reports.append(
        generate_with_strategy(
            test_prompt,
            max_tokens=MAX_TOKENS,
            strategy="random_p",
            random_keep_prob=random_keep_prob,
            seed=RANDOM_SEED,
        )
    )

    print("\n" + "=" * 88)

    # F. Random-k：精确保留数量对齐 Vortex。
    reports.append(
        generate_with_strategy(
            test_prompt,
            max_tokens=MAX_TOKENS,
            strategy="random_k",
            random_keep_budget=int(vortex_report["kept_count"]),
            random_decision_budget=int(vortex_report["processed_decode_tokens"]),
            seed=RANDOM_SEED,
        )
    )

    attach_quality_metrics(reports)
    print_summary(reports)
