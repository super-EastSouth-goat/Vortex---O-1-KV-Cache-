# -*- coding: utf-8 -*-
"""
Vortex V5.3: paper-grade ablation prototype

核心变化：
1. 长上下文安全：prefill 时不再 mx.eval(full logits)，只 materialize next_token。
2. 质量指标拆分：output recall 与 retained-stream recall 分开报告。
3. 归因增强：decision reason histogram + feature hit histogram。
4. logic attribution 修正：current token 与 rolling phrase 都参与 logic reason 统计。
5. 消融开关进入 STATE：logic / boundary / jump 分离，vortex_no_logic 不再关闭 boundary。
6. 新增 geo_no_jump 与 vortex_no_jump，验证 JUMP_THRESHOLD 的独立贡献。
7. Lexical-only 改名并拆分为 jieba_only 与 guard_only。
8. baseline similarity 增加 LCS ratio，补足顺序敏感性。
9. Jieba 词法先验升级为 2/3/4-gram rolling actual-token phrase。
10. 输出 JSON/CSV，方便复现实验与画图。

注意：
- offset -= 1 是逻辑 eviction，不保证立即释放 allocator 已分配的物理内存。
- FUSE_KEEP 是 post-hit local fuse，不是 H2O 式 strict recent window。
- important token 命中后，保护后续 FUSE_KEEP - 1 个 decode token。
"""

import csv
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import jieba
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


# ========================================================
# 0. 模型与全局配置
# ========================================================

MODEL_ID = "google/gemma-2-2b-it"

print("🚀 启动 M4 统一内存... 载入 Vortex V5.3 内存收割者 (Gemma-2-2B)")
model, tokenizer = load(MODEL_ID)

# 探针层：沿用前序实验的第 12 层。
TARGET_LAYER = 12

# 几何体积阈值：base_volume = (1 - cos(q_t, prompt_centroid)) * 10000
VOLUME_THRESHOLD = 3550.0

# 局部 Q 特征跳跃阈值。
# 当当前 token 与上一个幸存 token 的 Q 方向发生明显突变时，认为可能出现逻辑断层/语义反转。
JUMP_THRESHOLD = 0.22
JUMP_BONUS = 1200.0

# Jieba 词法先验与 guard bonus。
LEXICAL_BONUS = 1500.0
STRONG_LOGIC_BONUS = 900.0
WEAK_LOGIC_BONUS = 450.0
BOUNDARY_BONUS = 450.0

# post-hit local fuse。
# important token 命中后，保护后续 FUSE_KEEP - 1 个 decode token。
FUSE_KEEP = 2

# 开局保护：前 N 个生成的 token 绝对不杀，稳定解码轨迹！
STARTUP_KEEP = 8

# Jieba rolling phrase 最大 n-gram。4 足够覆盖很多中文实体短语，同时不会让组合爆炸。
LEXICAL_NGRAM_MAX = 4

# Random 对照组配置。
DEFAULT_RANDOM_KEEP_PROB = 0.60
# 论文模式建议 5 个 seed；烟测时可改成 (42,)。
RANDOM_SEEDS = (1, 2, 3, 4, 5)

# 输出长度。
MAX_TOKENS = 150

# 是否输出逐 token 调试信息。False 时只打印生成文本和最后战报。
DEBUG_TOKENS = False

# Random 多 seed 时，为避免刷屏，只打印第一个 seed 的 token 流。
PRINT_ONLY_FIRST_RANDOM_STREAM = True

# 报告文件。
REPORT_JSON_PATH = "vortex_v5_3_reports.json"
REPORT_CSV_PATH = "vortex_v5_3_reports.csv"

# 关键实体质量指标。
QUALITY_ENTITIES = ("李星辰", "舰队", "黑洞", "时间")

# 轻量逻辑质量指标。logic_recall 会以 Baseline 输出中实际出现的这些词为 target。
LOGIC_QUALITY_MARKERS = (
    "不", "没", "没有", "无", "未", "非", "并非", "不是", "不能", "不要", "无法",
    "但", "但是", "却", "然而", "不过", "否则",
    "因为", "所以", "因此", "导致", "如果", "若", "则", "除非",
    "之前", "之后", "以前", "以后", "直到", "正在", "已经", "曾经", "将要",
)

# 强逻辑词：hard keep。
STRONG_LOGIC_GUARD_STRINGS = {
    "不", "没", "没有", "无", "未", "非", "并非", "不是", "不能", "不要", "无法",
    "但", "但是", "却", "然而", "不过", "否则",
    "因为", "所以", "因此", "导致", "如果", "若", "则", "除非",
    "之前", "之后", "以前", "以后", "直到", "正在", "已经", "曾经", "将要",
}

# 弱逻辑/功能词：只加 bonus，不直接 hard keep。
# 这样避免“中文虚词白名单”吞噬压缩率。
WEAK_LOGIC_GUARD_STRINGS = {
    "因", "而", "前", "后", "在", "由", "被", "把", "从", "向", "至", "与", "或", "并", "则",
}

ALL_LOGIC_GUARD_STRINGS = STRONG_LOGIC_GUARD_STRINGS | WEAK_LOGIC_GUARD_STRINGS

# 结构边界保护：与 logic guard 分离，vortex_no_logic 不会关闭它。
# 结构边界保护：与 logic guard 分离
BOUNDARY_GUARD_STRINGS = {
    "。", "！", "？", "：", "；", "\n", "\n\n", "“", "”", "《", "》", "（", "）", "(", ")",
    "\"", "'", ":", ";", ",", ".", "?", "!", "[", "]"
}

# 初始化 Jieba。词法命中时使用 HMM=False，避免“新词发现”伪装成词典命中。
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
# 2. 文本工具
# ========================================================

def decode_token(token_id: Optional[int]) -> str:
    if token_id is None:
        return ""
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return ""


def decode_ids(token_ids: Sequence[int]) -> str:
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


def token_piece(token_id: Optional[int]) -> str:
    return normalize_piece(decode_token(token_id))


def join_pieces(pieces: Sequence[str]) -> str:
    # 中文场景下直接拼接更接近可读 retained stream；空片段跳过。
    return "".join(p for p in pieces if p)


def safe_percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * percentile / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def distribution_summary(prefix: str, values: Sequence[float]) -> Dict[str, Optional[float]]:
    return {
        f"{prefix}_p50": safe_percentile(values, 50),
        f"{prefix}_p75": safe_percentile(values, 75),
        f"{prefix}_p90": safe_percentile(values, 90),
        f"{prefix}_p95": safe_percentile(values, 95),
    }


# ========================================================
# 3. Vortex 运行态
# ========================================================

@dataclass
class VortexRuntime:
    active_strategy: str = "baseline"

    # 策略开关：放进 STATE，避免 global 开关污染后续实验。
    enable_lexical: bool = False
    enable_logic_guard: bool = False
    enable_boundary_guard: bool = False
    enable_jump: bool = False
    enable_fuse: bool = False

    # 当前 token 分数。
    current_score: float = 10.0
    base_volume: float = 10.0
    lexical_bonus: float = 0.0
    logic_bonus: float = 0.0
    weak_logic_bonus: float = 0.0
    boundary_bonus: float = 0.0
    jump_bonus: float = 0.0
    local_jump: float = 0.0

    # 探针状态。
    prompt_centroid: Optional[mx.array] = None
    current_q_tensor: Optional[mx.array] = None
    current_input_id: Optional[int] = None

    # 几何参考：上一个真正幸存进 KV 的 token。
    prev_kept_q: Optional[mx.array] = None
    prev_kept_token_id: Optional[int] = None

    # 词法参考：真实 decode token 流，不包含 prefill 尾 token。
    prev_actual_token_id: Optional[int] = None
    actual_token_window: List[int] = field(default_factory=list)

    # post-hit local fuse。
    fuse_remaining: int = 0

    # 当前 token 命中特征。
    lexical_hit: bool = False
    logic_hit: bool = False          # strong logic hard-hit
    weak_logic_hit: bool = False     # weak logic bonus-only hit
    boundary_hit: bool = False
    jump_hit: bool = False

    # 调试字段。
    actual_phrases: List[str] = field(default_factory=list)
    combined_word: str = ""
    lexical_phrase: str = ""
    current_text: str = ""
    keep_reason: str = ""

    # 统计。
    evicted_count: int = 0
    kept_count: int = 0
    processed_decode_tokens: int = 0
    output_tokens: int = 0

    # retained stream。
    kept_input_ids: List[int] = field(default_factory=list)
    evicted_input_ids: List[int] = field(default_factory=list)
    kept_pieces: List[str] = field(default_factory=list)
    evicted_pieces: List[str] = field(default_factory=list)

    # 归因统计。
    reason_counts: Counter = field(default_factory=Counter)
    logic_reason_counts: Counter = field(default_factory=Counter)
    feature_hit_counts: Counter = field(default_factory=Counter)

    # 阈值分布统计。
    base_volume_values: List[float] = field(default_factory=list)
    local_jump_values: List[float] = field(default_factory=list)
    score_values: List[float] = field(default_factory=list)

    def reset(self, strategy: str = "baseline") -> None:
        self.active_strategy = strategy

        self.enable_lexical = strategy in {
            "jieba_only", "guard_only", "vortex_no_logic", "vortex_no_jump", "vortex"
        }
        self.enable_logic_guard = strategy in {"guard_only", "vortex_no_jump", "vortex"}
        self.enable_boundary_guard = strategy in {"guard_only", "vortex_no_logic", "vortex_no_jump", "vortex"}
        self.enable_jump = strategy in {"geo", "vortex_no_logic", "vortex"}
        self.enable_fuse = strategy in {"vortex_no_logic", "vortex_no_jump", "vortex"}

        self.current_score = 10.0
        self.base_volume = 10.0
        self.lexical_bonus = 0.0
        self.logic_bonus = 0.0
        self.weak_logic_bonus = 0.0
        self.boundary_bonus = 0.0
        self.jump_bonus = 0.0
        self.local_jump = 0.0

        self.prompt_centroid = None
        self.current_q_tensor = None
        self.current_input_id = None

        self.prev_kept_q = None
        self.prev_kept_token_id = None
        self.prev_actual_token_id = None
        self.actual_token_window = []

        self.fuse_remaining = 0

        self.lexical_hit = False
        self.logic_hit = False
        self.weak_logic_hit = False
        self.boundary_hit = False
        self.jump_hit = False

        self.actual_phrases = []
        self.combined_word = ""
        self.lexical_phrase = ""
        self.current_text = ""
        self.keep_reason = ""

        self.evicted_count = 0
        self.kept_count = 0
        self.processed_decode_tokens = 0
        self.output_tokens = 0

        self.kept_input_ids = []
        self.evicted_input_ids = []
        self.kept_pieces = []
        self.evicted_pieces = []

        self.reason_counts = Counter()
        self.logic_reason_counts = Counter()
        self.feature_hit_counts = Counter()

        self.base_volume_values = []
        self.local_jump_values = []
        self.score_values = []


STATE = VortexRuntime()


# ========================================================
# 4. Jieba / guard 先验
# ========================================================

def build_actual_phrases(prev_window: Sequence[int], current_token_id: Optional[int]) -> List[str]:
    """用真实 decode token 的 rolling window 构造 2/3/4-gram phrase。"""
    if current_token_id is None:
        return []
    window = [int(x) for x in prev_window]
    max_n = min(LEXICAL_NGRAM_MAX, len(window) + 1)
    phrases: List[str] = []
    for n in range(2, max_n + 1):
        ids = window[-(n - 1):] + [int(current_token_id)]
        phrase = normalize_piece(decode_ids(ids))
        if phrase:
            phrases.append(phrase)
    return phrases


def is_jieba_single_word(phrase: str) -> bool:
    if len(phrase) <= 1:
        return False
    if len(phrase) > 16:
        return False
    if not contains_cjk(phrase):
        return False
    try:
        words = list(jieba.cut(phrase, cut_all=False, HMM=False))
    except TypeError:
        words = list(jieba.cut(phrase, cut_all=False))
    return len(words) == 1 and words[0] == phrase


def compute_lexical_prior(phrases: Sequence[str]) -> Tuple[float, bool, str]:
    """最长 phrase 优先。命中后返回 Jieba 词法 bonus。"""
    for phrase in sorted((p for p in phrases if p), key=len, reverse=True):
        if is_jieba_single_word(phrase):
            return LEXICAL_BONUS, True, phrase
    return 0.0, False, ""


def compute_guard_prior(current_text: str, phrases: Sequence[str]) -> Tuple[float, bool, bool, bool, str]:
    """
    返回：bonus, strong_logic_hit, weak_logic_hit, boundary_hit, matched_guard
    - strong logic：hard keep。
    - weak logic：bonus-only，不直接 hard keep。
    - boundary：hard keep，但与 logic guard 分开开关。
    """
    piece = normalize_piece(current_text)
    candidates = [piece] + [normalize_piece(p) for p in phrases if p]
    candidates = [x for x in candidates if x]

    matched_guard = ""
    strong_hit = False
    weak_hit = False
    boundary_hit = False

    if STATE.enable_logic_guard:
        for item in candidates:
            if item in STRONG_LOGIC_GUARD_STRINGS:
                strong_hit = True
                matched_guard = matched_guard or item
                break
        for item in candidates:
            if item in WEAK_LOGIC_GUARD_STRINGS:
                weak_hit = True
                matched_guard = matched_guard or item
                break

    if STATE.enable_boundary_guard:
        for item in candidates:
            if item in BOUNDARY_GUARD_STRINGS:
                boundary_hit = True
                matched_guard = matched_guard or item
                break

    bonus = 0.0
    if strong_hit:
        bonus += STRONG_LOGIC_BONUS
    if weak_hit:
        bonus += WEAK_LOGIC_BONUS
    if boundary_hit:
        bonus += BOUNDARY_BONUS
    return bonus, strong_hit, weak_hit, boundary_hit, matched_guard


def current_token_is_logic_marker() -> bool:
    piece = normalize_piece(STATE.current_text)
    phrases = [normalize_piece(p) for p in STATE.actual_phrases]
    return piece in ALL_LOGIC_GUARD_STRINGS or any(p in ALL_LOGIC_GUARD_STRINGS for p in phrases)


def clear_token_features() -> None:
    STATE.current_q_tensor = None
    STATE.current_text = decode_token(STATE.current_input_id)

    STATE.current_score = 10.0
    STATE.base_volume = 10.0
    STATE.lexical_bonus = 0.0
    STATE.logic_bonus = 0.0
    STATE.weak_logic_bonus = 0.0
    STATE.boundary_bonus = 0.0
    STATE.jump_bonus = 0.0
    STATE.local_jump = 0.0

    STATE.lexical_hit = False
    STATE.logic_hit = False
    STATE.weak_logic_hit = False
    STATE.boundary_hit = False
    STATE.jump_hit = False

    STATE.actual_phrases = build_actual_phrases(STATE.actual_token_window, STATE.current_input_id)
    STATE.combined_word = STATE.actual_phrases[-1] if STATE.actual_phrases else ""
    STATE.lexical_phrase = ""


def refresh_non_probe_token_state() -> None:
    """Baseline/Random 这类不挂探针策略使用。"""
    clear_token_features()


def refresh_guard_state() -> None:
    """Jieba-only / Guard-only：完全不挂 Q 探针，只基于真实 decode 邻接 token。"""
    clear_token_features()

    if STATE.enable_lexical:
        lexical_bonus, lexical_hit, phrase = compute_lexical_prior(STATE.actual_phrases)
        STATE.lexical_bonus = lexical_bonus
        STATE.lexical_hit = lexical_hit
        STATE.lexical_phrase = phrase

    guard_bonus, strong_hit, weak_hit, boundary_hit, matched_guard = compute_guard_prior(
        STATE.current_text,
        STATE.actual_phrases,
    )
    STATE.logic_bonus = STRONG_LOGIC_BONUS if strong_hit else 0.0
    STATE.weak_logic_bonus = WEAK_LOGIC_BONUS if weak_hit else 0.0
    STATE.boundary_bonus = BOUNDARY_BONUS if boundary_hit else 0.0
    STATE.logic_hit = strong_hit
    STATE.weak_logic_hit = weak_hit
    STATE.boundary_hit = boundary_hit
    if matched_guard and not STATE.lexical_phrase:
        STATE.lexical_phrase = matched_guard

    STATE.current_score = STATE.lexical_bonus + guard_bonus


# ========================================================
# 5. 质量指标
# ========================================================

def similarity_units(text: str) -> List[str]:
    """中文按字，英文/数字按连续串。"""
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


def lcs_ratio(candidate: str, baseline: str) -> float:
    a = similarity_units(candidate)
    b = similarity_units(baseline)
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return 100.0 * prev[-1] / max(len(a), len(b))


def recall_pct(targets: Sequence[str], haystack: str) -> Tuple[Optional[float], List[str]]:
    clean_targets = [x for x in targets if x]
    if not clean_targets:
        return None, []
    hits = [x for x in clean_targets if x in haystack]
    return 100.0 * len(hits) / len(clean_targets), hits


def compute_quality_metrics(candidate_text: str, baseline_text: str, retained_piece_text: str) -> Dict[str, object]:
    """output 与 retained stream 分开报告，避免混淆模型输出质量和 KV 生存流质量。"""
    output_entity_recall, output_entity_hits = recall_pct(QUALITY_ENTITIES, candidate_text)
    retained_entity_recall, retained_entity_hits = recall_pct(QUALITY_ENTITIES, retained_piece_text)

    logic_targets = [m for m in LOGIC_QUALITY_MARKERS if m in baseline_text]
    output_logic_recall, output_logic_hits = recall_pct(logic_targets, candidate_text)
    retained_logic_recall, retained_logic_hits = recall_pct(logic_targets, retained_piece_text)

    return {
        "output_entity_recall": output_entity_recall,
        "output_entity_hits": output_entity_hits,
        "retained_entity_recall": retained_entity_recall,
        "retained_entity_hits": retained_entity_hits,
        "output_logic_recall": output_logic_recall,
        "output_logic_hits": output_logic_hits,
        "retained_logic_recall": retained_logic_recall,
        "retained_logic_hits": retained_logic_hits,
        "logic_targets": logic_targets,
        "baseline_similarity": multiset_overlap_f1(candidate_text, baseline_text),
        "baseline_lcs": lcs_ratio(candidate_text, baseline_text),
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
                retained_piece_text=str(report.get("retained_piece_text", "")),
            )
        )


# ========================================================
# 6. Vortex 探针层代理
# ========================================================

class VortexLayerProxy:
    def __init__(self, original_layer):
        self.layer = original_layer

    def __getattr__(self, name):
        return getattr(self.layer, name)

    def __call__(self, x, mask=None, cache=None, **kwargs):
        # 只读探针：额外计算 Q 特征，不改变原 layer forward 输出。
        h = self.layer.input_layernorm(x)
        q_raw = self.layer.self_attn.q_proj(h)
        q_norm = q_raw / (mx.linalg.norm(q_raw, axis=-1, keepdims=True) + 1e-6)

        # Prefill：建立 prompt centroid，不做 token eviction 相关统计。
        if x.shape[1] > 1:
            centroid = mx.mean(q_norm, axis=1)
            STATE.prompt_centroid = centroid / (mx.linalg.norm(centroid, axis=-1, keepdims=True) + 1e-6)
            mx.eval(STATE.prompt_centroid)

            STATE.current_q_tensor = None
            STATE.current_score = 10.0
            STATE.base_volume = 10.0
            STATE.lexical_bonus = 0.0
            STATE.logic_bonus = 0.0
            STATE.weak_logic_bonus = 0.0
            STATE.boundary_bonus = 0.0
            STATE.jump_bonus = 0.0
            STATE.local_jump = 0.0
            STATE.lexical_hit = False
            STATE.logic_hit = False
            STATE.weak_logic_hit = False
            STATE.boundary_hit = False
            STATE.jump_hit = False
            STATE.actual_phrases = []
            STATE.combined_word = ""
            STATE.lexical_phrase = ""
            STATE.current_text = decode_token(STATE.current_input_id)

            return self.layer(x, mask=mask, cache=cache, **kwargs)

        # Decode：单 token 进入，计算当前输入 token 的 Vortex 特征。
        q_current = q_norm[:, 0, :]
        STATE.current_q_tensor = q_current
        current_token_id = STATE.current_input_id
        STATE.current_text = decode_token(current_token_id)
        STATE.actual_phrases = build_actual_phrases(STATE.actual_token_window, current_token_id)
        STATE.combined_word = STATE.actual_phrases[-1] if STATE.actual_phrases else ""

        if STATE.prompt_centroid is None or current_token_id is None:
            STATE.current_score = 10.0
            STATE.base_volume = 10.0
            return self.layer(x, mask=mask, cache=cache, **kwargs)

        q_32 = q_current.astype(mx.float32)
        centroid_32 = STATE.prompt_centroid.astype(mx.float32)

        cos_sim_global = mx.sum(q_32 * centroid_32, axis=-1)
        base_volume = (1.0 - cos_sim_global) * 10000.0
        STATE.base_volume = float(base_volume.item())

        # local jump 总是测量；是否启用为保留机制由 STATE.enable_jump 决定。
        STATE.local_jump = 0.0
        STATE.jump_bonus = 0.0
        STATE.jump_hit = False
        if STATE.prev_kept_q is not None:
            cos_sim_local = mx.sum(q_32 * STATE.prev_kept_q.astype(mx.float32), axis=-1)
            STATE.local_jump = float((1.0 - cos_sim_local).item())
            if STATE.enable_jump and STATE.local_jump > JUMP_THRESHOLD:
                STATE.jump_hit = True
                STATE.jump_bonus = JUMP_BONUS

        # Geo 消融：保持纯净，不计算 Jieba/guard。
        if STATE.active_strategy in {"geo_no_jump", "geo"}:
            STATE.lexical_bonus = 0.0
            STATE.lexical_hit = False
            STATE.logic_bonus = 0.0
            STATE.weak_logic_bonus = 0.0
            STATE.boundary_bonus = 0.0
            STATE.logic_hit = False
            STATE.weak_logic_hit = False
            STATE.boundary_hit = False
            STATE.lexical_phrase = ""
            STATE.current_score = STATE.base_volume + STATE.jump_bonus
            return self.layer(x, mask=mask, cache=cache, **kwargs)

        # Vortex 系列：几何 + 可选 Jieba + 可选 logic/boundary + 可选 jump。
        STATE.lexical_bonus = 0.0
        STATE.lexical_hit = False
        STATE.lexical_phrase = ""
        if STATE.enable_lexical:
            lexical_bonus, lexical_hit, phrase = compute_lexical_prior(STATE.actual_phrases)
            STATE.lexical_bonus = lexical_bonus
            STATE.lexical_hit = lexical_hit
            STATE.lexical_phrase = phrase

        guard_bonus, strong_hit, weak_hit, boundary_hit, matched_guard = compute_guard_prior(
            STATE.current_text,
            STATE.actual_phrases,
        )
        STATE.logic_bonus = STRONG_LOGIC_BONUS if strong_hit else 0.0
        STATE.weak_logic_bonus = WEAK_LOGIC_BONUS if weak_hit else 0.0
        STATE.boundary_bonus = BOUNDARY_BONUS if boundary_hit else 0.0
        STATE.logic_hit = strong_hit
        STATE.weak_logic_hit = weak_hit
        STATE.boundary_hit = boundary_hit
        if matched_guard and not STATE.lexical_phrase:
            STATE.lexical_phrase = matched_guard

        STATE.current_score = STATE.base_volume + STATE.jump_bonus + STATE.lexical_bonus + guard_bonus

        return self.layer(x, mask=mask, cache=cache, **kwargs)


ORIGINAL_TARGET_LAYER = model.model.layers[TARGET_LAYER]
VORTEX_PROXY = VortexLayerProxy(ORIGINAL_TARGET_LAYER)


def set_probe_enabled(enabled: bool) -> None:
    model.model.layers[TARGET_LAYER] = VORTEX_PROXY if enabled else ORIGINAL_TARGET_LAYER


# ========================================================
# 7. KV cache 回退与策略决策
# ========================================================

def rollback_current_token_from_cache(cache, prompt_len: int) -> int:
    """
    当前 token 已经写入 KV cache；若判定为低价值，就把每一层 offset 回退一格。
    这是逻辑 eviction，不保证立即释放 allocator 已申请的物理内存。
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


STRATEGY_NAMES = {
    "baseline": "Baseline 原生模式",
    "random_p": "Random-p 概率对齐对照",
    "random_k": "Random-k 数量精确对照",
    "geo_no_jump": "Geo-no-jump 体积-only 对照",
    "geo": "Geo 几何+Jump 对照",
    "jieba_only": "Jieba-only 词法对照",
    "guard_only": "Guard-only 词法/逻辑/边界对照",
    "vortex_no_logic": "Vortex-no-logic 消融",
    "vortex_no_jump": "Vortex-no-jump 消融",
    "vortex": "Vortex V5.3 完整模式",
}


def strategy_needs_probe(strategy: str) -> bool:
    return strategy in {"geo_no_jump", "geo", "vortex_no_logic", "vortex_no_jump", "vortex"}


def strategy_uses_guard_refresh(strategy: str) -> bool:
    return strategy in {"jieba_only", "guard_only"}


def strategy_uses_vortex_decision(strategy: str) -> bool:
    return strategy in {"vortex_no_logic", "vortex_no_jump", "vortex"}


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
    - important: 是否由核心规则命中，而不是仅由 post-hit fuse 保护。
    - reason: priority decision reason，不等价于因果归因；因果特征另看 feature_hit_counts。
    """
    if is_prefill:
        return True, False, "prefill"
    
    if STATE.processed_decode_tokens < STARTUP_KEEP:
        return True, True, "startup_keep"

    if strategy == "baseline":
        return True, False, "baseline"

    if strategy == "random_p":
        keep = rng.random() < random_keep_prob
        return keep, keep, "random_p_keep" if keep else "random_p_drop"

    if strategy == "random_k":
        decision_index = STATE.processed_decode_tokens
        if random_decision_budget is not None and decision_index >= random_decision_budget:
            return False, False, "random_k_after_budget"
        keep = random_keep_indices is not None and decision_index in random_keep_indices
        return keep, keep, "random_k_keep" if keep else "random_k_drop"

    if strategy == "geo_no_jump":
        important = STATE.base_volume >= VOLUME_THRESHOLD
        if important:
            return True, True, "geo_no_jump_volume"
        return False, False, "geo_no_jump_low"

    if strategy == "geo":
        important = STATE.base_volume >= VOLUME_THRESHOLD or STATE.jump_hit
        if STATE.jump_hit:
            return True, True, "geo_jump"
        if STATE.base_volume >= VOLUME_THRESHOLD:
            return True, True, "geo_volume"
        return False, False, "geo_low"

    if strategy == "jieba_only":
        if STATE.lexical_hit:
            return True, True, "jieba"
        return False, False, "jieba_low"

    if strategy == "guard_only":
        important = STATE.lexical_hit or STATE.logic_hit or STATE.boundary_hit or STATE.current_score >= VOLUME_THRESHOLD
        if STATE.lexical_hit:
            return True, True, "guard_jieba"
        if STATE.logic_hit:
            return True, True, "guard_logic"
        if STATE.boundary_hit:
            return True, True, "guard_boundary"
        if STATE.current_score >= VOLUME_THRESHOLD:
            return True, True, "guard_score"
        return False, False, "guard_low"

    if strategy_uses_vortex_decision(strategy):
        # 局部跳跃“破盾”：如果语义突然跳变，不继续沿用上一段的 post-hit fuse。
        # 当前 token 本身会因 jump_hit 被视为 important，再重新点燃 fuse。
        if STATE.jump_hit:
            STATE.fuse_remaining = 0

        fuse_keep = STATE.fuse_remaining > 0
        volume_hit = STATE.base_volume >= VOLUME_THRESHOLD
        score_hit = STATE.current_score >= VOLUME_THRESHOLD
        important = (
            volume_hit
            or score_hit
            or STATE.jump_hit
            or STATE.lexical_hit
            or STATE.logic_hit
            or STATE.boundary_hit
        )

        prefix = strategy
        if important:
            if STATE.jump_hit:
                return True, True, f"{prefix}_jump"
            if STATE.lexical_hit:
                return True, True, f"{prefix}_jieba"
            if STATE.logic_hit:
                return True, True, f"{prefix}_logic"
            if STATE.boundary_hit:
                return True, True, f"{prefix}_boundary"
            if volume_hit:
                return True, True, f"{prefix}_volume"
            return True, True, f"{prefix}_score"

        if fuse_keep:
            return True, False, f"{prefix}_post_hit_fuse"

        return False, False, f"{prefix}_low"

    raise ValueError(f"未知策略: {strategy}")


def record_feature_hits(is_prefill: bool, reason: str, strategy: str) -> None:
    if is_prefill:
        return

    STATE.reason_counts[reason] += 1

    if current_token_is_logic_marker():
        STATE.logic_reason_counts[reason] += 1
        STATE.feature_hit_counts["logic_marker_seen"] += 1

    if STATE.local_jump > JUMP_THRESHOLD:
        STATE.feature_hit_counts["raw_jump_over_threshold"] += 1
    if STATE.base_volume >= VOLUME_THRESHOLD:
        STATE.feature_hit_counts["volume_hit"] += 1
    if STATE.jump_hit:
        STATE.feature_hit_counts["jump_hit"] += 1
    if STATE.lexical_hit:
        STATE.feature_hit_counts["lexical_hit"] += 1
    if STATE.logic_hit:
        STATE.feature_hit_counts["strong_logic_hit"] += 1
    if STATE.weak_logic_hit:
        STATE.feature_hit_counts["weak_logic_hit"] += 1
    if STATE.boundary_hit:
        STATE.feature_hit_counts["boundary_hit"] += 1
    if STATE.current_score >= VOLUME_THRESHOLD:
        STATE.feature_hit_counts["score_hit"] += 1

    if strategy_needs_probe(strategy):
        STATE.base_volume_values.append(float(STATE.base_volume))
        STATE.local_jump_values.append(float(STATE.local_jump))
        STATE.score_values.append(float(STATE.current_score))


def update_after_decision(is_prefill: bool, keep: bool, important: bool, reason: str, strategy: str) -> None:
    """决策之后统一更新几何参考、词法参考、post-hit fuse 与统计。"""
    record_feature_hits(is_prefill=is_prefill, reason=reason, strategy=strategy)

    if not is_prefill:
        STATE.processed_decode_tokens += 1
        current_id = STATE.current_input_id
        piece = token_piece(current_id)

        if keep:
            STATE.kept_count += 1
            if current_id is not None:
                STATE.kept_input_ids.append(int(current_id))
                STATE.kept_pieces.append(piece)
            if STATE.current_q_tensor is not None:
                # 🚨 拔除幽灵图：强制求值，切断对历史网络状态的显存依赖
                mx.eval(STATE.current_q_tensor) 
                STATE.prev_kept_q = STATE.current_q_tensor
        else:
            STATE.evicted_count += 1
            if current_id is not None:
                STATE.evicted_input_ids.append(int(current_id))
                STATE.evicted_pieces.append(piece)

        if STATE.enable_fuse:
            if keep and not important and STATE.fuse_remaining > 0:
                STATE.fuse_remaining -= 1
            if important and keep:
                STATE.fuse_remaining = max(STATE.fuse_remaining, max(FUSE_KEEP - 1, 0))

        # 真实 decode token 参考只在非 prefill 阶段更新，避免 prompt 尾 token 污染首个生成 token。
        if current_id is not None:
            STATE.prev_actual_token_id = int(current_id)
            STATE.actual_token_window.append(int(current_id))
            # 为 4-gram 保留最近 3 个历史 token。
            STATE.actual_token_window = STATE.actual_token_window[-(LEXICAL_NGRAM_MAX - 1):]


# ========================================================
# 8. 生成主循环
# ========================================================

def generate_with_strategy(
    prompt_text: str,
    max_tokens: int = MAX_TOKENS,
    strategy: str = "vortex",
    random_keep_prob: float = DEFAULT_RANDOM_KEEP_PROB,
    random_keep_budget: Optional[int] = None,
    random_decision_budget: Optional[int] = None,
    seed: int = 42,
    debug_tokens: bool = DEBUG_TOKENS,
    print_stream: bool = True,
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

    print(f"\n[{STRATEGY_NAMES[strategy]} | seed={seed}] 正在生成...")
    if strategy == "random_p":
        print(f"🎲 Random-p 目标保留率: {random_keep_prob * 100:.1f}%")
    if strategy == "random_k":
        if resolved_decision_budget is None or resolved_keep_budget is None:
            raise ValueError("Random-k 需要 random_keep_budget 和 random_decision_budget")
        print(f"🎯 Random-k 精确保留预算: {resolved_keep_budget}/{resolved_decision_budget} 个 decode token")
    print("-" * 132)
    sys.stdout.flush()

    started_at = time.time()

    for step in range(max_tokens):
        is_prefill = step == 0 and x.shape[1] > 1

        # 长上下文安全：不要 materialize [batch, prompt_len, vocab] 的 full logits。
        # 只保留最后位置并只 eval next_token。
        last_logits = model(x, cache=cache)[:, -1, :]
        next_token = mx.argmax(last_logits, axis=-1)
        mx.eval(next_token)

        # 非探针策略不会进入 VortexLayerProxy，因此需要外层刷新当前 token 特征。
        if strategy_uses_guard_refresh(strategy):
            refresh_guard_state()
        elif not strategy_needs_probe(strategy):
            refresh_non_probe_token_state()

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

        update_after_decision(is_prefill, keep, important, reason, strategy)

        word = decode_token(next_token_id)
        if debug_tokens:
            marker = "✅" if keep else "✂️"
            print(
                f"{marker}[out={word!r} | in={STATE.current_text!r} | "
                f"score={STATE.current_score:.1f} base={STATE.base_volume:.1f} "
                f"jump={STATE.local_jump:.3f}/{STATE.jump_hit} "
                f"lex={STATE.lexical_hit} logic={STATE.logic_hit} weak={STATE.weak_logic_hit} "
                f"boundary={STATE.boundary_hit} fuse={STATE.fuse_remaining} "
                f"phrase={STATE.lexical_phrase!r} reason={reason}]",
                flush=True,
            )
        elif print_stream:
            print(word, end="", flush=True)

        # 下一轮将刚生成的 token 作为输入；届时才决定它是否留在 KV。
        x = mx.array([[next_token_id]])
        STATE.current_input_id = next_token_id

        if next_token_id == tokenizer.eos_token_id:
            break

    elapsed = max(time.time() - started_at, 1e-9)
    generated_text = decode_ids(generated_ids)
    retained_decode_text = decode_ids(STATE.kept_input_ids)
    evicted_decode_text = decode_ids(STATE.evicted_input_ids)
    retained_piece_text = join_pieces(STATE.kept_pieces)
    evicted_piece_text = join_pieces(STATE.evicted_pieces)
    mem = memory_snapshot_mb()

    if print_stream and not debug_tokens:
        print()
    print("-" * 132)

    processed = STATE.processed_decode_tokens
    kept = STATE.kept_count
    evicted = STATE.evicted_count
    retention_rate = (kept / processed * 100.0) if processed > 0 else 100.0
    tokens_per_sec = STATE.output_tokens / elapsed

    print(
        f"📊 逻辑战报: 输出 {STATE.output_tokens} 个 token | "
        f"decode 决策 {processed} 个 | 保留 {kept} 个 | 斩杀 {evicted} 个 | "
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

    report: Dict[str, object] = {
        "strategy": strategy,
        "strategy_name": STRATEGY_NAMES[strategy],
        "seed": seed,
        "generated_text": generated_text,
        "retained_decode_text": retained_decode_text,
        "evicted_decode_text": evicted_decode_text,
        "retained_piece_text": retained_piece_text,
        "evicted_piece_text": evicted_piece_text,
        "output_tokens": STATE.output_tokens,
        "processed_decode_tokens": processed,
        "kept_count": kept,
        "evicted_count": evicted,
        "retention_rate": retention_rate,
        "tokens_per_sec": tokens_per_sec,
        "random_keep_budget": resolved_keep_budget,
        "random_decision_budget": resolved_decision_budget,
        "reason_counts": dict(STATE.reason_counts),
        "logic_reason_counts": dict(STATE.logic_reason_counts),
        "feature_hit_counts": dict(STATE.feature_hit_counts),
        **distribution_summary("base_volume", STATE.base_volume_values),
        **distribution_summary("local_jump", STATE.local_jump_values),
        **distribution_summary("score", STATE.score_values),
        **mem,
    }
    return report


# ========================================================
# 9. 报告输出
# ========================================================

def fmt_pct(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "n/a"


def fmt_num(value: object, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None, None
    mean = sum(xs) / len(xs)
    if len(xs) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return mean, var ** 0.5


def aggregate_reports(reports: List[Dict[str, object]], strategy: str, label: str) -> Optional[Dict[str, object]]:
    if not reports:
        return None
    numeric_keys = [
        "output_tokens", "processed_decode_tokens", "kept_count", "evicted_count",
        "retention_rate", "tokens_per_sec", "active_mb", "peak_mb", "cache_mb",
        "output_entity_recall", "retained_entity_recall", "output_logic_recall", "retained_logic_recall",
        "baseline_similarity", "baseline_lcs",
        "base_volume_p50", "base_volume_p75", "base_volume_p90", "base_volume_p95",
        "local_jump_p50", "local_jump_p75", "local_jump_p90", "local_jump_p95",
        "score_p50", "score_p75", "score_p90", "score_p95",
    ]
    agg: Dict[str, object] = {
        "strategy": strategy,
        "strategy_name": label,
        "seed": "mean",
        "n": len(reports),
        "generated_text": "",
        "retained_piece_text": "",
        "reason_counts": dict(sum((Counter(r.get("reason_counts", {})) for r in reports), Counter())),
        "logic_reason_counts": dict(sum((Counter(r.get("logic_reason_counts", {})) for r in reports), Counter())),
        "feature_hit_counts": dict(sum((Counter(r.get("feature_hit_counts", {})) for r in reports), Counter())),
    }
    for key in numeric_keys:
        vals = [r.get(key) for r in reports if r.get(key) is not None]
        mean, std = mean_std([float(v) for v in vals]) if vals else (None, None)
        agg[key] = mean
        agg[f"{key}_std"] = std
    return agg


SUMMARY_ORDER = {
    "baseline": 0,
    "random_p_mean": 1,
    "random_k_mean": 2,
    "random_p": 1,
    "random_k": 2,
    "geo_no_jump": 3,
    "geo": 4,
    "jieba_only": 5,
    "guard_only": 6,
    "vortex_no_logic": 7,
    "vortex_no_jump": 8,
    "vortex": 9,
}

def ordered_reports(reports: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(reports, key=lambda r: (SUMMARY_ORDER.get(str(r.get("strategy")), 999), str(r.get("seed", ""))))

def print_summary(reports: List[Dict[str, object]]) -> None:
    reports = ordered_reports(reports)
    print("\n" + "=" * 164)
    print("📌 Ablation Summary")
    print("=" * 164)
    print(
        f"{'策略':<32} | {'seed':>5} | {'输出':>5} | {'决策':>5} | {'保留':>5} | {'斩杀':>5} | "
        f"{'保留率':>8} | {'Peak':>8} | {'tok/s':>8} | "
        f"{'OutEnt':>8} | {'RetEnt':>8} | {'OutLog':>8} | {'RetLog':>8} | {'BaseF1':>8} | {'LCS':>8}"
    )
    print("-" * 164)
    for r in reports:
        print(
            f"{str(r.get('strategy_name', ''))[:32]:<32} | "
            f"{str(r.get('seed', ''))[:5]:>5} | "
            f"{fmt_num(r.get('output_tokens'), 0):>5} | "
            f"{fmt_num(r.get('processed_decode_tokens'), 0):>5} | "
            f"{fmt_num(r.get('kept_count'), 0):>5} | "
            f"{fmt_num(r.get('evicted_count'), 0):>5} | "
            f"{fmt_pct(r.get('retention_rate')):>8} | "
            f"{fmt_num(r.get('peak_mb'), 1):>8} | "
            f"{fmt_num(r.get('tokens_per_sec'), 2):>8} | "
            f"{fmt_pct(r.get('output_entity_recall')):>8} | "
            f"{fmt_pct(r.get('retained_entity_recall')):>8} | "
            f"{fmt_pct(r.get('output_logic_recall')):>8} | "
            f"{fmt_pct(r.get('retained_logic_recall')):>8} | "
            f"{fmt_pct(r.get('baseline_similarity')):>8} | "
            f"{fmt_pct(r.get('baseline_lcs')):>8}"
        )
    print("=" * 164)

    baseline = next((r for r in reports if r.get("strategy") == "baseline"), None)
    if baseline is not None:
        logic_targets = baseline.get("logic_targets", [])
        print(f"🔎 Entity targets: {', '.join(QUALITY_ENTITIES)}")
        print(f"🔎 Logic targets from Baseline: {', '.join(logic_targets) if logic_targets else 'n/a'}")

    print("\n📎 Reason / feature attribution for Vortex-family rows:")
    for r in reports:
        if str(r.get("strategy", "")).startswith("vortex"):
            print(f"\n- {r.get('strategy_name')} seed={r.get('seed')}")
            print(f"  decision reasons: {r.get('reason_counts', {})}")
            print(f"  logic-token reasons: {r.get('logic_reason_counts', {})}")
            print(f"  feature hits: {r.get('feature_hit_counts', {})}")
            print(
                "  distributions: "
                f"base p50/p75/p90/p95={fmt_num(r.get('base_volume_p50'),1)}/"
                f"{fmt_num(r.get('base_volume_p75'),1)}/{fmt_num(r.get('base_volume_p90'),1)}/"
                f"{fmt_num(r.get('base_volume_p95'),1)}, "
                f"jump p50/p75/p90/p95={fmt_num(r.get('local_jump_p50'),3)}/"
                f"{fmt_num(r.get('local_jump_p75'),3)}/{fmt_num(r.get('local_jump_p90'),3)}/"
                f"{fmt_num(r.get('local_jump_p95'),3)}"
            )


def json_ready(obj):
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_reports(detailed_reports: List[Dict[str, object]], summary_reports: List[Dict[str, object]]) -> None:
    payload = {
        "config": {
            "model_id": MODEL_ID,
            "target_layer": TARGET_LAYER,
            "volume_threshold": VOLUME_THRESHOLD,
            "jump_threshold": JUMP_THRESHOLD,
            "fuse_keep": FUSE_KEEP,
            "lexical_ngram_max": LEXICAL_NGRAM_MAX,
            "random_seeds": list(RANDOM_SEEDS),
            "max_tokens": MAX_TOKENS,
            "startup_keep": STARTUP_KEEP
        },
        "detailed_reports": detailed_reports,
        "summary_reports": summary_reports,
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, ensure_ascii=False, indent=2)

    flat_keys = [
        "strategy", "strategy_name", "seed", "n", "output_tokens", "processed_decode_tokens",
        "kept_count", "evicted_count", "retention_rate", "active_mb", "peak_mb", "cache_mb",
        "tokens_per_sec", "output_entity_recall", "retained_entity_recall",
        "output_logic_recall", "retained_logic_recall", "baseline_similarity", "baseline_lcs",
        "base_volume_p50", "base_volume_p75", "base_volume_p90", "base_volume_p95",
        "local_jump_p50", "local_jump_p75", "local_jump_p90", "local_jump_p95",
        "score_p50", "score_p75", "score_p90", "score_p95",
    ]
    with open(REPORT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
        writer.writeheader()
        for r in summary_reports:
            writer.writerow({k: r.get(k) for k in flat_keys})

    print(f"\n💾 已保存 JSON: {REPORT_JSON_PATH}")
    print(f"💾 已保存 CSV:  {REPORT_CSV_PATH}")


# ========================================================
# 10. 消融实验入口
# ========================================================

def build_test_prompt() -> str:
    raw_prompt = """
公元3024年，人类联邦的星际舰队在银河系边缘遭遇了一个无法解释的物理现象：一个巨大的黑洞正在吞噬周围的时间。
舰队指挥官李星辰站在舰桥上，看着眼前的全息投影，深吸了一口气，说道：
""".strip()

    messages = [{"role": "user", "content": raw_prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_random_family(
    prompt: str,
    strategy: str,
    vortex_report: Dict[str, object],
    random_keep_prob: float,
) -> List[Dict[str, object]]:
    reports: List[Dict[str, object]] = []
    for idx, seed in enumerate(RANDOM_SEEDS):
        print_stream = not PRINT_ONLY_FIRST_RANDOM_STREAM or idx == 0
        if strategy == "random_p":
            report = generate_with_strategy(
                prompt,
                max_tokens=MAX_TOKENS,
                strategy="random_p",
                random_keep_prob=random_keep_prob,
                seed=seed,
                print_stream=print_stream,
            )
        elif strategy == "random_k":
            report = generate_with_strategy(
                prompt,
                max_tokens=MAX_TOKENS,
                strategy="random_k",
                random_keep_budget=int(vortex_report["kept_count"]),
                random_decision_budget=int(vortex_report["processed_decode_tokens"]),
                seed=seed,
                print_stream=print_stream,
            )
        else:
            raise ValueError(strategy)
        reports.append(report)
        print("\n" + "=" * 132)
    return reports


if __name__ == "__main__":
    test_prompt = build_test_prompt()
    print(f"✅ 提示词已组装: {test_prompt[:60]!r}...")

    deterministic_reports: List[Dict[str, object]] = []

    # A. Baseline：原生模式，不挂 proxy，不 eviction。
    deterministic_reports.append(generate_with_strategy(test_prompt, max_tokens=MAX_TOKENS, strategy="baseline", seed=0))
    print("\n" + "=" * 132)

    # D/E. 几何消融：volume-only vs volume+jump。
    deterministic_reports.append(generate_with_strategy(test_prompt, max_tokens=MAX_TOKENS, strategy="geo_no_jump", seed=0))
    print("\n" + "=" * 132)
    deterministic_reports.append(generate_with_strategy(test_prompt, max_tokens=MAX_TOKENS, strategy="geo", seed=0))
    print("\n" + "=" * 132)

    # F/G. 词法/guard 消融：Jieba-only vs Jieba+logic+boundary。
    deterministic_reports.append(generate_with_strategy(test_prompt, max_tokens=MAX_TOKENS, strategy="jieba_only", seed=0))
    print("\n" + "=" * 132)
    deterministic_reports.append(generate_with_strategy(test_prompt, max_tokens=MAX_TOKENS, strategy="guard_only", seed=0))
    print("\n" + "=" * 132)

    # H/I/J. Vortex 系列消融。
    deterministic_reports.append(generate_with_strategy(test_prompt, max_tokens=MAX_TOKENS, strategy="vortex_no_logic", seed=0))
    print("\n" + "=" * 132)
    deterministic_reports.append(generate_with_strategy(test_prompt, max_tokens=MAX_TOKENS, strategy="vortex_no_jump", seed=0))
    print("\n" + "=" * 132)
    vortex_report = generate_with_strategy(test_prompt, max_tokens=MAX_TOKENS, strategy="vortex", seed=0)
    deterministic_reports.append(vortex_report)
    print("\n" + "=" * 132)

    # B/C. Random 对照：对齐完整 Vortex 的保留率 / 保留数量，多 seed。
    random_keep_prob = max(0.0, min(1.0, float(vortex_report["retention_rate"]) / 100.0))
    random_p_reports = run_random_family(test_prompt, "random_p", vortex_report, random_keep_prob)
    random_k_reports = run_random_family(test_prompt, "random_k", vortex_report, random_keep_prob)

    detailed_reports = deterministic_reports + random_p_reports + random_k_reports
    attach_quality_metrics(detailed_reports)

    random_p_agg = aggregate_reports(random_p_reports, "random_p_mean", f"Random-p mean n={len(random_p_reports)}")
    random_k_agg = aggregate_reports(random_k_reports, "random_k_mean", f"Random-k mean n={len(random_k_reports)}")

    summary_reports: List[Dict[str, object]] = list(deterministic_reports)
    if random_p_agg is not None:
        summary_reports.append(random_p_agg)
    if random_k_agg is not None:
        summary_reports.append(random_k_agg)

    # 聚合 report 也需要质量指标；由于 aggregate_reports 在 attach_quality_metrics 后调用，字段已经是均值。
    print_summary(summary_reports)
    save_reports(detailed_reports, summary_reports)
