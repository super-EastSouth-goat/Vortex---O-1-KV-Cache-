import mlx.core as mx
from mlx_lm import load
import matplotlib.pyplot as plt
import numpy as np

print("🛰️ 启动 M4 统一内存... 载入流形扫描雷达 (Gemma-2-2B)")
model, tokenizer = load("google/gemma-2-2b-it")

# 全局变量，用于截获扫描到的体积数据
token_volumes = []
token_labels = []

# ========================================================
# 1. 雷达探针：潜入第 12 层截获几何体积
# ========================================================
TARGET_LAYER = 12
target_layer_instance = model.model.layers[TARGET_LAYER]
original_call = type(target_layer_instance).__call__

def radar_forward_hook(self, x, mask=None, cache=None, **kwargs):
    global token_volumes
    
    if self is target_layer_instance:
        # 模拟三大物理场
        hidden_in = x
        hidden_attn = mx.maximum(x, 0)
        hidden_mlp = mx.sin(x)
        
        # L2 归一化防作弊
        h_in_norm = hidden_in / (mx.linalg.norm(hidden_in, axis=-1, keepdims=True) + 1e-6)
        h_attn_norm = hidden_attn / (mx.linalg.norm(hidden_attn, axis=-1, keepdims=True) + 1e-6)
        h_mlp_norm = hidden_mlp / (mx.linalg.norm(hidden_mlp, axis=-1, keepdims=True) + 1e-6)
        
        # 逐个 Token 计算 3x3 矩阵行列式！(序列维度遍历)
        # 我们要看清每一个汉字的思维体积！
        seq_len = x.shape[1]
        
        for i in range(seq_len):
            q = h_in_norm[:, i, :]
            k = h_in_norm[:, max(0, i-1), :]
            v = h_in_norm[:, max(0, i-2), :]
            
            row1 = mx.stack([mx.sum(h_in_norm[:, i, :] * q, axis=-1), mx.sum(h_in_norm[:, i, :] * k, axis=-1), mx.sum(h_in_norm[:, i, :] * v, axis=-1)], axis=-1)
            row2 = mx.stack([mx.sum(h_attn_norm[:, i, :] * q, axis=-1), mx.sum(h_attn_norm[:, i, :] * k, axis=-1), mx.sum(h_attn_norm[:, i, :] * v, axis=-1)], axis=-1)
            row3 = mx.stack([mx.sum(h_mlp_norm[:, i, :] * q, axis=-1), mx.sum(h_mlp_norm[:, i, :] * k, axis=-1), mx.sum(h_mlp_norm[:, i, :] * v, axis=-1)], axis=-1)
            
            M = mx.stack([row1, row2, row3], axis=-1)
            a, b, c = M[:, 0, 0], M[:, 0, 1], M[:, 0, 2]
            d, e, f = M[:, 1, 0], M[:, 1, 1], M[:, 1, 2]
            g, h, i_val = M[:, 2, 0], M[:, 2, 1], M[:, 2, 2]
            
            det_M = a*(e*i_val - f*h) - b*(d*i_val - f*g) + c*(d*h - e*g)
            volume = mx.abs(det_M * 10000000.0).item()
            token_volumes.append(volume)
        
    return original_call(self, x, mask=mask, cache=cache, **kwargs)

# 植入探针
type(target_layer_instance).__call__ = radar_forward_hook

# ========================================================
# 2. 扫锚文本：《后汉书·郡国志》
# ========================================================
# 这里请放一段后汉书的原文，不要太长，50-100个字最适合画图
text = "雒阳，二十七城，户五万二千八百三十九，口百一万八百二十七。有河南城。有荥阳。有京。有密。有卷。有阳武。"
print(f"📖 正在扫描文本: {text}")

tokens = tokenizer.encode(text)
# 解码回汉字，用于画在图表的 X 轴上
for t in tokens:
    token_labels.append(tokenizer.decode([t]))

# 执行一次前向传播，触发雷达探针！
input_ids = mx.array([tokens])
_ = model(input_ids)

# ========================================================
# 3. 绘制上帝视角的“语境密度图”
# ========================================================
print("📊 正在绘制《后汉书》特征几何密度图...")
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "sans-serif"]
plt.figure(figsize=(15, 6))

# 把极其微小或极大的数字做一个 Log 平滑，方便人眼观看
volumes_log = [np.log1p(v) for v in token_volumes]

plt.plot(range(len(token_labels)), volumes_log, marker='o', color='crimson', linewidth=2, markersize=8)
plt.fill_between(range(len(token_labels)), volumes_log, color='crimson', alpha=0.1)

plt.xticks(range(len(token_labels)), token_labels, rotation=45, fontsize=10)
plt.ylabel("特征流形体积 (Log Volume)", fontsize=12)
plt.title("《后汉书·郡国志》上下文几何密度扫描 (Vortex Radar)", fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('radar_houhanshu.png', dpi=300)
print("✅ 雷达图已保存为 'radar_houhanshu.png'！")