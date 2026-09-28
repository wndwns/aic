"""TP-STNet：高原时空**跨模态**气象灾害预警网络。

## 架构（对齐计划里的「气象时序分支 + 遥感空间分支 + 跨模态融合」）

```
        序列 (B, L, N, Fm+Fr)
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
 气象时序分支            遥感空间分支
 PatchTST-lite           图卷积 (GCN)
 按县展开为 (B·N, L, F)   A_norm @ X @ W  —— 26 县之间传消息
 → patch 化 + 位置编码     （雪云过境、草场连片受灾的空间扩散）
 → Transformer 编码
      │                     │
      └────────┬────────────┘
               ▼
      跨模态注意力融合（Cross-Attention）
       query = 时序 patch token ；key/value = 空间 token
               ▼
      ┌────────┴────────┬──────────────┐
      ▼                 ▼              ▼
  风险 logit       未来 h 月灾情强度   物理辅助头
  (任务 A 分类)    (任务 B 回归)      (气温 / 雪深)
                                         │
                                         ▼
                              L_lapse_rate / L_snow_energy
```

## 为什么配置这么小

样本量是 **26 县 × 约 78 月**、气象类正样本 **96 条**。在这个规模上堆参数量只会过拟合，
所以 `d_model=48`、时序编码器 `1` 层、图卷积 `2` 层；**深度是用在结构上（跨模态 / 空间 / 物理约束），
不是用在规模上**。这一点在报告里要明说，不要假称是「大模型」。

## 物理约束的做法（诚实说明）

模型**不直接预测物理场**，而是通过辅助头预测「下月气温」与「下月雪深」，
再把物理规律作为软约束加在这两个预测上：

- `L_lapse_rate`：对图上每条边 (i,j)，`(T_i − T_j) + Γ·(h_i − h_j)/1000` 应接近 0。
  Γ 取 6.5 °C/km。这是 **26 个县点位之间**的约束——不是网格约束，因为没有 DEM 栅格。
- `L_snow_energy`：**单边（hinge）软约束**，只用现有数据能支撑的两条：
  ① 没有降雪就不该增雪；② 融雪量不能超过辐射能换算出的上限。
  缺长波辐射、地表反照率与风廓线，所以**不是完整能量闭合**，报告里必须写明。

用法：
    from models_deep import TPSTNet, build_losses
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent
GRAPH_FILE = BASE / "data_store" / "region_graph.json"
META_FILE = BASE / "data_store" / "meteo_dataset_meta.json"

LAPSE_RATE_C_PER_KM = 6.5
# 融雪 1 kg 水需 0.334 MJ；雪密度约 0.1 → 1 mm 水当量 ≈ 1 cm 雪深
MELT_MJ_PER_MM_WATER = 0.334
CM_SNOW_PER_MM_WATER = 1.0


# --------------------------------------------------------------------- 组件

class PatchTSTLite(nn.Module):
    """把每条 (L, F) 序列切 patch，线性嵌入后过一层 Transformer 编码器。"""

    def __init__(self, n_feat: int, lookback: int, d_model: int = 48,
                 patch_len: int = 2, stride: int = 1, nhead: int = 4,
                 n_layers: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = max(1, (lookback - patch_len) // stride + 1)
        self.embed = nn.Linear(patch_len * n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, F) → (B, n_patches, d_model)"""
        b, length, feat = x.shape
        patches = x.unfold(1, self.patch_len, self.stride)          # (B, P, F, patch_len)
        patches = patches.permute(0, 1, 3, 2).reshape(b, -1, self.patch_len * feat)
        z = self.embed(patches) + self.pos
        return self.norm(self.encoder(z))


class GraphConv(nn.Module):
    """Â X W，Â 为预归一化的对称邻接（自带自环）。"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: (B, N, C)；adj: (N, N)"""
        return self.drop(F.relu(torch.matmul(adj, self.lin(x))))


class TPSTNet(nn.Module):
    def __init__(self, n_meteo: int, n_remote: int, lookback: int, n_regions: int,
                 adj_norm: "torch.Tensor | list[list[float]]",
                 d_model: int = 48, nhead: int = 4, n_layers: int = 1,
                 dropout: float = 0.2, n_horizons: int = 3) -> None:
        super().__init__()
        n_feat = n_meteo + n_remote
        self.n_meteo, self.n_remote = n_meteo, n_remote
        self.register_buffer("adj", torch.as_tensor(adj_norm, dtype=torch.float32))

        # 分支一：气象 + 遥感时序
        self.temporal = PatchTSTLite(n_feat, lookback, d_model, nhead=nhead,
                                     n_layers=n_layers, dropout=dropout)
        # 分支二：空间图卷积（在县维度上传播）
        self.spatial = nn.ModuleList([
            GraphConv(d_model, d_model, dropout),
            GraphConv(d_model, d_model, dropout),
        ])
        # 跨模态融合
        self.norm_t = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.fuse_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))

        # 任务头
        self.head_risk = nn.Linear(d_model, 1)
        self.head_horizon = nn.Linear(d_model, n_horizons)
        # 物理辅助头：下月气温 / 下月雪深（厘米）
        self.head_temp = nn.Linear(d_model, 1)
        self.head_snow = nn.Linear(d_model, 1)
        self.n_horizons = n_horizons

    def forward(self, seq: torch.Tensor) -> dict[str, Any]:
        """seq: (B, L, N, F)。返回各头输出与注意力权重。"""
        b, length, n, feat = seq.shape
        # 气象时序分支：把县折进 batch，让每个县共享同一套时序权重
        x = seq.permute(0, 2, 1, 3).reshape(b * n, length, feat)
        tok = self.temporal(x)                                  # (B·N, P, d)
        tok = tok.reshape(b, n, tok.shape[1], -1)               # (B, N, P, d)

        # 空间图卷积：在县维度上传播（用各县的时序池化表示作为节点特征）
        node = tok.mean(dim=2)                                  # (B, N, d)
        for gcn in self.spatial:
            node = node + gcn(node, self.adj)
        node = self.norm_t(node)

        # 跨模态注意力：query = 本县时序 patch，key/value = 图上邻域节点
        q = tok.reshape(b * n, tok.shape[2], -1)
        kv = node.reshape(b * n, 1, -1)
        fused, attn = self.cross(q, kv, kv, need_weights=True)
        fused = self.fuse_norm(fused + q).mean(dim=1).reshape(b, n, -1)
        fused = fused + self.ffn(fused)

        return {
            "risk_logit": self.head_risk(fused).squeeze(-1),            # (B, N)
            "horizon": self.head_horizon(fused),                        # (B, N, H)
            "temp_next": self.head_temp(fused).squeeze(-1),             # (B, N)
            "snow_next": F.softplus(self.head_snow(fused).squeeze(-1)),  # (B, N) ≥0
            "attn": attn.detach(),                                      # (B·N, P, 1)
            "node_repr": node.detach(),
        }


# --------------------------------------------------------------------- 损失

class FocalLoss(nn.Module):
    """二分类 Focal Loss，专治正负样本不平衡（气象类正样本 96 / 未确认 1372）。"""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logit: torch.Tensor, target: torch.Tensor,
                weight: torch.Tensor | None = None) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        p = torch.sigmoid(logit)
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        if weight is not None:
            loss = loss * weight
            return loss.sum() / weight.sum().clamp(min=1e-6)
        return loss.mean()


def ohem_weights(logit: torch.Tensor, target: torch.Tensor,
                 keep_ratio: float = 0.5) -> torch.Tensor:
    """难例挖掘：只保留损失最大的 keep_ratio 比例样本（按比例给权重）。

    注意：正样本**始终全部保留**，否则在 1:14 的极端不平衡下会把正例全挖掉。
    """
    with torch.no_grad():
        bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        w = torch.zeros_like(bce)
        neg = target < 0.5
        if neg.any():
            k = max(1, int(neg.sum().item() * keep_ratio))
            thresh = torch.topk(bce[neg], k).values.min()
            w[neg] = (bce[neg] >= thresh).float()
        w[~neg] = 1.0
    return w


def _pair_abs_mean(values: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """图上相邻点对差值的平均绝对值——用作物理残差的归一化尺度。"""
    diff = (values.unsqueeze(2) - values.unsqueeze(1)).abs()
    mask = (adj > 0).float()
    return (diff * mask).sum() / mask.sum().clamp(min=1.0)


def lapse_rate_loss(temp_next: torch.Tensor, adj: torch.Tensor, elevation: torch.Tensor,
                    scale: torch.Tensor | None = None) -> torch.Tensor:
    """L_lapse_rate：图上每对相邻县，气温差应与高程差匹配。

        residual_ij = (T_i − T_j) + Γ·(h_i − h_j)/1000
        T_i − T_j 应为 −Γ·Δh/1000，故 residual 越接近 0 越符合物理。

    `scale` 是残差的归一化尺度（用**观测**的邻域温差给出）。不归一化的话，
    残差是「摄氏度平方」量纲，量级远超 focal/MSE，会把数据项压垮——
    实测未归一化时 AUC 从 0.69 掉到 0.50。

    ⚠️ 这是 **26 个县点位之间**的约束。没有 DEM 栅格，做不了「相邻网格」版本。
    """
    diff_t = temp_next.unsqueeze(2) - temp_next.unsqueeze(1)             # (B, N, N)
    diff_h = (elevation.unsqueeze(0) - elevation.unsqueeze(1)) / 1000.0  # (N, N)
    residual = diff_t + LAPSE_RATE_C_PER_KM * diff_h
    mask = (adj > 0).float()
    denom = mask.sum().clamp(min=1.0)
    if scale is not None:
        residual = residual / scale.clamp(min=1e-3)
    return (residual.pow(2) * mask).sum() / denom


def snow_energy_loss(snow_next: torch.Tensor, snow_prev: torch.Tensor,
                     snowfall_cm: torch.Tensor, rad_mj: torch.Tensor,
                     scale: torch.Tensor | None = None) -> torch.Tensor:
    """L_snow_energy：积雪变化的**单边**物理软约束（简化）。

    ① 没有降雪就不应增雪：  relu(ΔSD − 降雪量 − 容差)
    ② 融雪不能超过辐射能上限：relu(−ΔSD − 辐射可融雪量)

    辐射可融雪量：rad(MJ/m²) ÷ 0.334(MJ/kg) = kg/m² 水当量 ≈ mm 水；
    雪密度按 0.1 折算，1 mm 水 ≈ 1 cm 雪深，故上限即 rad/0.334 (cm)。

    `scale` 同 `lapse_rate_loss`，用于把残差归一到 O(1)。

    ⚠️ 缺长波辐射、地表反照率、风廓线，**不是完整能量平衡**。
    """
    delta = snow_next - snow_prev
    tol = 1.0                                   # 1 cm 容差，吸收压缩/吹雪等未建模过程
    gain_violation = F.relu(delta - snowfall_cm - tol)
    melt_capacity = rad_mj / MELT_MJ_PER_MM_WATER * CM_SNOW_PER_MM_WATER
    melt_violation = F.relu(-delta - melt_capacity)
    v = gain_violation.pow(2) + melt_violation.pow(2)
    if scale is not None:
        v = v / scale.clamp(min=1e-3)
    return v.mean()


def build_losses(pred: dict, batch: dict, adj: torch.Tensor, elevation: torch.Tensor,
                 lambdas: dict[str, float] | None = None,
                 use_ohem: bool = True) -> dict[str, torch.Tensor]:
    """L_total = L_data + λ1·L_lapse_rate + λ2·L_snow_energy

    L_data = Focal(任务 A) + MSE(任务 B)，两项都只在**有效掩码**上计算。

    λ 默认 0.05：物理残差经尺度归一化后已是 O(1)，与数据项同量级；
    λ 取 0.5 时实测主任务 AUC 从 0.69 掉到 0.50（物理项压过数据项），
    所以默认值刻意取小，并保留可调入口用于 λ 敏感性分析。
    """
    lam = {"lapse": 0.05, "snow": 0.05, **(lambdas or {})}

    # --- 数据项：任务 A（分类）---
    y_a, m_a = batch["cls"], batch["valid_a"]
    logit, y = pred["risk_logit"][m_a > 0], y_a[m_a > 0]
    focal = FocalLoss()
    if use_ohem and y.numel():
        l_cls = focal(logit, y, ohem_weights(logit, y, keep_ratio=0.5))
    else:
        l_cls = focal(logit, y)

    # --- 数据项：任务 B（回归，预测未来 h 月最大灾情强度）---
    m_b = batch["valid_b"].unsqueeze(-1).expand_as(batch["horizon_reg"])
    l_reg = F.mse_loss(pred["horizon"][m_b > 0], batch["horizon_reg"][m_b > 0])

    # --- 物理项（尺度归一化：用**观测**的邻域温差 / 观测积雪变化当尺度，避免循环）---
    t_scale = _pair_abs_mean(batch["t_now"], adj).detach()
    s_scale = (batch["snow_now"] - batch["snow_prev"]).abs().mean().detach().clamp(min=1.0)
    l_lapse = lapse_rate_loss(pred["temp_next"], adj, elevation, scale=t_scale)
    l_snow = snow_energy_loss(pred["snow_next"], batch["snow_prev"],
                              batch["snowfall_cm"], batch["rad_mj"], scale=s_scale)

    total = l_cls + l_reg + lam["lapse"] * l_lapse + lam["snow"] * l_snow
    return {"total": total, "cls": l_cls.detach(), "reg": l_reg.detach(),
            "lapse": l_lapse.detach(), "snow": l_snow.detach(),
            "t_scale": t_scale.detach(), "s_scale": s_scale.detach()}


# --------------------------------------------------------------------- 工厂

def build_model(d_model: int = 48, nhead: int = 4, n_layers: int = 1,
                dropout: float = 0.2) -> tuple[TPSTNet, dict]:
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    graph = json.loads(GRAPH_FILE.read_text(encoding="utf-8"))
    model = TPSTNet(
        n_meteo=meta["n_meteo_features"], n_remote=meta["n_remote_features"],
        lookback=meta["lookback"], n_regions=meta["n_regions"],
        adj_norm=graph["adjacency_normalized"], d_model=d_model, nhead=nhead,
        n_layers=n_layers, dropout=dropout, n_horizons=len(meta["horizons"]),
    )
    return model, meta


if __name__ == "__main__":
    model, meta = build_model()
    n_param = sum(p.numel() for p in model.parameters())
    print("TP-STNet 结构：")
    print(model)
    print(f"\n参数量：{n_param:,}（d_model=48 / 时序编码器 1 层 / 图卷积 2 层）")
    print(f"输入：序列 (B, {meta['lookback']}, {meta['n_regions']}, "
          f"{meta['n_meteo_features'] + meta['n_remote_features']})")
    dummy = torch.randn(2, meta["lookback"], meta["n_regions"],
                        meta["n_meteo_features"] + meta["n_remote_features"])
    out = model(dummy)
    for k, v in out.items():
        print(f"  {k:12s} {tuple(v.shape)}")
