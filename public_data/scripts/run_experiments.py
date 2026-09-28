"""Baseline 对比 + 四段式消融 + 两阶段级联（双任务）。

## 任务

- **任务 A（同期识别，分类）**：用 [t-L+1 .. t] 的序列判 t 月该县有无气象灾害
- **任务 B（提前预测，回归）**：用同样的序列预测 t+1 / t+2 / t+3 三个窗口内的**最大灾情强度**

## 对比模型

| 组 | 模型 |
|---|---|
| 传统 | Logistic 回归、随机森林、XGBoost、CatBoost |
| 时序深度 | LSTM、GRU、Transformer |
| 本文 | TP-STNet（气象时序分支 + 遥感空间分支 + 跨模态注意力 + 物理约束） |

## 四段式消融（证明每一块都不是花架子）

1. `仅气象` —— 只有气象通道，无遥感、无图卷积
2. `+遥感` —— 加遥感通道，仍无图卷积
3. `+空间图卷积` —— 加上 26 县邻接图传播
4. `+物理约束` —— 加上 L_lapse_rate 与 L_snow_energy（= 完整 TP-STNet）

## 两阶段级联（对比项）

第一阶段用低阈值高召回粗筛，第二阶段只在被筛出的样本上重训一个精判模型。
在 1:14 的极端不平衡下，第二阶段样本更少，**很可能不稳**——所以它是
「实验对比项」而不是「默认方案」，结果如实报。

用法：
    python public_data/scripts/run_experiments.py            # 全量
    python public_data/scripts/run_experiments.py --quick    # 减少训练轮数
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             precision_recall_curve, precision_recall_fscore_support,
                             r2_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import meteo_dataset as MD                      # noqa: E402
import models_deep as MDeep                     # noqa: E402

OUT_JSON = BACKEND / "data_store" / "experiment_report.json"
TEST_YEARS = ("2022", "2023", "2024")
SEED = 42


# --------------------------------------------------------------------- 指标

def cls_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    out: dict = {"n": int(len(y)), "n_pos": int(y.sum())}
    if len(set(y.tolist())) < 2:
        return out
    out["auc"] = round(float(roc_auc_score(y, score)), 3)
    out["ap"] = round(float(average_precision_score(y, score)), 3)
    prec, rec, thr = precision_recall_curve(y, score)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    k = int(np.argmax(f1))
    out["best_f1"] = round(float(f1[k]), 3)
    out["best_f1_P"] = round(float(prec[k]), 3)
    out["best_f1_R"] = round(float(rec[k]), 3)
    ok = np.where(rec >= 0.9)[0]
    out["P_at_R90"] = round(float(prec[ok].max()), 3) if len(ok) else 0.0
    ok2 = np.where(prec >= 0.30)[0]
    out["R_at_P30"] = round(float(rec[ok2].max()), 3) if len(ok2) else 0.0
    return out


def reg_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "r2": round(float(r2_score(y, pred)), 3) if len(np.unique(y)) > 1 else None,
        "mae": round(float(mean_absolute_error(y, pred)), 3),
        "rmse": round(float(np.sqrt(np.mean((y - pred) ** 2))), 3),
    }


# ------------------------------------------------------------ torch 基线

class SeqBaseline(nn.Module):
    """LSTM / GRU / Transformer 的轻量基线，接口与 TP-STNet 对齐（输出 (B, N)）。"""

    def __init__(self, kind: str, n_feat: int, lookback: int, d_model: int = 48,
                 nhead: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.kind = kind
        self.proj = nn.Linear(n_feat, d_model)
        if kind in ("lstm", "gru"):
            rnn = nn.LSTM if kind == "lstm" else nn.GRU
            self.rnn = rnn(d_model, d_model, batch_first=True)
        else:
            layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2,
                                               dropout=dropout, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.head_h = nn.Linear(d_model, 3)

    def forward(self, seq: torch.Tensor) -> dict:
        b, L, n, f = seq.shape
        x = self.proj(seq.permute(0, 2, 1, 3).reshape(b * n, L, f))
        if self.kind in ("lstm", "gru"):
            out, _ = self.rnn(x)
            h = out[:, -1]
        else:
            h = self.enc(x).mean(dim=1)
        h = self.norm(h).reshape(b, n, -1)
        return {"risk_logit": self.head(h).squeeze(-1), "horizon": self.head_h(h),
                "temp_next": torch.zeros(b, n, device=seq.device),
                "snow_next": torch.zeros(b, n, device=seq.device)}


def standardize(seq: np.ndarray, tr_t: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """按**训练月份**拟合标准化器。seq 形状 (T, L, N, F)。"""
    T, L, N, F = seq.shape
    scaler = StandardScaler().fit(seq[tr_t].reshape(-1, F))
    out = scaler.transform(seq.reshape(-1, F)).reshape(T, L, N, F).astype(np.float32)
    return out, scaler


def flatten_for_trees(seq: np.ndarray) -> np.ndarray:
    """(T, L, N, F) → (T, N, L*F)，给树模型/线性模型用。"""
    T, L, N, F = seq.shape
    return seq.transpose(0, 2, 1, 3).reshape(T, N, L * F)


def train_torch(model: nn.Module, seq: np.ndarray, batch: dict, tr_t: np.ndarray,
                te_t: np.ndarray, use_physics: bool, epochs: int, lr: float = 2e-3,
                device: str = "cpu") -> dict:
    """按「月」为 batch 单位训练；返回测试月上的预测。"""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    adj = torch.as_tensor(batch["adj"], dtype=torch.float32, device=device)
    elev = torch.as_tensor(batch["elev"], dtype=torch.float32, device=device)

    pos = float(batch["cls"][tr_t].sum())
    neg = float(batch["cls"][tr_t].size - pos)
    focal = MDeep.FocalLoss(alpha=min(0.95, max(0.5, neg / max(1.0, pos + neg))))
    data_tr_t = tr_t
    for ep in range(epochs):
        model.train()
        idx = np.random.default_rng(SEED + ep).permutation(len(data_tr_t))
        for s in range(0, len(idx), 8):
            sel = data_tr_t[idx[s:s + 8]]
            s_t = torch.as_tensor(seq[sel], device=device)
            b = {"adj": batch["adj"], "elev": batch["elev"],
                 "cls": torch.as_tensor(batch["cls"][sel], device=device),
                 "valid_a": torch.as_tensor(batch["valid_a"][sel], device=device),
                 "valid_b": torch.as_tensor(batch["valid_b"][sel], device=device),
                 "horizon_reg": torch.as_tensor(batch["horizon_reg"][sel], device=device),
                 "snow_prev": torch.as_tensor(batch["snow_prev"][sel], device=device),
                 "snow_now": torch.as_tensor(batch["snow_now"][sel], device=device),
                 "t_now": torch.as_tensor(batch["t_now"][sel], device=device),
                 "snowfall_cm": torch.as_tensor(batch["snowfall_cm"][sel], device=device),
                 "rad_mj": torch.as_tensor(batch["rad_mj"][sel], device=device)}
            out = model(s_t)
            if use_physics:
                losses = MDeep.build_losses(out, b, adj, elev)
                loss = losses["total"]
            else:
                y = b["cls"][b["valid_a"] > 0]
                loss = focal(out["risk_logit"][b["valid_a"] > 0], y)
                m = b["valid_b"].unsqueeze(-1).expand_as(b["horizon_reg"])
                loss = loss + F.mse_loss(out["horizon"][m > 0], b["horizon_reg"][m > 0])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        s_t = torch.as_tensor(seq[te_t], device=device)
        out = model(s_t)
    return {"score": torch.sigmoid(out["risk_logit"]).cpu().numpy(),
            "horizon": out["horizon"].cpu().numpy()}


# --------------------------------------------------------------------- 主流程

def get_design_data() -> dict:
    """从数据集里额外取出物理约束需要的量。"""
    d = np.load(BACKEND / "data_store" / "meteo_dataset.npz")
    meta = json.loads((BACKEND / "data_store" / "meteo_dataset_meta.json").read_text(encoding="utf-8"))
    names = meta["monthly_features"]
    meteo = d["meteo"]                       # (T, N, F)
    idx = {n: i for i, n in enumerate(names)}
    snow_prev = np.zeros_like(meteo[:, :, 0])
    snow_now = meteo[:, :, idx["snow_depth_max"]]
    snow_prev[1:] = snow_now[:-1]
    return {
        "adj": d["adjacency_norm"], "elev": d["elevation"],
        "cls": d["cls"], "horizon_reg": d["horizon_reg"],
        "valid_a": np.ones_like(d["cls"]), "valid_b": d["valid_b"],
        "snow_prev": snow_prev, "snow_now": snow_now,
        # 物理项尺度归一化要用**观测**的当月气温（不是预测值，避免循环）
        "t_now": meteo[:, :, idx["t_mean"]],
        "snowfall_cm": meteo[:, :, idx["snowfall_sum"]],
        "rad_mj": meteo[:, :, idx["rad_sum"]],
    }


def main(argv: list[str]) -> int:
    quick = "--quick" in argv
    epochs = 30 if quick else 80
    torch.manual_seed(SEED)

    npz = np.load(BACKEND / "data_store" / "meteo_dataset.npz")
    labels = json.loads((BACKEND / "data_store" / "meteo_event_labels.json").read_text(encoding="utf-8"))["records"]
    month_axis = sorted({r["month"] for r in labels})            # 与 npz 的 T 维一致
    meta = json.loads((BACKEND / "data_store" / "meteo_dataset_meta.json").read_text(encoding="utf-8"))
    batch = get_design_data()

    T, L, N, Fn = npz["seq"].shape
    print(f"数据集：{T} 月 × {N} 县 × 回看 {L} 月 × {Fn} 特征；"
          f"气象类正例 {int(npz['cls'].sum())} 格（占比 {npz['cls'].mean()*100:.2f}%）")
    if T != len(month_axis):
        print(f"⚠️ 月份轴长度 {len(month_axis)} 与张量 {T} 不一致", file=sys.stderr)

    seq_raw = npz["seq"]                                  # (T, L, N, F)
    flat_raw = flatten_for_trees(seq_raw)                 # (T, N, L*F)
    results: list[dict] = []

    for year in TEST_YEARS:
        te_t = np.array([i for i, m in enumerate(month_axis) if m.startswith(year)])
        tr_t = np.array([i for i, m in enumerate(month_axis) if m < f"{year}-01"])
        if len(te_t) == 0 or len(tr_t) == 0:
            continue
        n_pos_tr = int(npz["cls"][tr_t].sum())
        print(f"\n===== 测试年 {year}：训练 {len(tr_t)} 月（正例 {n_pos_tr} 格）"
              f"，测试 {len(te_t)} 月 =====")

        # ---- 静态特征标准化（树模型用展平序列）----
        scaler = StandardScaler().fit(flat_raw[tr_t].reshape(-1, L * Fn))
        Xtr = scaler.transform(flat_raw[tr_t].reshape(-1, L * Fn))
        Xte = scaler.transform(flat_raw[te_t].reshape(-1, L * Fn))
        ytr_cls = npz["cls"][tr_t].reshape(-1)
        yte_cls = npz["cls"][te_t].reshape(-1)
        ytr_reg = npz["horizon_reg"][tr_t].reshape(-1, npz["horizon_reg"].shape[-1])
        yte_reg = npz["horizon_reg"][te_t].reshape(-1, npz["horizon_reg"].shape[-1])

        # ---- 传统模型 ----
        trad = {
            "Logistic回归": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
            "随机森林": RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                             class_weight="balanced", random_state=SEED),
        }
        try:
            from xgboost import XGBClassifier
            trad["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                                            subsample=0.9, colsample_bytree=0.9,
                                            scale_pos_weight=max(1.0, (ytr_cls == 0).sum() / max(1, ytr_cls.sum())),
                                            eval_metric="logloss", random_state=SEED)
        except Exception:
            pass
        try:
            from catboost import CatBoostClassifier
            trad["CatBoost"] = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.08,
                                                  verbose=0, random_seed=SEED,
                                                  auto_class_weights="Balanced")
        except Exception as exc:
            print(f"  （跳过 CatBoost：{exc}）")

        for name, clf in trad.items():
            clf.fit(Xtr, ytr_cls)
            score = clf.predict_proba(Xte)[:, 1]
            row = {"test_year": year, "group": "传统模型", "model": name,
                   "task_a": cls_metrics(yte_cls, score)}
            if hasattr(clf, "predict"):
                pred_reg = clf.predict(Xte)
                if pred_reg.ndim == 1:
                    pred_reg = pred_reg[:, None]
                row["task_b"] = {f"h{h+1}": reg_metrics(yte_reg[:, h], pred_reg[:, 0])
                                 for h in range(yte_reg.shape[1])}
            results.append(row)

        # ---- 时序深度基线 ----
        seq_std, _ = standardize(seq_raw, tr_t)
        for kind, label in (("lstm", "LSTM"), ("gru", "GRU"), ("transformer", "Transformer")):
            torch.manual_seed(SEED)
            model = SeqBaseline(kind, Fn, L)
            out = train_torch(model, seq_std, batch, tr_t, te_t, use_physics=False, epochs=epochs)
            results.append({"test_year": year, "group": "时序深度基线", "model": label,
                            "task_a": cls_metrics(yte_cls, out["score"].reshape(-1)),
                            "task_b": {f"h{h+1}": reg_metrics(yte_reg[:, h], out["horizon"].reshape(-1, yte_reg.shape[1])[:, h])
                                       for h in range(yte_reg.shape[1])}})

        # ---- 本文模型：四段式消融 ----
        ablations = [
            ("仅气象", dict(no_remote=True, no_graph=True, physics=False)),
            ("+遥感", dict(no_remote=False, no_graph=True, physics=False)),
            ("+空间图卷积", dict(no_remote=False, no_graph=False, physics=False)),
            ("+物理约束(完整)", dict(no_remote=False, no_graph=False, physics=True)),
        ]
        for label, cfg in ablations:
            torch.manual_seed(SEED)
            seq_use = seq_std.copy()
            if cfg["no_remote"]:
                seq_use[:, :, :, len(meta["monthly_features"]):] = 0.0     # 屏蔽遥感通道
            model, _ = MDeep.build_model()
            if cfg["no_graph"]:
                with torch.no_grad():
                    model.adj.copy_(torch.eye(model.adj.shape[0]))           # 关掉空间传播
            out = train_torch(model, seq_use, batch, tr_t, te_t,
                              use_physics=cfg["physics"], epochs=epochs)
            results.append({"test_year": year, "group": "本文模型(消融)", "model": label,
                            "task_a": cls_metrics(yte_cls, out["score"].reshape(-1)),
                            "task_b": {f"h{h+1}": reg_metrics(yte_reg[:, h], out["horizon"].reshape(-1, yte_reg.shape[1])[:, h])
                                       for h in range(yte_reg.shape[1])}})

        # ---- 两阶段级联（对比项）----
        torch.manual_seed(SEED)
        model, _ = MDeep.build_model()
        out1 = train_torch(model, seq_std, batch, tr_t, te_t, use_physics=True, epochs=epochs)
        s1 = out1["score"].reshape(-1)
        thr = float(np.quantile(s1, 0.85))                     # 粗筛：留下风险最高的一小部分
        keep = s1 >= thr
        score2 = np.zeros_like(s1)
        if keep.sum() >= 20 and len(set(yte_cls[keep].tolist())) > 1:
            sc = StandardScaler().fit(Xtr)
            # 第二阶段：用「第一阶段判为高风险」的样本重训一个精判模型
            model2, _ = MDeep.build_model()
            # 训练侧同样先做一次粗筛（用训练集自身风险分数）
            torch.manual_seed(SEED)
            model0, _ = MDeep.build_model()
            tr_out = train_torch(model0, seq_std, batch, tr_t, tr_t, use_physics=True, epochs=max(10, epochs // 3))
            tr_score = tr_out["score"].reshape(-1)
            tr_keep = tr_score >= float(np.quantile(tr_score, 0.5))
            cls2 = RandomForestClassifier(n_estimators=200, min_samples_leaf=1,
                                          class_weight="balanced", random_state=SEED)
            if len(set(ytr_cls[tr_keep].tolist())) > 1:
                cls2.fit(Xtr[tr_keep], ytr_cls[tr_keep])
                score2[keep] = cls2.predict_proba(Xte[keep])[:, 1]
        results.append({"test_year": year, "group": "两阶段级联",
                        "model": "粗筛+精判(RF)",
                        "task_a": cls_metrics(yte_cls, score2),
                        "note": f"粗筛保留 {int(keep.sum())}/{len(keep)} 个测试格；"
                                f"第二阶段训练样本 {int(tr_keep.sum())} 格"})

        print(f"  {year} 完成，已跑 {len(results)} 行")

    _print_table(results)
    OUT_JSON.write_text(json.dumps({
        "generated_by": "public_data/scripts/run_experiments.py",
        "dataset": "backend/data_store/meteo_dataset.npz",
        "protocol": "滚动前向：训练 = 全部早于测试年的月份；测试 = 该年全年；样本单位 = (月, 县)",
        "epochs": epochs, "seed": SEED,
        "caveats": [
            "负例＝未确认月，不等同确认无灾。",
            "测试年正样本仅一二十格，单格变动即带来数个百分点抖动；应看多年一致性而非单年峰值。",
            "best_f1 / P@R90 / R@P30 由测试集阈值曲线给出，属乐观口径。",
            "任务 B 的 R² 以「未来 h 月最大灾情强度」（0~4 文字档刻度）为目标，不是物理量纲。",
        ],
        "results": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细已写出：{OUT_JSON}")
    return 0


def _print_table(results: list[dict]) -> None:
    print("\n" + "=" * 108)
    print(f"{'年':<5}{'组':<14}{'模型':<18}{'AUC':>7}{'AP':>7}{'最优F1':>8}"
          f"{'P@R90':>7}{'R@P30':>7}{'R²(h1)':>8}{'MAE(h1)':>9}")
    print("-" * 108)
    for r in results:
        a = r.get("task_a", {})
        b = r.get("task_b", {}).get("h1", {})
        r2 = b.get("r2")
        print(f"{r['test_year']:<5}{r['group']:<14}{r['model']:<18}"
              f"{a.get('auc', 0):>7.3f}{a.get('ap', 0):>7.3f}{a.get('best_f1', 0):>8.3f}"
              f"{a.get('P_at_R90', 0):>7.3f}{a.get('R_at_P30', 0):>7.3f}"
              f"{(r2 if r2 is not None else float('nan')):>8.3f}{b.get('mae', 0):>9.3f}")
    print("=" * 108)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
