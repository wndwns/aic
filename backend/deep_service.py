"""深度模型推理服务：加载/训练 TP-STNet 并对外提供风险、注意力与归因。

设计取向：**可复现 + 可解释 + 不假装**。

- 训练配置（轮数 / 种子 / λ）全部写进 checkpoint，`status()` 会回显，便于答辩复现。
- 归因用 **通道遮挡（occlusion）** 而不是 SHAP：把某一类输入置零，看风险下降多少。
  它的语义是「模型对该通道的依赖度」，可以直接对上物理含义（温度 / 降水 / 积雪 / 辐射 / 风 / 湿度 / 遥感），
  比 SHAP 更经得起「这个数怎么算的」这一问。
- 物理残差（递减率 / 积雪能量）单独作为「物理归因分量」输出，不做成好看的百分比。

用法：
    from deep_service import get_service
    svc = get_service()
    svc.status(); svc.ensure_trained(); svc.predict("2024-12")
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent
STORE = BASE / "data_store"
NPZ = STORE / "meteo_dataset.npz"
META = STORE / "meteo_dataset_meta.json"
CKPT = STORE / "tpstnet.pt"

# 把 29 维气象特征归成可解释的物理组，供遮挡归因使用
FEATURE_GROUPS: dict[str, list[str]] = {
    "温度": ["t_mean", "t_max", "t_min", "t_range", "t_min_abs", "cold_days",
             "severe_cold_days", "frost_days", "t_anom", "d_t"],
    "降水": ["p_sum", "p_max", "rain_days", "max_dry_run", "p_anom_z"],
    "积雪": ["snowfall_sum", "snow_days", "snow_depth_max", "snow_depth_mean",
             "snow_depth_trend"],
    "辐射": ["rad_sum", "rad_per_day", "melt_index", "energy_index"],
    "风": ["wind_max", "wind_mean", "wind_dir_sin", "wind_dir_cos"],
    "湿度": ["rh_mean"],
    "遥感": ["ndvi", "ndvi_change_pct", "vegetation_cover_pct", "snow_cover_pct",
             "degradation_level"],
}

TRAIN_EPOCHS = 80
SEED = 42


class DeepService:
    def __init__(self) -> None:
        self._loaded = False
        self._model: Any = None
        self._meta: dict = {}
        self._data: dict = {}
        self._months: list[str] = []
        self._trained = False

    # ---------------------------------------------------------------- 基础

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import meteo_dataset as MD          # noqa: F401  （确保同目录可导入）
        import models_deep as MDeep

        if not NPZ.exists():
            raise FileNotFoundError(
                f"缺少数据集 {NPZ}；请先跑 python backend/meteo_dataset.py")

        d = np.load(NPZ)
        meta = json.loads(META.read_text(encoding="utf-8"))
        labels = json.loads((STORE / "meteo_event_labels.json").read_text(encoding="utf-8"))["records"]
        months = sorted({r["month"] for r in labels})

        names = meta["monthly_features"]
        idx = {n: i for i, n in enumerate(names)}
        meteo = d["meteo"]
        snow_now = meteo[:, :, idx["snow_depth_max"]]
        snow_prev = np.zeros_like(snow_now)
        snow_prev[1:] = snow_now[:-1]

        self._data = {
            "seq": d["seq"], "cls": d["cls"], "horizon_reg": d["horizon_reg"],
            "meteo": meteo,                     # (T, N, F) 原始月度气象场，供 /fields 使用
            "valid_a": np.ones_like(d["cls"]), "valid_b": d["valid_b"],
            "adj": d["adjacency_norm"], "elev": d["elevation"],
            "snow_prev": snow_prev, "snow_now": snow_now,
            "t_now": meteo[:, :, idx["t_mean"]],
            "snowfall_cm": meteo[:, :, idx["snowfall_sum"]],
            "rad_mj": meteo[:, :, idx["rad_sum"]],
            "region_ids": [r["region_id"] for r in json.loads(
                (STORE / "region_graph.json").read_text(encoding="utf-8"))["regions"]],
            "region_names": {r["region_id"]: r["region_name"] for r in json.loads(
                (STORE / "region_graph.json").read_text(encoding="utf-8"))["regions"]},
        }
        self._meta = meta
        self._months = months
        self._MDeep = MDeep
        self._loaded = True

    # ---------------------------------------------------------------- 训练

    def ensure_trained(self, force: bool = False) -> dict:
        self._ensure_loaded()
        if self._trained and not force:
            return self.status()
        if CKPT.exists() and not force:
            try:
                blob = torch.load(CKPT, map_location="cpu", weights_only=False)
                model, _ = self._MDeep.build_model()
                model.load_state_dict(blob["state_dict"])
                model.eval()
                self._model = model
                self._trained = True
                self._ckpt_info = blob.get("info", {})
                return self.status()
            except Exception:
                pass
        return self.train(force=True)

    def train(self, force: bool = False) -> dict:
        self._ensure_loaded()
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        d = self._data
        T = d["seq"].shape[0]
        tr_t = np.arange(0, max(1, T - 12))          # 留出最后一年做验证
        model, _ = self._MDeep.build_model()
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        seq = torch.as_tensor(d["seq"], dtype=torch.float32)
        adj = torch.as_tensor(d["adj"], dtype=torch.float32)
        elev = torch.as_tensor(d["elev"], dtype=torch.float32)

        pos = float(d["cls"][tr_t].sum())
        neg = float(d["cls"][tr_t].size - pos)
        focal = self._MDeep.FocalLoss(alpha=min(0.95, max(0.5, neg / max(1.0, pos + neg))))
        history: list[float] = []
        for ep in range(TRAIN_EPOCHS):
            model.train()
            order = np.random.default_rng(SEED + ep).permutation(len(tr_t))
            ep_loss = 0.0
            for s in range(0, len(order), 8):
                sel = tr_t[order[s:s + 8]]
                b = {k: torch.as_tensor(d[k][sel], dtype=torch.float32)
                     for k in ("cls", "valid_a", "valid_b", "horizon_reg",
                               "snow_prev", "snow_now", "snowfall_cm", "rad_mj", "t_now")}
                out = model(seq[sel])
                loss = self._MDeep.build_losses(out, b, adj, elev)["total"]
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += float(loss)
            history.append(round(ep_loss, 4))
        model.eval()
        self._model = model
        self._trained = True
        info = {"epochs": TRAIN_EPOCHS, "seed": SEED, "train_months": int(len(tr_t)),
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "loss_curve_tail": history[-5:]}
        self._ckpt_info = info
        torch.save({"state_dict": model.state_dict(), "info": info}, CKPT)
        return self.status()

    # ---------------------------------------------------------------- 推理

    def _month_index(self, month: str | None) -> int:
        if not self._months:
            return 0
        if month and month in self._months:
            return self._months.index(month)
        return len(self._months) - 1

    def predict(self, month: str | None = None) -> dict:
        self.ensure_trained()
        i = self._month_index(month)
        seq = torch.as_tensor(self._data["seq"][i:i + 1], dtype=torch.float32)
        with torch.no_grad():
            out = self._model(seq)
        risk = torch.sigmoid(out["risk_logit"])[0].cpu().numpy()
        horizon = out["horizon"][0].cpu().numpy()
        attn = out["attn"].cpu().numpy().reshape(len(self._data["region_ids"]), -1)
        rows = []
        for k, rid in enumerate(self._data["region_ids"]):
            rows.append({
                "region_id": rid,
                "region_name": self._data["region_names"].get(rid, rid),
                "risk_score": round(float(risk[k]), 4),
                "risk_level": _level(float(risk[k])),
                "horizon_max_severity": [round(float(v), 3) for v in horizon[k]],
                "attention_peak": round(float(np.max(attn[k])), 4),
            })
        rows.sort(key=lambda r: -r["risk_score"])
        return {
            "month": self._months[i],
            "model": "TP-STNet",
            "n_regions": len(rows),
            "regions": rows,
            "caveats": [
                "risk_score 是模型输出概率，不是灾害发生概率的校准值。",
                "训练标签为带公开凭证的气象灾害事件；其余月份为「未确认」，不等同无灾。",
            ],
        }

    def attention(self, month: str | None = None) -> dict:
        """跨模态注意力的时间分布（注意力热力图数据源）。"""
        self.ensure_trained()
        i = self._month_index(month)
        seq = torch.as_tensor(self._data["seq"][i:i + 1], dtype=torch.float32)
        with torch.no_grad():
            out = self._model(seq)
        a = out["attn"].cpu().numpy().reshape(len(self._data["region_ids"]), -1)
        lo = max(0, i - self._meta["lookback"] + 1)
        window = self._months[lo:i + 1]
        if len(window) < a.shape[1]:
            window = [window[0]] * (a.shape[1] - len(window)) + window
        return {
            "month": self._months[i],
            "lookback_months": window,
            "regions": self._data["region_ids"],
            "attention": [[round(float(v), 5) for v in row] for row in a],
            "note": "注意力为跨模态注意力对时间 patch 的权重，用于展示模型关注的时段，不等同因果归因。",
        }

    def attribution(self, region_id: str, month: str | None = None) -> dict:
        """通道遮挡归因 + 物理残差分量。"""
        self.ensure_trained()
        if region_id not in self._data["region_ids"]:
            raise KeyError(f"未知 region_id: {region_id}")
        k = self._data["region_ids"].index(region_id)
        i = self._month_index(month)
        names = self._meta["monthly_features"] + self._meta["remote_features"]
        n_meteo = self._meta["n_meteo_features"]

        base = self._data["seq"][i:i + 1].copy()
        seq = torch.as_tensor(base, dtype=torch.float32)
        with torch.no_grad():
            base_risk = float(torch.sigmoid(self._model(seq)["risk_logit"][0, k]))

        drops: dict[str, float] = {}
        for group, feats in FEATURE_GROUPS.items():
            cols = [names.index(f) for f in feats if f in names]
            if not cols:
                continue
            ablated = base.copy()
            ablated[0, :, k, cols] = 0.0
            with torch.no_grad():
                r = float(torch.sigmoid(self._model(torch.as_tensor(ablated, dtype=torch.float32))
                                        ["risk_logit"][0, k]))
            drops[group] = round(base_risk - r, 4)

        # 物理残差（观测侧的诊断，说明当前物理约束离"闭合"有多远）
        adj = torch.as_tensor(self._data["adj"], dtype=torch.float32)
        elev = torch.as_tensor(self._data["elev"], dtype=torch.float32)
        t_now = torch.as_tensor(self._data["t_now"][i:i + 1], dtype=torch.float32)
        lapse = float(self._MDeep.lapse_rate_loss(t_now, adj, elev))

        total = sum(abs(v) for v in drops.values()) or 1.0
        return {
            "region_id": region_id,
            "region_name": self._data["region_names"].get(region_id, region_id),
            "month": self._months[i],
            "risk_score": round(base_risk, 4),
            "risk_level": _level(base_risk),
            "channel_attribution": [
                {"channel": g, "risk_drop": d, "share_pct": round(abs(d) / total * 100, 1)}
                for g, d in sorted(drops.items(), key=lambda x: -abs(x[1]))
            ],
            "physics_residual": {
                "lapse_rate_residual_c2": round(lapse, 4),
                "lapse_rate_ref_c_per_km": self._MDeep.LAPSE_RATE_C_PER_KM,
            },
            "method": "通道遮挡：把该通道在整个回看窗口上置零，测风险下降幅度。"
                      "语义是「模型对该通道的依赖度」，不是因果贡献。",
        }

    def fields(self, month: str | None = None) -> dict:
        """返回该月 26 县的气象场原始值，供前端画色斑图 / 等温线 / 风场。

        只输出**观测派生量**，不做任何再加工，单位与 `meteo_dataset` 一致
        （温度 ℃、风速 m/s、雪深 cm、降水 mm、辐射 MJ/m²）。
        """
        self._ensure_loaded()
        i = self._month_index(month)
        d = self._data
        graph = json.loads((STORE / "region_graph.json").read_text(encoding="utf-8"))
        regs = {r["region_id"]: r for r in graph["regions"]}
        idx = {n: k for k, n in enumerate(self._meta["monthly_features"])}
        meteo = d["meteo"][i]                                   # (N, F)
        wdir = meteo[:, [idx["wind_dir_sin"], idx["wind_dir_cos"]]]
        speed = meteo[:, idx["wind_mean"]]

        rows = []
        for k, rid in enumerate(d["region_ids"]):
            r = regs.get(rid, {})
            # 风向由 sin/cos 还原成角度，再由风速给出分量，便于前端画风羽/粒子
            ang = float(np.degrees(np.arctan2(wdir[k, 0], wdir[k, 1])) % 360)
            rows.append({
                "region_id": rid,
                "region_name": d["region_names"].get(rid, rid),
                "longitude": r.get("longitude"),
                "latitude": r.get("latitude"),
                "altitude": r.get("altitude"),
                "temperature_c": round(float(meteo[k, idx["t_mean"]]), 2),
                "temperature_min_c": round(float(meteo[k, idx["t_min"]]), 2),
                "precipitation_mm": round(float(meteo[k, idx["p_sum"]]), 1),
                "snowfall_cm": round(float(meteo[k, idx["snowfall_sum"]]), 1),
                "snow_depth_cm": round(float(meteo[k, idx["snow_depth_max"]]), 1),
                "radiation_mj": round(float(meteo[k, idx["rad_sum"]]), 1),
                "wind_speed_ms": round(float(speed[k]), 2),
                "wind_dir_deg": round(ang, 1),
                "wind_u": round(float(speed[k] * np.cos(np.radians(ang))), 2),
                "wind_v": round(float(speed[k] * np.sin(np.radians(ang))), 2),
                "humidity_pct": round(float(meteo[k, idx["rh_mean"]]), 0),
            })
        return {
            "month": self._months[i],
            "units": {"temperature_c": "℃", "precipitation_mm": "mm", "snowfall_cm": "cm",
                      "snow_depth_cm": "cm", "radiation_mj": "MJ/m²", "wind_speed_ms": "m/s",
                      "wind_dir_deg": "度（气象风向，0°=北）", "humidity_pct": "%"},
            "n_regions": len(rows),
            "regions": rows,
            "note": "26 个县为点位数据；前端如需面状色斑，由前端按 IDW 插值生成，本接口不做插值。",
        }

    # ---------------------------------------------------------------- 状态

    def status(self) -> dict:
        self._ensure_loaded()
        return {
            "ready": self._loaded,
            "trained": self._trained,
            "checkpoint": CKPT.name if CKPT.exists() else None,
            "train_info": getattr(self, "_ckpt_info", {}),
            "n_months": len(self._months),
            "month_range": [self._months[0], self._months[-1]] if self._months else [],
            "n_regions": len(self._data.get("region_ids", [])),
            "lookback": self._meta.get("lookback"),
            "n_meteo_features": self._meta.get("n_meteo_features"),
            "n_remote_features": self._meta.get("n_remote_features"),
            "feature_groups": {k: len(v) for k, v in FEATURE_GROUPS.items()},
            "device": "cpu",
            "caveats": [
                "本机无 GPU（torch CPU 版），模型规模刻意取小；深度用在结构（跨模态/空间/物理约束），不是规模。",
                "训练标签为规则核验过的公开事件，其余月份未确认。",
            ],
        }


def _level(p: float) -> str:
    if p >= 0.7:
        return "高"
    if p >= 0.45:
        return "中"
    if p >= 0.25:
        return "低"
    return "正常"


_SERVICE: DeepService | None = None


def get_service() -> DeepService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = DeepService()
    return _SERVICE


def reset_service() -> DeepService:
    global _SERVICE
    _SERVICE = DeepService()
    return _SERVICE


if __name__ == "__main__":
    svc = get_service()
    print("训练中（80 轮，CPU）…")
    t0 = time.time()
    st = svc.train()
    print(f"完成，用时 {time.time() - t0:.1f}s")
    print(json.dumps(st, ensure_ascii=False, indent=1))
    print("\n最新月预测（前 5）：")
    print(json.dumps(svc.predict()["regions"][:5], ensure_ascii=False, indent=1))
    print("\n归因示例：")
    print(json.dumps(svc.attribution("naqu-bange"), ensure_ascii=False, indent=1))
