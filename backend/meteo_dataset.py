"""多变量气象特征 → **双任务数据集**（同期识别 + 提前预测）。

## 特征设计原则

1. **只用观测派生量**，不用「高/中/低」这类规则等级 —— 实测规则等级接近噪音（AUC 0.42~0.55）。
2. **过程量优先于瞬时量**：雪灾不是「某一天的雪深」，而是「连续降雪 + 积雪累积」，
   所以要有连续降雪日数、积雪趋势、最长连续无雨日这类**过程特征**。
3. **物理量单独成列**，供 PINN 约束直接用：`melt_index`（正积温度日）、
   `energy_index`（短波辐射日累计）—— 它们是 `L_snow_energy` 的输入。

## ⚠️ 关于「积雪消融能量平衡」的诚实说明

完整能量平衡需要：净短波 + 净长波 + 感热 + 潜热 + 地热 + 相变潜热。本数据只有
**日累计短波辐射**，没有长波辐射、没有地表反照率、没有风速廓线。
因此本模块给出的是**简化代理**（`energy_index` 与 `melt_index`），
用它与雪深变化的一致性作为软约束，**不是**严格能量闭合。
方案与报告里必须写明这一点，不能声称做了完整能量平衡。

## 双任务定义

- 任务 A（同期识别，分类）：用 t 月及之前的序列 → 判 t 月是否有气象灾害
- 任务 B（提前预测，回归）：用 ≤t 月序列 → 预测 t+1 / t+2 / t+3 的灾情强度

任务 B 才是「预警」；也是时序模型相对树模型真正有优势的地方
（树模型没有序列记忆、不能外推时间趋势）。

用法：
    python backend/meteo_dataset.py            # 构建并落盘
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
STORE = BASE / "data_store"
MULTIVAR = STORE / "climate_era5_multivar.json"
OLD_CLIMATE = STORE / "climate_era5.json"
RS_FILE = STORE / "remote_sensing_data.json"
LABEL_FILE = STORE / "meteo_event_labels.json"
GRAPH_FILE = STORE / "region_graph.json"
OUT_NPZ = STORE / "meteo_dataset.npz"
OUT_META = STORE / "meteo_dataset_meta.json"

# --- 判据常量 ---
COLD_DAY_C = -10.0
SEVERE_COLD_DAY_C = -20.0
FROST_DAY_C = 0.0
RAIN_DAY_MM = 1.0
SNOW_TEMP_C = 1.0
WET_DAY_MM = 0.2
DRY_DAY_MM = 0.5
LAPSE_RATE_C_PER_KM = 6.5      # 高原自由大气垂直递减率（用于 L_lapse_rate）
LATENT_HEAT_MJ_PER_KG = 2.5    # 融化 1 kg 雪约需 0.334 MJ，此处用「度日→能量」经验系数
SEVERITY_SCALE = {"低": 1.0, "中": 2.0, "高": 3.0, "特重": 4.0, "特重度": 4.0, "严重": 3.0}

MONTHLY_FEATURES: list[str] = [
    # 温度
    "t_mean", "t_max", "t_min", "t_range", "t_min_abs",
    "cold_days", "severe_cold_days", "frost_days",
    "t_anom", "d_t",
    # 降水与积雪
    "p_sum", "p_max", "rain_days", "snowfall_sum", "snow_days",
    "max_dry_run", "snow_depth_max", "snow_depth_mean", "snow_depth_trend",
    "p_anom_z",
    # 辐射
    "rad_sum", "rad_per_day",
    # 风
    "wind_max", "wind_mean", "wind_dir_sin", "wind_dir_cos",
    # 湿度
    "rh_mean",
    # 物理量（供 PINN）
    "melt_index", "energy_index",
]

RS_FEATURES: list[str] = [
    "ndvi", "ndvi_change_pct", "vegetation_cover_pct", "snow_cover_pct",
    "degradation_level",
]


# --------------------------------------------------------------------- 工具

def _month(d: str) -> str:
    return d[:7]


def _stat(vals: list[float | None], how: str, default: float = 0.0) -> float:
    xs = [float(v) for v in vals if v is not None]
    if not xs:
        return default
    if how == "mean":
        return sum(xs) / len(xs)
    if how == "max":
        return max(xs)
    if how == "min":
        return min(xs)
    if how == "sum":
        return sum(xs)
    raise ValueError(how)


def _max_consecutive(flags: list[bool]) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


# --------------------------------------------------------------------- 取数

def load_multivar() -> tuple[list[str], dict[str, dict[str, list]]]:
    if not MULTIVAR.exists():
        raise FileNotFoundError(
            f"缺少多变量数据 {MULTIVAR}；请先跑 public_data/scripts/fetch_era5_multivar.py")
    payload = json.loads(MULTIVAR.read_text(encoding="utf-8"))
    return payload["dates"], payload["regions"]


def load_rs_monthly() -> dict[tuple[str, str], dict[str, float]]:
    """遥感月度聚合（有则用，无则空）。"""
    if not RS_FILE.exists():
        return {}
    rows = json.loads(RS_FILE.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("records", [])
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        rid = r.get("region_id")
        date = str(r.get("scene_date") or r.get("observed_at") or "")
        if not rid or len(date) < 7:
            continue
        buckets[(rid, date[:7])].append(r)

    def pct(v: Any, default: float) -> float:
        try:
            return float(str(v).replace("%", ""))
        except (TypeError, ValueError):
            return default

    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, recs in buckets.items():
        out[key] = {
            "ndvi": _stat([r.get("ndvi") for r in recs], "mean", 0.4),
            "ndvi_change_pct": _stat([pct(r.get("ndvi_change"), 0.0) for r in recs], "mean"),
            "vegetation_cover_pct": _stat([pct(r.get("vegetation_cover"), 50.0) for r in recs], "mean", 50.0),
            "snow_cover_pct": _stat([pct(r.get("snow_cover"), 10.0) for r in recs], "max", 10.0),
            "degradation_level": _stat(
                [{"基本稳定": 0, "轻度退化": 1, "中度退化": 2, "重度退化": 3}.get(
                    str(r.get("degradation_level", "轻度退化")), 1) for r in recs], "max", 1.0),
        }
    return out


def load_labels() -> dict[tuple[str, str], dict]:
    payload = json.loads(LABEL_FILE.read_text(encoding="utf-8"))
    return {(r["region_id"], r["month"]): r for r in payload["records"]}


# --------------------------------------------------------------------- 特征

def monthly_features(dates: list[str], regions: dict[str, dict[str, list]]
                     ) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for rid, cols in regions.items():
        idx_by_month: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(dates):
            idx_by_month[_month(d)].append(i)

        # 往年同月气候态（扩张窗口，防未来信息泄漏）
        clim: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
        for ym, idxs in sorted(idx_by_month.items()):
            clim[ym[5:7]].append((
                int(ym[:4]),
                _stat([(cols.get("temperature_2m_mean") or [None] * len(dates))[i] for i in idxs], "mean"),
                _stat([(cols.get("precipitation_sum") or [None] * len(dates))[i] for i in idxs], "sum"),
            ))

        prev_t: float | None = None
        prev_sd: float | None = None
        for ym in sorted(idx_by_month):
            idxs = idx_by_month[ym]
            g = lambda name: [(cols.get(name) or [None] * len(dates))[i] for i in idxs]

            tmax, tmin, tmean = g("temperature_2m_max"), g("temperature_2m_min"), g("temperature_2m_mean")
            prec, snowf = g("precipitation_sum"), g("snowfall_sum")
            sdep, rad = g("snow_depth_max"), g("shortwave_radiation_sum")
            wind, wdir, rh = g("wind_speed_10m_max"), g("wind_direction_10m_dominant"), g("relative_humidity_2m_mean")

            temps = [float(v) for v in tmean if v is not None]
            precs = [float(v) for v in prec if v is not None]
            snowds = [float(v) * 100 for v in sdep if v is not None]      # API 单位是【米】→ 厘米
            winds_kmh = [float(v) for v in wind if v is not None]
            dirs = [float(v) for v in wdir if v is not None]
            t_mean = sum(temps) / len(temps) if temps else 0.0

            year, month_no = int(ym[:4]), ym[5:7]
            prior = [h for h in sorted(clim[month_no]) if h[0] < year]
            if prior:
                clim_t = sum(h[1] for h in prior) / len(prior)
                clim_p = sum(h[2] for h in prior) / len(prior)
                clim_std = statistics.pstdev([h[2] for h in prior]) if len(prior) >= 2 else 0.0
                t_anom = t_mean - clim_t
                p_anom_z = (sum(precs) - clim_p) / clim_std if clim_std > 1e-9 else 0.0
                clima_ready = True
            else:
                t_anom = p_anom_z = 0.0
                clima_ready = False

            melt_index = sum(max(0.0, float(t)) for t in temps if t is not None)   # 度日
            rad_sum = _stat(rad, "sum")
            sd_mean = (sum(snowds) / len(snowds)) if snowds else 0.0
            sd_max = max(snowds) if snowds else 0.0

            records.append({
                "region_id": rid, "month": ym,
                # 温度
                "t_mean": round(t_mean, 2),
                "t_max": round(_stat(tmax, "max"), 2),
                "t_min": round(_stat(tmin, "min"), 2),
                "t_range": round(_stat(tmax, "max") - _stat(tmin, "min"), 2),
                "t_min_abs": round(_stat(tmin, "min"), 2),
                "cold_days": sum(1 for t in temps if t < COLD_DAY_C),
                "severe_cold_days": sum(1 for t in temps if t < SEVERE_COLD_DAY_C),
                "frost_days": sum(1 for t in temps if t < FROST_DAY_C),
                "t_anom": round(t_anom, 2),
                "d_t": round(t_mean - prev_t, 2) if prev_t is not None else 0.0,
                # 降水与积雪
                "p_sum": round(sum(precs), 2),
                "p_max": round(max(precs) if precs else 0.0, 2),
                "rain_days": sum(1 for p in precs if p > RAIN_DAY_MM),
                "snowfall_sum": round(_stat(snowf, "sum"), 2),
                "snow_days": sum(1 for t, p in zip(temps, precs)
                                 if t < SNOW_TEMP_C and p > WET_DAY_MM),
                "max_dry_run": _max_consecutive([p < DRY_DAY_MM for p in precs]),
                "snow_depth_max": round(sd_max, 1),
                "snow_depth_mean": round(sd_mean, 1),
                "snow_depth_trend": round(sd_mean - prev_sd, 1) if prev_sd is not None else 0.0,
                "p_anom_z": round(p_anom_z, 3),
                # 辐射
                "rad_sum": round(rad_sum, 2),
                "rad_per_day": round(rad_sum / max(1, len(idxs)), 3),
                # 风（km/h → m/s 在这里就换算掉，后面的模块不用再管单位）
                "wind_max": round(max(winds_kmh) / 3.6, 2) if winds_kmh else 0.0,
                "wind_mean": round(sum(winds_kmh) / len(winds_kmh) / 3.6, 2) if winds_kmh else 0.0,
                "wind_dir_sin": round(statistics.mean([math.sin(math.radians(d)) for d in dirs]), 4) if dirs else 0.0,
                "wind_dir_cos": round(statistics.mean([math.cos(math.radians(d)) for d in dirs]), 4) if dirs else 0.0,
                # 湿度
                "rh_mean": round(_stat(rh, "mean"), 1),
                # 物理量（供 PINN）
                "melt_index": round(melt_index, 1),
                "energy_index": round(rad_sum - LATENT_HEAT_MJ_PER_KG * melt_index * 0.1, 2),
                # 元信息
                "days_in_month": len(idxs),
                "clima_ready": clima_ready,
                "data_source": "era5_multivar",
                "is_derived": True,
                "derived_note": "由 ERA5 多变量逐日序列聚合；雪深已由米换算为厘米，风速已由 km/h 换算为 m/s",
            })
            prev_t, prev_sd = t_mean, sd_mean

    records.sort(key=lambda r: (r["region_id"], r["month"]))
    return records


# --------------------------------------------------------------- 数据集装配

def build_dataset(lookback: int = 6, horizons: tuple[int, ...] = (1, 2, 3)) -> tuple[Any, dict]:
    import numpy as np

    dates, regions = load_multivar()
    mv_meta = json.loads(MULTIVAR.read_text(encoding="utf-8"))["meta"]
    records = monthly_features(dates, regions)
    labels = load_labels()
    rs = load_rs_monthly()
    graph = json.loads(GRAPH_FILE.read_text(encoding="utf-8"))

    region_ids: list[str] = graph["region_ids"]
    r_index = {rid: i for i, rid in enumerate(region_ids)}
    months = sorted({r["month"] for r in records})
    m_index = {m: i for i, m in enumerate(months)}

    # 高程取 **ERA5 网格高程**（API 自报），不取 CSV 海拔：
    # 温度就来自同一网格，两者搭配才自洽；CSV 海拔最多差 ~380 m，会把递减率约束算歪。
    elev_api = mv_meta.get("elevations_api") or {}
    elevation = np.asarray(
        [float(elev_api.get(rid) or next(
            (r["altitude"] for r in graph["regions"] if r["region_id"] == rid), 0.0))
         for rid in region_ids], dtype=np.float32)

    by_key = {(r["region_id"], r["month"]): r for r in records}
    F, R = len(MONTHLY_FEATURES), len(RS_FEATURES)
    T, N = len(months), len(region_ids)

    # 张量：[时间, 县, 变量]
    meteo = np.zeros((T, N, F), dtype=np.float32)
    remote = np.zeros((T, N, R), dtype=np.float32)
    meteo_mask = np.zeros((T, N), dtype=np.float32)
    for (rid, m), rec in by_key.items():
        if rid not in r_index:
            continue
        meteo[m_index[m], r_index[rid]] = [float(rec[n]) for n in MONTHLY_FEATURES]
        meteo_mask[m_index[m], r_index[rid]] = 1.0
        if (rid, m) in rs:
            remote[m_index[m], r_index[rid]] = [float(rs[(rid, m)][n]) for n in RS_FEATURES]

    # 标签张量
    cls = np.zeros((T, N), dtype=np.float32)
    confirmed = np.zeros((T, N), dtype=np.float32)     # 1 = 明确有灾
    unlabeled = np.ones((T, N), dtype=np.float32)      # 1 = 该月未确认（不等于无灾）
    # 灾情强度：0 无 / 1 低 / 2 中 / 3 高（任务 B 回归用的连续目标）
    severity = np.zeros((T, N), dtype=np.float32)
    for (rid, m), lb in labels.items():
        if rid not in r_index or m not in m_index:
            continue
        i, j = m_index[m], r_index[rid]
        is_pos = lb["label"] == 1
        cls[i, j] = 1.0 if is_pos and lb.get("is_meteo") else 0.0
        confirmed[i, j] = 1.0 if is_pos else 0.0
        unlabeled[i, j] = 0.0 if is_pos else 1.0
        if is_pos and lb.get("is_meteo"):
            severity[i, j] = SEVERITY_SCALE.get(str(lb.get("severity") or ""), 1.0)

    # --- 任务 A：t 时刻用 [t-L+1 .. t] 判 t 月有无气象灾害（分类）-----------
    # --- 任务 B：t 时刻用 [t-L+1 .. t] 预测 t+1..t+h 的**灾情强度**（回归）---
    #     回归目标取「未来 h 个月内的最大灾情强度」，是连续量，R² 才有业务含义；
    #     同时保留一份二值版（h 月内是否有灾）供 PR 曲线使用。
    horizon_reg = np.zeros((T, N, len(horizons)), dtype=np.float32)
    horizon_bin = np.zeros((T, N, len(horizons)), dtype=np.float32)
    for k, h in enumerate(horizons):
        for t in range(T - h):
            window = severity[t + 1:t + 1 + h, :]
            horizon_reg[t, :, k] = window.max(axis=0)
            horizon_bin[t, :, k] = (window.max(axis=0) > 0).astype(np.float32)

    def sequences(window: int) -> np.ndarray:
        """返回 (T, window, N, F)：每个样本是「某月」的全 26 县序列。

        形状约定与 `models_deep.TPSTNet` 对齐：(B, L, N, F)，
        即**时间维在前、县维在后**——这样模型把县折进 batch 做时序、再在县维上做图卷积。
        """
        out = np.zeros((T, window, N, F + R), dtype=np.float32)
        for t in range(T):
            lo = max(0, t - window + 1)
            block = np.concatenate([meteo[lo:t + 1], remote[lo:t + 1]], axis=-1)   # (L, N, F)
            if block.shape[0] < window:                     # 前置不足则沿时间轴左侧补齐
                pad = np.repeat(block[:1], window - block.shape[0], axis=0)
                block = np.concatenate([pad, block], axis=0)
            out[t] = block
        return out

    seq = sequences(lookback)

    # 有效月份：任务 A 所有月份可用；任务 B 需要 t+h 落在数据范围内
    valid_a = np.ones((T, N), dtype=np.float32)
    valid_b = np.zeros((T, N), dtype=np.float32)
    for h in horizons:
        valid_b[:T - h, :] += 1.0
    valid_b = (valid_b > 0).astype(np.float32)

    np.savez_compressed(
        OUT_NPZ,
        meteo=meteo, remote=remote, mask=meteo_mask,
        seq=seq, cls=cls, confirmed=confirmed, unlabeled=unlabeled,
        severity=severity, horizon_reg=horizon_reg, horizon_bin=horizon_bin,
        valid_a=valid_a, valid_b=valid_b,
        adjacency=np.asarray(graph["adjacency"], dtype=np.float32),
        adjacency_norm=np.asarray(graph["adjacency_normalized"], dtype=np.float32),
        distance_km=np.asarray(graph["distance_km"], dtype=np.float32),
        elevation=np.asarray(elevation, dtype=np.float32),
    )

    meta = {
        "built_from": MULTIVAR.name,
        "n_months": T, "n_regions": N,
        "month_range": [months[0], months[-1]],
        "monthly_features": MONTHLY_FEATURES,
        "remote_features": RS_FEATURES,
        "n_meteo_features": F, "n_remote_features": R,
        "lookback": lookback, "horizons": list(horizons),
        "seq_shape": list(seq.shape),
        "positive_cells": int(cls.sum()),
        "confirmed_cells": int(confirmed.sum()),
        "unlabeled_cells": int(unlabeled.sum()),
        "rs_months_covered": len(rs),
        "lapse_rate_c_per_km": LAPSE_RATE_C_PER_KM,
        "severity_scale": SEVERITY_SCALE,
        "caveats": [
            "负例＝「未确认」月，不等同确认无灾；unlabeled 张量单独保留该信息。",
            "积雪能量平衡为简化代理（只有日累计短波辐射，无长波/反照率/风廓线），不是严格闭合。",
            "任务 B 目标是「未来 h 个月内的最大灾情强度」（0~4 连续量），同时保留二值版 horizon_bin。",
            "灾情强度的数值刻度来自标签里的 severity 文字档（低/中/高），不是物理量纲。",
            "遥感特征仅覆盖 2020-2024，缺失月份为 0。",
        ],
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return np.load(OUT_NPZ, allow_pickle=False), meta


def main() -> None:
    data, meta = build_dataset()
    print(f"已写出 {OUT_NPZ}（{OUT_NPZ.stat().st_size / 1048576:.1f} MB）与 {OUT_META.name}")
    print(f"  网格：{meta['n_months']} 月 × {meta['n_regions']} 县 "
          f"（{meta['month_range'][0]} ~ {meta['month_range'][1]}）")
    print(f"  气象特征 {meta['n_meteo_features']} 维 / 遥感特征 {meta['n_remote_features']} 维 / "
          f"序列张量 {meta['seq_shape']}（回看 {meta['lookback']} 月）")
    print(f"  任务 A 正例 {int(data['cls'].sum())} 格；确认有灾 {int(data['confirmed'].sum())} 格")
    print(f"  遥感月度覆盖 {meta['rs_months_covered']} 个(县,月)组合")
    print(f"  空间图：邻接 {data['adjacency'].shape}，归一化邻接 {data['adjacency_norm'].shape}")
    print("\n  特征清单：")
    for i, n in enumerate(meta["monthly_features"], 1):
        print(f"    {i:2d}. {n}")


if __name__ == "__main__":
    main()
