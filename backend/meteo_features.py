"""高原牧区**月度气象特征**（ERA5 逐日 → 月度聚合）。

## 为什么需要这个模块

现有特征表 `backend/data_store/weather_data.json` 是「**每月 1 条**」的记录
（`observed_at` 恒为每月 01 日，`precipitation_mm_24h` 是**那一天的**降水量）。
用它去解释「整月发生的灾害」在物理上不可能成立——实例：班戈 2020-08 在事件标签里是
「强降雨冰雹、损失 52 只羊」，而该月气象记录的降水只有 0.06 mm。

本模块改用 `climate_era5.json` 的**逐日**序列重新聚合，得到真正的月度统计量。

## 数据事实边界

- 源：`backend/data_store/climate_era5.json`（ERA5 再分析）
- 覆盖：26 个县，逐日 2020-01-01 起（每县约 2192 天）
- **只含两列**：`temp_mean`（℃）、`precip_mm`（mm/日）
- 源数据存在缺测（`temp_mean` 或 `precip_mm` 为 null）。本模块**一律剔除、不插补**，
  并逐县记录剔除条数，供上游如实披露。

## 不具备数据条件、因此**未实现**的指标

| 想做 | 缺什么数据 |
|---|---|
| 积雪消融能量平衡约束 | 地表辐射、地表反照率、日照时数 |
| 风场 / 风寒指数 | 风速 |
| 相对湿度 / 露点 / 感热通量 | 湿度、气压 |
| 地形抬升与垂直递减率的**网格**约束 | DEM 栅格（`public_data/raw/dem/` 为空）；现仅有 26 个县的点位高程 |

## 气候态与信息泄漏

`t_anom` / `p_anom_z` 用**扩张窗口**计算：某年某月的距平只使用**该年之前**的同年月样本。
首次出现的年份（无往年样本）取 0，并在 `clima_ready` 中标记 `False`，
避免把未来信息带进历史样本。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
CLIMATE_FILE = BASE / "data_store" / "climate_era5.json"
OUTPUT_FILE = BASE / "data_store" / "meteo_monthly_features.json"

# --- 判据常量（改这里就能复现不同口径，不要在函数体里写魔法数）---
COLD_DAY_C = -10.0        # 日最低/日均温低于此值记为严寒日
SEVERE_COLD_DAY_C = -20.0  # 极寒日
RAIN_DAY_MM = 1.0          # 日降水超过此值记为降水日
DRY_DAY_MM = 0.5           # 日降水低于此值记为无水日
SNOW_TEMP_C = 1.0          # 判定「降雪日」的温度上限（与降水阈值联用）

FEATURE_NAMES: list[str] = [
    "t_mean", "t_min", "t_max", "t_range",
    "cold_days", "severe_cold_days",
    "p_sum", "p_max", "rain_days", "snow_days", "max_dry_run",
    "t_anom", "p_anom_z", "d_t",
]

# --- 事件分类学：哪些算「气象灾害」 ---
# 依据：与气温 / 降水 / 风等气象要素存在可追溯的致灾关系。
METEO_EVENT_TYPES: frozenset[str] = frozenset({
    "snowstorm", "rainstorm", "rain", "hail_storm", "wind",
    "cold_wave", "drought", "lightning",
    "landslide",  # 滑坡：以降水为诱发因子的地质灾害，归入气象诱发
})

# 非气象事件：出现在事件表里，但不应作为「气象灾害」正样本。
NON_METEO_EVENT_TYPES: dict[str, str] = {
    "insurance_coverage": "保险覆盖不足——业务指标，且同时是模型输入特征，用作标签会造成泄漏",
    "earthquake": "地震——构造活动，非气象致灾",
    "grassland_pressure": "草场压力——承载与经营问题，非气象事件",
    "disease": "牲畜疫病——生物致灾",
    "vaccination": "疫苗接种——防疫工作记录",
    "comprehensive": "综合类——无单一气象归因",
}


# ---------------------------------------------------------------------------
# 取数
# ---------------------------------------------------------------------------

def load_daily() -> tuple[dict[str, list[tuple[str, float, float]]], dict[str, int]]:
    """读逐日序列。返回 (县 → [(日期, 均温, 降水)]，县 → 缺测剔除条数)。"""
    if not CLIMATE_FILE.exists():
        raise FileNotFoundError(f"缺少气候数据：{CLIMATE_FILE}")
    raw: dict[str, list[dict[str, Any]]] = json.loads(CLIMATE_FILE.read_text(encoding="utf-8"))

    daily: dict[str, list[tuple[str, float, float]]] = {}
    dropped: dict[str, int] = {}
    for region_id, rows in raw.items():
        seq: list[tuple[str, float, float]] = []
        miss = 0
        for r in rows:
            t, p = r.get("temp_mean"), r.get("precip_mm")
            if t is None or p is None:
                miss += 1
                continue
            seq.append((str(r["date"]), float(t), float(p)))
        daily[region_id] = sorted(seq)
        dropped[region_id] = miss
    return daily, dropped


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------

def _dry_run_max(precips: list[float]) -> int:
    """最长连续无水日数。"""
    best = run = 0
    for p in precips:
        run = run + 1 if p < DRY_DAY_MM else 0
        if run > best:
            best = run
    return best


def _monthly_buckets(
    seq: list[tuple[str, float, float]],
) -> dict[str, list[tuple[str, float, float]]]:
    buckets: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for date, t, p in seq:
        buckets[date[:7]].append((date, t, p))
    return dict(buckets)


def monthly_features(
    daily: dict[str, list[tuple[str, float, float]]],
) -> list[dict[str, Any]]:
    """逐县逐月聚合。气候态走扩张窗口（只用往年）。"""
    records: list[dict[str, Any]] = []

    for region_id, seq in daily.items():
        buckets = _monthly_buckets(seq)
        if not buckets:
            continue

        # 该县各「月号」的历史值，(年, 月均温, 月降水) 升序
        history: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
        for ym in sorted(buckets):
            rows = buckets[ym]
            history[ym[5:7]].append(
                (int(ym[:4]), sum(r[1] for r in rows) / len(rows), sum(r[2] for r in rows))
            )
        # 扩张窗口前缀和，避免每次重算；也确保只用往年
        for month_no in history:
            history[month_no].sort()

        prev_t_mean: float | None = None
        for ym in sorted(buckets):
            rows = buckets[ym]
            temps = [r[1] for r in rows]
            precips = [r[2] for r in rows]
            year, month_no = int(ym[:4]), ym[5:7]

            t_mean = sum(temps) / len(temps)
            prior = [h for h in history[month_no] if h[0] < year]
            if len(prior) >= 1:
                clim_t = sum(h[1] for h in prior) / len(prior)
                clim_p = sum(h[2] for h in prior) / len(prior)
                if len(prior) >= 2:
                    var = sum((h[2] - clim_p) ** 2 for h in prior) / (len(prior) - 1)
                    clim_std = var ** 0.5
                else:
                    clim_std = 0.0
                t_anom = t_mean - clim_t
                p_anom_z = (sum(precips) - clim_p) / clim_std if clim_std > 1e-9 else 0.0
                clima_ready = True
            else:
                t_anom = p_anom_z = 0.0
                clima_ready = False

            records.append({
                "region_id": region_id,
                "month": ym,
                "t_mean": round(t_mean, 2),
                "t_min": round(min(temps), 2),
                "t_max": round(max(temps), 2),
                "t_range": round(max(temps) - min(temps), 2),
                "cold_days": sum(1 for t in temps if t < COLD_DAY_C),
                "severe_cold_days": sum(1 for t in temps if t < SEVERE_COLD_DAY_C),
                "p_sum": round(sum(precips), 2),
                "p_max": round(max(precips), 2),
                "rain_days": sum(1 for p in precips if p > RAIN_DAY_MM),
                "snow_days": sum(1 for t, p in zip(temps, precips) if t < SNOW_TEMP_C and p > DRY_DAY_MM),
                "max_dry_run": _dry_run_max(precips),
                "t_anom": round(t_anom, 2),
                "p_anom_z": round(p_anom_z, 3),
                "d_t": round(t_mean - prev_t_mean, 2) if prev_t_mean is not None else 0.0,
                # 元信息：供上游如实披露
                "days_in_month": len(rows),
                "clima_ready": clima_ready,
                "data_source": "era5",
                "is_derived": True,
                "derived_note": "由 ERA5 逐日 temp_mean/precip_mm 聚合；气候态取往年同月（扩张窗口）",
            })
            prev_t_mean = t_mean

    records.sort(key=lambda r: (r["region_id"], r["month"]))
    return records


def feature_matrix(records: list[dict[str, Any]]) -> list[list[float]]:
    return [[float(r[name]) for name in FEATURE_NAMES] for r in records]


def build_all() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    daily, dropped = load_daily()
    records = monthly_features(daily)
    regions = sorted({r["region_id"] for r in records})
    months = sorted({r["month"] for r in records})
    meta = {
        "source_file": CLIMATE_FILE.name,
        "source_kind": "ERA5 再分析（逐日）",
        "regions": len(regions),
        "region_ids": regions,
        "months": len(months),
        "month_range": [months[0], months[-1]] if months else [],
        "n_records": len(records),
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "dropped_incomplete_days": sum(dropped.values()),
        "dropped_by_region": {k: v for k, v in dropped.items() if v},
        "clima_ready_records": sum(1 for r in records if r["clima_ready"]),
        "caveats": [
            "源数据仅含温度与降水，无法派生积雪消融能量平衡、风场、湿度类指标。",
            "缺测日一律剔除、未插补；剔除条数见 dropped_incomplete_days。",
            "气候态为往年同月扩张窗口，首年样本的距平记为 0（clima_ready=False）。",
        ],
    }
    return records, meta


def save(path: Path | None = None) -> Path:
    records, meta = build_all()
    target = path or OUTPUT_FILE
    target.write_text(
        json.dumps({"meta": meta, "records": records}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return target


def get_status() -> dict[str, Any]:
    """供 API 层查询：本模块能提供什么、不能提供什么。"""
    return {
        "module": "meteo_features",
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "source": CLIMATE_FILE.name,
        "meteo_event_types": sorted(METEO_EVENT_TYPES),
        "non_meteo_event_types": NON_METEO_EVENT_TYPES,
        "unavailable": {
            "snow_energy_balance": "缺地表辐射 / 反照率 / 日照时数",
            "wind_and_windchill": "缺风速",
            "terrain_lapse_rate_grid": "缺 DEM 栅格（仅有点位高程）",
        },
    }


if __name__ == "__main__":
    out = save()
    recs, info = build_all()
    print(f"已写出 {out}")
    print(f"  {info['n_records']} 条 = {info['regions']} 县 × {info['months']} 月 "
          f"（{info['month_range'][0]} ~ {info['month_range'][1]}）")
    print(f"  {info['n_features']} 维特征：{', '.join(info['feature_names'])}")
    print(f"  剔除缺测日：{info['dropped_incomplete_days']} 条")
    print(f"  气候态可用样本：{info['clima_ready_records']}/{info['n_records']}")
