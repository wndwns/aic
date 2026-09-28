"""为 26 个县补齐 ERA5 多变量逐日序列（Open-Meteo 档案接口）。

## 为什么要补

现有 `backend/data_store/climate_era5.json` **只含两列**（`temp_mean`、`precip_mm`），
导致下面这些计划里的模块没有数据支撑：

| 想做 | 原来缺什么 |
|---|---|
| PINN 的积雪消融能量平衡 `L_snow_energy` | 地表短波辐射 |
| 前端风场粒子流 | 风速 + 风向 |
| 真实雪深（雪灾是最大灾种 41/96） | 雪深 |
| 日较差 / 寒潮过程特征 | 日最高、日最低温 |

本脚本一次把这些变量全补上。

## 两个硬性纪律

1. **单位一律取 API 自报的 `daily_units`，不猜**。写进输出的 `meta.daily_units`，
   并额外给出 `unit_notes` 说明转成项目既有单位要怎么换算。
   （项目以前在这上面栽过：把雪深单位当成厘米，实际是米。）
2. **同源校验**：把补下来的温度/降水与既有 `climate_era5.json` 逐日逐值比对。
   不一致说明两个数据集口径不同，拼在一起会引入偏差——那就必须停下来查。

用法：
    python public_data/scripts/fetch_era5_multivar.py                # 全量抓取
    python public_data/scripts/fetch_era5_multivar.py --probe        # 只试一年，验证接口与结构
    python public_data/scripts/fetch_era5_multivar.py 2020 2021      # 只抓指定年份
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
REGION_LIST = ROOT / "public_data" / "region_list.csv"
OLD_CLIMATE = BACKEND / "data_store" / "climate_era5.json"
OUT_FILE = BACKEND / "data_store" / "climate_era5_multivar.json"

ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "snowfall_sum",
    "snow_depth_max",
    "shortwave_radiation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "relative_humidity_2m_mean",
]

# 保留小数位（既压缩体积，也不虚增精度）
DECIMALS = {
    "temperature_2m_max": 1, "temperature_2m_min": 1, "temperature_2m_mean": 1,
    "precipitation_sum": 2, "snowfall_sum": 2, "snow_depth_max": 3,
    "shortwave_radiation_sum": 2, "wind_speed_10m_max": 1,
    "wind_direction_10m_dominant": 0, "relative_humidity_2m_mean": 0,
}

FIRST_YEAR = 2020
TZ = "Asia/Shanghai"
# 免费额度限制：一次请求「26 点位 × 366 天 × 10 变量」会直接 429，
# 所以按「年份 × 6 点位」分批，并在批间留间隔。
GROUP_SIZE = 6
SLEEP_BETWEEN = 10.0
PARTIAL = OUT_FILE.with_suffix(".partial.json")


# --------------------------------------------------------------------- 取数

def load_regions() -> list[dict]:
    with REGION_LIST.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        rid = (r.get("region_id") or "").strip()
        if not rid:
            continue
        out.append({
            "region_id": rid,
            "region_name": (r.get("region_name") or rid).strip(),
            "longitude": float(r["longitude"]),
            "latitude": float(r["latitude"]),
            "altitude_csv": float(r.get("altitude") or 0),
        })
    return out


def fetch(start: str, end: str, regions: list[dict]) -> list[dict]:
    params = {
        "latitude": ",".join(f"{r['latitude']:.4f}" for r in regions),
        "longitude": ",".join(f"{r['longitude']:.4f}" for r in regions),
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": TZ,
    }
    url = f"{ENDPOINT}?{urllib.parse.urlencode(params)}"
    last_err: Exception | None = None
    for attempt in range(1, 5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "aic-meteo-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if isinstance(payload, dict):
                payload = [payload]
            if len(payload) != len(regions):
                raise ValueError(f"返回 {len(payload)} 个点位，期望 {len(regions)}")
            return payload
        except urllib.error.HTTPError as exc:
            last_err = exc
            # 429 是限流，退避要够长；其它 HTTP 错误短退避即可
            wait = (30 * attempt if exc.code == 429 else 5 * attempt)
            print(f"    HTTP {exc.code}，{wait}s 后重试（第 {attempt}/4 次）", file=sys.stderr)
            time.sleep(wait)
        except Exception as exc:                      # 网络抖动、超时
            last_err = exc
            wait = 5 * attempt
            print(f"    {type(exc).__name__}: {exc}，{wait}s 后重试", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"{start}~{end} 抓取失败：{last_err}")


# ---------------------------------------------------------------- 同源校验

def consistency_check(regions: list[dict], dates: list[str],
                      series: dict[str, dict[str, list]]) -> dict:
    """与既有 climate_era5.json 逐日逐值比对温度与降水。

    分三档判定，因为「不同数据源」和「舍入差」必须区分开：
      - exact      ：完全一致
      - rounding   ：差 ≤ 0.1（温度）/ ≤ 0.2（降水），属小数位舍入
      - systematic ：超过上面阈值，说明**不是同一个数据源**，必须查明

    实测（2020-2024 / 26 县）：96.5% 完全一致；超过阈值的**全部**落在 `linzhi-bayi`
    一个县，而该县不在 1500 条标签的 25 县之内（项目文档亦记载林芝数据多处为规则推导）。
    """
    if not OLD_CLIMATE.exists():
        return {"checked": False, "reason": "既有气候文件不存在"}
    old = json.loads(OLD_CLIMATE.read_text(encoding="utf-8"))
    idx = {d: i for i, d in enumerate(dates)}

    tol = {"temperature_2m_mean": 0.1, "precipitation_sum": 0.2}
    stats = {k: {"exact": 0, "rounding": 0, "systematic": 0} for k in tol}
    by_region: dict[str, int] = {}
    worst: list[str] = []

    for r in regions:
        rid = r["region_id"]
        for rec in old.get(rid, []):
            i = idx.get(str(rec["date"]))
            if i is None:
                continue
            for new_key, old_key in (("temperature_2m_mean", "temp_mean"),
                                     ("precipitation_sum", "precip_mm")):
                new_v, old_v = series[rid][new_key][i], rec.get(old_key)
                if new_v is None or old_v is None:
                    continue
                d = abs(float(new_v) - float(old_v))
                if d == 0:
                    stats[new_key]["exact"] += 1
                elif d <= tol[new_key]:
                    stats[new_key]["rounding"] += 1
                else:
                    stats[new_key]["systematic"] += 1
                    by_region[rid] = by_region.get(rid, 0) + 1
                    if len(worst) < 5:
                        worst.append(f"{rid} {rec['date']} {new_key}: 新 {new_v} vs 旧 {old_v}")

    systematic = sum(v["systematic"] for v in stats.values())
    total = sum(v["exact"] + v["rounding"] + v["systematic"] for v in stats.values())
    if systematic == 0:
        conclusion = "同源：无超阈值差异"
    else:
        conclusion = (f"部分不同源：{systematic} 处超阈值，集中在 {list(by_region)}；"
                      f"该口径差异在使用时需排除或在报告中单独说明")
    return {
        "checked": True,
        "compared_values": total,
        "per_variable": stats,
        "systematic_mismatches": systematic,
        "systematic_by_region": by_region,
        "systematic_examples": worst,
        "conclusion": conclusion,
    }


# --------------------------------------------------------------------- 主流程

def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def recheck_existing() -> int:
    """对已生成的正式文件重算「同源校验」并回写 meta（口径复核可重复跑）。"""
    if not OUT_FILE.exists():
        print(f"{OUT_FILE} 不存在，无法复核", file=sys.stderr)
        return 1
    payload = json.loads(OUT_FILE.read_text(encoding="utf-8"))
    check = consistency_check(load_regions(), payload["dates"], payload["regions"])
    payload["meta"]["consistency_with_climate_era5"] = check
    payload["meta"]["region_climate_consistent"] = {
        rid: (check["systematic_by_region"].get(rid, 0) <= 10)
        for rid in payload["regions"]
    }
    payload["meta"]["elevation_note"] = (
        "API 返回的 elevation 是 ERA5 网格高程，与项目 CSV 的 altitude 最多相差约 380 m"
        "（yushu-chengduo 3822 vs 4200）。物理约束（L_lapse_rate）必须用 API 高程，"
        "因为温度来自同一网格；用 CSV 海拔会把递减率约束算歪。"
    )
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8")
    print(f"已复核 {OUT_FILE.name} → {check['conclusion']}")
    for var, s in check["per_variable"].items():
        print(f"  {var:24s} 完全一致 {s['exact']} / 舍入差 {s['rounding']} / 超阈值 {s['systematic']}")
    if check["systematic_by_region"]:
        print(f"  超阈值集中在：{check['systematic_by_region']}")
    return 0


def main(argv: list[str]) -> int:
    if "--recheck" in argv:
        return recheck_existing()
    probe = "--probe" in argv
    finalize_only = "--finalize" in argv          # 只把断点收尾成正式文件，不再请求接口
    years = [int(a) for a in argv[1:] if a.isdigit()] or None

    regions = load_regions()
    today = date.today()
    end_date = (today - timedelta(days=2)).isoformat()      # 档案接口有滞后
    if probe:
        years = [FIRST_YEAR]
    plan = years or list(range(FIRST_YEAR, today.year + 1))

    print(f"点位 {len(regions)} 个 / 变量 {len(DAILY_VARS)} 个 / 分 {GROUP_SIZE} 个一批 / "
          f"批间间隔 {SLEEP_BETWEEN:.0f}s（免费额度限流，一次全量会被 429）")
    print(f"截止 {end_date}；计划年份 {plan}")

    # ---- 断点续抓：partial 里记着已完成的年份 ----
    dates: list[str] = []
    series: dict[str, dict[str, list]] = {r["region_id"]: {} for r in regions}
    units: dict[str, str] = {}
    elevations: dict[str, float] = {}
    done_years: list[int] = []
    if PARTIAL.exists():
        try:
            saved = json.loads(PARTIAL.read_text(encoding="utf-8"))
            dates = saved["dates"]
            units = saved["units"]
            elevations = saved["elevations"]
            done_years = saved["done_years"]
            for rid, cols in saved["regions"].items():
                if rid in series:
                    series[rid] = {k: list(v) for k, v in cols.items()}
            print(f"  发现断点：已完成 {done_years}，共 {len(dates)} 天，继续。")
        except Exception as exc:
            print(f"  断点文件不可用（{exc}），从头开始。", file=sys.stderr)
            dates, done_years = [], []

    for year in plan:
        if finalize_only or year in done_years:
            if finalize_only:
                continue
            print(f"  {year} 已完成，跳过")
            continue
        start = f"{year}-01-01"
        end = end_date if year == today.year else f"{year}-12-31"
        if start > end_date:
            continue
        print(f"  抓取 {year}（{start} ~ {end}）…")
        year_dates: list[str] | None = None
        for gi, group in enumerate(_chunks(regions, GROUP_SIZE), start=1):
            payload = fetch(start, end, group)
            chunk_dates = list(payload[0]["daily"]["time"])
            if year_dates is None:
                year_dates = chunk_dates
                if dates and chunk_dates[0] != dates[-1]:
                    dates.extend(chunk_dates)
                elif not dates:
                    dates.extend(chunk_dates)
            units.update({k: str(v) for k, v in payload[0].get("daily_units", {}).items()})
            for region, item in zip(group, payload):
                rid = region["region_id"]
                elevations[rid] = item.get("elevation")
                for var in DAILY_VARS:
                    vals = item["daily"].get(var)
                    if vals is None:
                        continue
                    dp = DECIMALS.get(var, 2)
                    series[rid].setdefault(var, []).extend(
                        [None if v is None else round(float(v), dp) for v in vals]
                    )
            print(f"    批 {gi}/{-(-len(regions) // GROUP_SIZE)} 完成（{len(group)} 个点位）")
            time.sleep(SLEEP_BETWEEN)
        done_years.append(year)
        _save_partial(dates, series, units, elevations, done_years)
        print(f"    {year} 完成，已存断点。")

    if not dates:
        print("没有抓到任何数据", file=sys.stderr)
        return 1

    missing = [y for y in plan if y not in done_years]
    if missing:
        print(f"\n⚠️ 尚未抓到的年份：{missing}（接口限流）。"
              f"已抓部分仍可正常使用；补抓时直接重跑本脚本，会从断点续。")

    check = consistency_check(regions, dates, series)
    print(f"\n同源校验：比对 {check.get('compared_values')} 个值 → {check.get('conclusion')}")
    for var, s in (check.get("per_variable") or {}).items():
        print(f"  {var:24s} 完全一致 {s['exact']} / 舍入差 {s['rounding']} / 超阈值 {s['systematic']}")
    for line in check.get("systematic_examples", []):
        print("   ", line)

    meta = _build_meta(regions, dates, units, elevations, check)
    if probe:
        print("\n--probe 模式：只抓了首年，不写正式文件。")
        print(json.dumps({"variables": DAILY_VARS, "n_days": len(dates), "units": units},
                         ensure_ascii=False, indent=1))
        return 0

    payload_out = {
        "meta": meta,
        "dates": dates,
        "regions": {r["region_id"]: series[r["region_id"]] for r in regions},
    }
    tmp = OUT_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload_out, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    tmp.replace(OUT_FILE)
    PARTIAL.unlink(missing_ok=True)
    print(f"\n已写出 {OUT_FILE}  （{OUT_FILE.stat().st_size / 1048576:.1f} MB，"
          f"{len(dates)} 天 × {len(regions)} 县 × {len(DAILY_VARS)} 变量）")
    return 0


def _save_partial(dates, series, units, elevations, done_years) -> None:
    PARTIAL.write_text(json.dumps({
        "dates": dates, "units": units, "elevations": elevations,
        "done_years": done_years, "regions": series,
    }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _build_meta(regions, dates, units, elevations, check) -> dict:
    return {
        "source": "Open-Meteo Archive API（ERA5 再分析）",
        "endpoint": ENDPOINT,
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": TZ,
        "date_range": [dates[0], dates[-1]],
        "n_days": len(dates),
        "region_count": len(regions),
        "variables": DAILY_VARS,
        "daily_units": units,
        # API 自报单位 → 项目既有单位的换算；换算一律写在这里，不散落在代码里
        "unit_notes": {
            "snow_depth_max": "API 单位是【米】(m)；项目既有 snow_depth_cm 是厘米，用时要 ×100",
            "snowfall_sum": "API 单位是【厘米】(cm)，本身就是厘米",
            "wind_speed_10m_max": "API 单位是【km/h】；项目既有 wind_speed_mps 是 m/s，用时要 ÷3.6",
            "shortwave_radiation_sum": "单位 MJ/m²（日累计），积雪消融能量平衡的辐射项",
            "wind_direction_10m_dominant": "单位度（气象风向，0°=北）",
            "relative_humidity_2m_mean": "单位 %",
        },
        "elevations_api": elevations,
        "elevations_csv": {r["region_id"]: r["altitude_csv"] for r in regions},
        "elevation_note": (
            "API 返回的 elevation 是 ERA5 网格高程，与项目 CSV 里的 altitude 最多相差约 380 m"
            "（yushu-chengduo 3822 vs 4200）。**物理约束（L_lapse_rate）必须用 API 高程**，"
            "因为温度就来自同一网格；用 CSV 海拔会把递减率约束算歪。"
        ),
        # 逐县的气候口径一致性标记：False 的县说明与既有 climate_era5.json 系统性不同源
        "region_climate_consistent": {
            r["region_id"]: (check.get("systematic_by_region", {}).get(r["region_id"], 0) <= 10)
            for r in regions
        },
        "consistency_with_climate_era5": check,
        "caveats": [
            "本文件为多变量补齐版；既有 climate_era5.json 未改动，两者可并存。",
            "单位以 meta.daily_units 为唯一权威（API 自报），换算见 unit_notes。",
            "若 consistency 结论非「同源」，禁止与既有文件混合使用。",
            "免费额度有限，抓取按「年份 × 6 点位」分批并留间隔；中断后可从 partial 续抓。",
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
