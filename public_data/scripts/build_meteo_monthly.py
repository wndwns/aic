"""构建 AIC 版本的两份基础数据产物。

1. `backend/data_store/meteo_monthly_features.json` —— ERA5 逐日聚合的真月度气象特征
   （由 `backend/meteo_features.py` 生成，本脚本只是触发与校验）
2. `backend/data_store/meteo_event_labels.json` —— 标签网格 + **事件分类学标注**

## 标签清洗要做的事（与《可行性核查》对应）

原 `real_labels_1500.json` 的 128 条正样本里，有 32 条**不是气象事件**
（保险覆盖不足 17、地震 7、草场压力 4、牲畜疫病 2、疫苗接种 1）。
把它们当作「高原气象灾害预测」的正样本，既让指标虚低，也经不起气象方向的评委追问。
本脚本给每条记录加上 `is_meteo` 与 `event_family`，**不删除任何原始记录**，
让下游按需选择口径（清洗是「标注」而不是「抹掉」）。

同时显式区分两类「非事件月」：
- `evidence = "source_url" / "yearbook_page"`：有公开凭证的事件月（正样本）
- `evidence = "unlabeled"`：**未确认**月，原始数据里记为 `label=0`，
  但项目自身的口径是「不等同无灾」。本脚本额外写入 `confirmed_no_event=False`，
  禁止把它当作「已确认无灾」使用。

用法：
    python public_data/scripts/build_meteo_monthly.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import meteo_features as MF  # noqa: E402

RAW_LABELS = BACKEND / "data_store" / "real_labels_1500.json"
LABEL_OUT = BACKEND / "data_store" / "meteo_event_labels.json"


def build_labels() -> tuple[list[dict], dict]:
    raw = json.loads(RAW_LABELS.read_text(encoding="utf-8"))
    out: list[dict] = []
    for r in raw:
        event_type = r.get("event_type") or "none"
        is_pos = r.get("label") == 1

        if not is_pos:
            family = "no_event_recorded"
            is_meteo: bool | None = None
        elif event_type in MF.METEO_EVENT_TYPES:
            family, is_meteo = "meteo_hazard", True
        elif event_type in MF.NON_METEO_EVENT_TYPES:
            family, is_meteo = "non_meteo", False
        else:
            family, is_meteo = "unclassified", None

        if r.get("source_url"):
            evidence = "source_url"
        elif r.get("source"):
            evidence = "yearbook_page"
        else:
            evidence = "unlabeled"

        out.append({
            "region_id": r["region_id"],
            "month": r["month"],
            "label": 1 if is_pos else 0,
            "is_meteo": is_meteo,
            "event_family": family,
            "event_type": event_type if is_pos else None,
            "severity": r.get("severity") if is_pos else None,
            "loss_amount": r.get("loss_amount") if is_pos else None,
            "source": r.get("source"),
            "source_url": r.get("source_url"),
            "note": r.get("note"),
            "evidence": evidence,
            "confirmed_no_event": False,
        })

    families = Counter(x["event_family"] for x in out)
    types = Counter(x["event_type"] for x in out if x["label"] == 1)
    meta = {
        "raw_file": RAW_LABELS.name,
        "n_records": len(out),
        "n_positive_all": families["meteo_hazard"] + families["non_meteo"] + families["unclassified"],
        "n_positive_meteo": families["meteo_hazard"],
        "n_negative_unlabeled": families["no_event_recorded"],
        "event_family_counts": dict(families),
        "positive_event_type_counts": dict(types.most_common()),
        "non_meteo_rule": MF.NON_METEO_EVENT_TYPES,
        "caveats": [
            "128 条正样本均带公开凭证；其余 1372 个月为「未确认」，不等同无灾。",
            "未确认月的 confirmed_no_event 一律为 False，禁止当作已确认无灾参与评价。",
            "清洗是「加标注」，原始记录一条未删；下游可选择 all / meteo 两种标签口径。",
        ],
    }
    return out, meta


def main() -> None:
    feat_path = MF.save()
    feat_records, feat_meta = MF.build_all()
    print(f"[1/2] 月度气象特征 → {feat_path}")
    print(f"      {feat_meta['n_records']} 条 = {feat_meta['regions']} 县 × {feat_meta['months']} 月"
          f"（{feat_meta['month_range'][0]} ~ {feat_meta['month_range'][1]}）"
          f"，{feat_meta['n_features']} 维，剔除缺测日 {feat_meta['dropped_incomplete_days']} 条")

    labels, label_meta = build_labels()
    LABEL_OUT.write_text(
        json.dumps({"meta": label_meta, "records": labels}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"[2/2] 标签网格 → {LABEL_OUT}")
    print(f"      正样本 {label_meta['n_positive_all']} 条"
          f"（其中气象类 {label_meta['n_positive_meteo']} 条，"
          f"非气象 {label_meta['event_family_counts'].get('non_meteo', 0)} 条）"
          f"；未确认月 {label_meta['n_negative_unlabeled']} 条")
    print("      气象类构成：", dict(
        (k, v) for k, v in label_meta["positive_event_type_counts"].items()
        if k in MF.METEO_EVENT_TYPES
    ))
    print("      被剔除的非气象类：", dict(
        (k, v) for k, v in label_meta["positive_event_type_counts"].items()
        if k in MF.NON_METEO_EVENT_TYPES
    ))

    keys_f = {(r["region_id"], r["month"]) for r in feat_records}
    keys_l = {(r["region_id"], r["month"]) for r in labels}
    print(f"      特征键 {len(keys_f)}，标签键 {len(keys_l)}，可对齐 {len(keys_f & keys_l)}，"
          f"仅标签有 {len(keys_l - keys_f)}（{sorted(keys_l - keys_f)[:3]}）")


if __name__ == "__main__":
    main()
