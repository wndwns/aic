"""对照评估：**现状 16 维特征** vs **ERA5 月度气象特征**，同划分、同模型、同标签口径。

划分方式沿用项目原有回测（`public_data/scripts/backtest.py`）的滚动前向思路：
    训练 = 全部 ≤ 上一年 12 月 ；测试 = 指定年份全年
标签只用 `meteo_event_labels.json`，可选两种口径：
    all   = 全部 128 条正样本（现状口径）
    meteo = 仅气象类 96 条（清洗后口径）

指标一律同时给出：
    AUC            —— 排序能力，0.5 = 随机
    AP（PR-AUC）   —— 不平衡数据下比 AUC 更敏感
    最优阈值 P/R/F1 —— **乐观口径**：阈值在测试集上挑最好的，实际部署拿不到
    P@R≥0.9        —— 业务可解释口径：要求不漏掉九成灾害时的小精确率
    R@P≥0.30       —— 业务可解释口径：精确率压到三成时的召回

用法：
    python public_data/scripts/eval_meteo_vs_current.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import meteo_features as MF  # noqa: E402
import models as M  # noqa: E402

FEAT_FILE = BACKEND / "data_store" / "meteo_monthly_features.json"
LABEL_FILE = BACKEND / "data_store" / "meteo_event_labels.json"
REPORT_FILE = BACKEND / "data_store" / "meteo_eval_report.json"


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

def metrics(y: np.ndarray, score: np.ndarray) -> dict:
    out: dict = {"n": int(len(y)), "n_pos": int(y.sum())}
    if len(set(y)) < 2:
        return out
    out["auc"] = round(float(roc_auc_score(y, score)), 3)
    out["ap"] = round(float(average_precision_score(y, score)), 3)

    prec, rec, thr = precision_recall_curve(y, score)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    k = int(np.argmax(f1))
    out["best_f1"] = round(float(f1[k]), 3)
    out["best_f1_P"] = round(float(prec[k]), 3)
    out["best_f1_R"] = round(float(rec[k]), 3)
    out["best_f1_thr"] = round(float(thr[k]), 3) if k < len(thr) else None

    # 业务口径一：召回 ≥ 0.9 时能拿到多高的精确率
    ok = np.where(rec >= 0.9)[0]
    out["P_at_R90"] = round(float(prec[ok].max()), 3) if len(ok) else 0.0
    # 业务口径二：精确率 ≥ 0.30 时能不漏掉多少
    ok2 = np.where(prec >= 0.30)[0]
    out["R_at_P30"] = round(float(rec[ok2].max()), 3) if len(ok2) else 0.0
    return out


def make_models() -> dict:
    from xgboost import XGBClassifier
    return {
        "RandomForest": lambda n_pos, n_neg: RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=42),
        "XGBoost": lambda n_pos, n_neg: XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=max(1.0, n_neg / max(1, n_pos)),
            eval_metric="logloss", random_state=42),
    }


# ---------------------------------------------------------------------------
# 装配数据
# ---------------------------------------------------------------------------

def load_current() -> tuple[list[tuple[str, str]], np.ndarray]:
    """现状特征（models.py 的 16 维，含 3 个信贷字段）。"""
    X, _y, meta, _info = M._extract_monthly_samples()
    keys = [(m["region_id"], m["month"]) for m in meta]
    return keys, X


def load_meteo() -> tuple[list[tuple[str, str]], np.ndarray]:
    payload = json.loads(FEAT_FILE.read_text(encoding="utf-8"))
    recs = payload["records"]
    keys = [(r["region_id"], r["month"]) for r in recs]
    return keys, np.array(MF.feature_matrix(recs), dtype=float)


def load_labels() -> dict[tuple[str, str], dict]:
    payload = json.loads(LABEL_FILE.read_text(encoding="utf-8"))
    return {(r["region_id"], r["month"]): r for r in payload["records"]}


def align(keys: list[tuple[str, str]], X: np.ndarray,
          labels: dict, scope: str) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray]:
    ks, rows, ys = [], [], []
    for i, k in enumerate(keys):
        lb = labels.get(k)
        if lb is None:
            continue
        if lb["label"] == 1:
            if scope == "meteo" and not lb["is_meteo"]:
                continue
            y = 1
        else:
            y = 0  # 未确认月：按项目现状口径作为负样本参与评价（已在报告里标注）
        ks.append(k)
        rows.append(X[i])
        ys.append(y)
    return ks, np.array(rows, dtype=float), np.array(ys, dtype=int)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def evaluate(name: str, keys, X, labels, scope: str, test_year: str) -> dict:
    ks, Xa, ya = align(keys, X, labels, scope)
    months = np.array([k[1] for k in ks])
    tr = months < f"{test_year}-01"
    te = (months >= f"{test_year}-01") & (months <= f"{test_year}-12")
    if tr.sum() == 0 or te.sum() == 0 or len(set(ya[te])) < 2:
        return {"feature_set": name, "label_scope": scope, "test_year": test_year,
                "skipped": "训练或测试不足"}

    Xtr, ytr, Xte, yte = Xa[tr], ya[tr], Xa[te], ya[te]
    row = {"feature_set": name, "label_scope": scope, "test_year": test_year,
           "n_train": int(tr.sum()), "n_train_pos": int(ytr.sum()),
           "n_test": int(te.sum()), "n_test_pos": int(yte.sum()), "models": {}}
    for mname, factory in make_models().items():
        clf = factory(int(ytr.sum()), int((ytr == 0).sum()))
        clf.fit(Xtr, ytr)
        score = clf.predict_proba(Xte)[:, 1]
        row["models"][mname] = metrics(yte, score)
    row["models"]["随机基线"] = {"auc": 0.5, "ap": round(float(yte.mean()), 3)}
    return row


def build_feature_sets() -> list[tuple[str, list, np.ndarray]]:
    """所有待比较的特征集。特别注意最后两组的用途：

    - `现状7维(仅气象快照)` 与 `现状13维(去信贷)` 用来做**消融**：
      判断「现状 16 维」的表面区分力是不是来自那 3 个**县内恒定**的信贷字段。
      县内恒定字段 = 县的固定效应，会退化成「这个县报灾报得多不多」的地区代理，
      而不是任何可迁移的气象信号。
    - `ERA5+县独热` 用来量化「县身份」本身能贡献多少虚高 AUC。
    """
    F = list(M.FEATURE_NAMES)
    fin_idx = [F.index(n) for n in ("avg_score", "avg_insurance_coverage", "finance_risk_score")]
    wx_idx = list(range(7))

    cur_keys, cur_X = load_current()
    met_keys, met_X = load_meteo()

    regions = sorted({k[0] for k in met_keys})
    onehot = np.zeros((len(met_keys), len(regions)))
    for i, k in enumerate(met_keys):
        onehot[i, regions.index(k[0])] = 1.0

    keep13 = [i for i in range(len(F)) if i not in fin_idx]
    return [
        ("现状16维", cur_keys, cur_X),
        ("现状13维(去信贷)", cur_keys, cur_X[:, keep13]),
        ("现状10维(去信贷去等级)", cur_keys,
         cur_X[:, [i for i in range(len(F)) if i not in fin_idx and i not in (4, 5, 6)]]),
        ("现状7维(仅气象快照)", cur_keys, cur_X[:, wx_idx]),
        ("现状4维(温降风雪实测)", cur_keys, cur_X[:, [0, 1, 2, 3]]),
        ("现状3维(规则风险等级)", cur_keys, cur_X[:, [4, 5, 6]]),
        ("ERA5月度14维", met_keys, met_X),
        ("ERA5+县独热", met_keys, np.hstack([met_X, onehot])),
    ]


def main() -> None:
    feature_sets = build_feature_sets()
    labels = load_labels()
    for name, keys, X in feature_sets:
        print(f"  {name:<20} {X.shape}")
    print(f"  标签 {len(labels)} 条\n")

    rows: list[dict] = []
    for scope in ("all", "meteo"):
        for year in ("2023", "2024"):
            for name, keys, X in feature_sets:
                rows.append(evaluate(name, keys, X, labels, scope, year))

    head = (f"{'标签口径':<7}{'测试年':<7}{'特征集':<20}{'模型':<13}"
            f"{'AUC':>6}{'AP':>6}{'最优F1':>8}{'P@R90':>7}{'R@P30':>7}")
    print(head)
    print("-" * len(head))
    for r in rows:
        if r.get("skipped"):
            print(f"{r['label_scope']:<7}{r['test_year']:<7}{r['feature_set']:<20}跳过：{r['skipped']}")
            continue
        for mname, m in r["models"].items():
            if "auc" not in m:
                continue
            print(f"{r['label_scope']:<7}{r['test_year']:<7}{r['feature_set']:<20}{mname:<13}"
                  f"{m['auc']:>6.3f}{m.get('ap', 0):>6.3f}{m.get('best_f1', 0):>8.3f}"
                  f"{m.get('P_at_R90', 0):>7.3f}{m.get('R_at_P30', 0):>7.3f}")

    REPORT_FILE.write_text(json.dumps({
        "generated_by": "public_data/scripts/eval_meteo_vs_current.py",
        "label_source": LABEL_FILE.name,
        "feature_sources": {
            "现状*": "backend/models.py::_extract_monthly_samples（16 维；末 3 维为县内恒定的信贷字段）",
            "ERA5月度14维": "backend/data_store/meteo_monthly_features.json",
            "ERA5+县独热": "ERA5 月度特征 + 26 县独热（用于量化「县身份」带来的虚高）",
        },
        "protocol": "滚动前向：训练 = 全部 ≤ 上一年 12 月；测试 = 指定年全年；分类器直接拟合二值标签",
        "caveats": [
            "未确认月按项目现状口径作为负样本参与评价；不等同已确认无灾。",
            "best_f1 / P@R90 / R@P30 由测试集上的阈值曲线给出，属乐观口径，不代表部署可达。",
            "测试年正样本仅一二十条，单个样本的变动即带来数个百分点抖动。",
            "本项目原有回测用的是回归器（RandomForestRegressor 拟合规则分数），本表用分类器直接拟合二值标签，两者不可直接比较。",
        ],
        "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细已写出：{REPORT_FILE}")


if __name__ == "__main__":
    main()
