"""同口径对比 + 配对 bootstrap：新旧两版的差异到底是真的，还是样本噪声？

只做一件事：在**完全相同的测试格**上比较「旧特征+随机森林」与「新特征+各模型」，
用配对 bootstrap 给出 AUC 差值的 95% 置信区间。
若区间跨 0，就不能说谁更好。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import models as M                       # noqa: E402
import models_deep as MDeep              # noqa: E402

SEED = 42
YEARS = ("2023", "2024")


def label_map() -> dict[tuple[str, str], int]:
    rows = json.loads((BACKEND / "data_store" / "meteo_event_labels.json")
                      .read_text(encoding="utf-8"))["records"]
    out = {}
    for r in rows:
        if r["label"] == 1 and not r.get("is_meteo"):
            continue                      # 非气象类正样本剔除（与实验口径一致）
        out[(r["region_id"], r["month"])] = 1 if r["label"] == 1 else 0
    return out


def old_matrix() -> tuple[list, np.ndarray]:
    X, _y, meta, _info = M._extract_monthly_samples()
    return [(m["region_id"], m["month"]) for m in meta], X


def new_matrix() -> tuple[list, np.ndarray]:
    d = np.load(BACKEND / "data_store" / "meteo_dataset.npz")
    graph = json.loads((BACKEND / "data_store" / "region_graph.json").read_text(encoding="utf-8"))
    ids = graph["region_ids"]
    labels = json.loads((BACKEND / "data_store" / "meteo_event_labels.json")
                        .read_text(encoding="utf-8"))["records"]
    months = sorted({r["month"] for r in labels})
    seq = d["seq"]                                    # (T, L, N, F)
    T, L, N, F = seq.shape
    flat = seq.transpose(0, 2, 1, 3).reshape(T, N, L * F)
    keys = [(ids[j], months[t]) for t in range(T) for j in range(N)]
    return keys, flat.reshape(T * N, L * F)


def collect(keys, X, truth, year):
    """滚动前向切分：**训练 = 严格早于测试年**，测试 = 测试年当年。

    ⚠️ 这里曾经写成 `te if startswith(year) else tr`，于是测试年**之后**的月份也被塞进了训练集
    （2023 年测试时把 2024 也训练了），训练集从 882 涨到 1174，AUC 从 0.612 掉到 0.507——
    既是数据泄漏，也说明这种错误会把结论完全带偏。切分必须用「月份小于起始月」而不是「不等于测试年」。
    """
    ks, ys, Xs = [], [], []
    tr, te = [], []
    cut, end = f"{year}-01", f"{year}-12"
    for i, k in enumerate(keys):
        if k not in truth:
            continue
        ks.append(k); ys.append(truth[k]); Xs.append(X[i])
        idx = len(ks) - 1
        if k[1] < cut:
            tr.append(idx)
        elif k[1] <= end:
            te.append(idx)
    return Xs, np.array(ys), ks, np.array(tr), np.array(te)


def rf_scores(X, y, tr, te, seed=SEED):
    """旧版 RF 会因特征未标准化而不同，两版都交给同一流程；树模型不需要标准化。"""
    Xa = np.asarray(X)
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                 class_weight="balanced", random_state=seed)
    clf.fit(Xa[tr], y[tr])
    return clf.predict_proba(Xa[te])[:, 1]


def paired_bootstrap(y, s1, s2, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    diffs = []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(set(y[b].tolist())) < 2:
            continue
        diffs.append(roc_auc_score(y[b], s2[b]) - roc_auc_score(y[b], s1[b]))
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> int:
    truth = label_map()
    ok, Xo = old_matrix()
    nk, Xn = new_matrix()

    print(f"{'年':<6}{'对比（新 − 旧）':<34}{'训练':>6}{'旧AUC':>8}{'新AUC':>8}{'差值':>8}{'95%CI':>20}  判定")
    print("-" * 108)
    for year in YEARS:
        # ⚠️ 必须用 collect 返回的**过滤后**矩阵（Xo_f），不能传原始全矩阵 Xo：
        # 后者行序与标签并不对应（含 2015-2019 无标签月份），用过滤后的索引去切会整体错位，
        # 实测会让 AUC 从 0.612 掉到 0.490（≈随机）。这个坑踩过一次，写在这里备忘。
        Xo_f, yo, ko, tro, teo = collect(ok, Xo, truth, year)
        if len(teo) == 0:
            continue
        s_old = rf_scores(Xo_f, yo, tro, teo)

        # 新特征：按同一批测试格对齐
        Xn_f, yl, kl, trl, tel = collect(nk, Xn, truth, year)
        pos = {k: i for i, k in enumerate(kl)}
        te_keys = [ko[i] for i in teo]
        s_new = rf_scores(Xn_f, yl, trl, tel)
        map_new = {kl[t]: s_new[i] for i, t in enumerate(tel)}
        s_new_aligned = np.array([map_new[k] for k in te_keys])
        y_common = np.array([truth[k] for k in te_keys])

        m, lo, hi = paired_bootstrap(y_common, s_old, s_new_aligned)
        verdict = "新显著更好" if lo > 0 else ("旧显著更好" if hi < 0 else "不显著（噪声内）")
        print(f"{year:<6}{'新34维×6月+RF  vs  旧16维单月+RF':<34}{len(tro):>6}"
              f"{roc_auc_score(y_common, s_old):>8.3f}{roc_auc_score(y_common, s_new_aligned):>8.3f}"
              f"{m:>8.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>20}  {verdict}")

    # 再补一组：新版最强模型（LSTM 三年均值最高）与旧版对比，口径同上
    print(f"\n{'年':<6}{'对比（新 − 旧）':<34}{'训练':>6}{'旧AUC':>8}{'新AUC':>8}{'差值':>8}{'95%CI':>20}  判定")
    print("-" * 108)
    import models_deep as _md  # noqa: F401
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    for year in YEARS:
        Xo_f, yo, ko, tro, teo = collect(ok, Xo, truth, year)
        Xn_f, yl, kl, trl, tel = collect(nk, Xn, truth, year)
        if len(teo) == 0 or len(tel) == 0:
            continue
        s_old = rf_scores(Xo_f, yo, tro, teo)

        # LSTM：结构与 run_experiments.SeqBaseline 一致
        sys.path.insert(0, str(ROOT / "public_data" / "scripts"))
        from run_experiments import SeqBaseline, train_torch, get_design_data, standardize
        d = np.load(BACKEND / "data_store" / "meteo_dataset.npz")
        meta = json.loads((BACKEND / "data_store" / "meteo_dataset_meta.json").read_text(encoding="utf-8"))
        labels = json.loads((BACKEND / "data_store" / "meteo_event_labels.json").read_text(encoding="utf-8"))["records"]
        months = sorted({r["month"] for r in labels})
        batch = get_design_data()
        torch.manual_seed(SEED)
        seq_std, _ = standardize(d["seq"], np.array([i for i, mm in enumerate(months) if mm < f"{year}-01"]))
        tr_t = np.array([i for i, mm in enumerate(months) if mm < f"{year}-01"])
        te_t = np.array([i for i, mm in enumerate(months) if mm.startswith(year)])
        model = SeqBaseline("lstm", meta["n_meteo_features"] + meta["n_remote_features"], meta["lookback"])
        out = train_torch(model, seq_std, batch, tr_t, te_t, use_physics=False, epochs=80)
        s_lstm = out["score"].reshape(-1)

        pos_new = {k: i for i, k in enumerate(kl)}
        te_keys = [ko[i] for i in teo]
        map_l = {kl[t]: s_lstm[i] for i, t in enumerate(tel)}
        s_new_aligned = np.array([map_l[k] for k in te_keys])
        y_common = np.array([truth[k] for k in te_keys])
        m, lo, hi = paired_bootstrap(y_common, s_old, s_new_aligned)
        verdict = "新显著更好" if lo > 0 else ("旧显著更好" if hi < 0 else "不显著（噪声内）")
        print(f"{year:<6}{'新LSTM  vs  旧16维+RF':<34}{len(tro):>6}"
              f"{roc_auc_score(y_common, s_old):>8.3f}{roc_auc_score(y_common, s_new_aligned):>8.3f}"
              f"{m:>8.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>20}  {verdict}")

    print("\n说明：测试年正例仅十几格；配对 bootstrap 采样 2000 次，区间跨 0 即不可下结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
