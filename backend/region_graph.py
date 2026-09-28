"""26 个县的空间邻接图（供图卷积 / ST-GCN 分支使用）。

## 为什么需要它

计划里的「遥感空间分支」要捕捉县与县之间的空间扩散（雪云过境、草场连片受灾）。
图卷积需要一个邻接矩阵，而项目现在没有任何空间关系的表达——26 个县是彼此独立的行。

## 构图方法（三步，每一步都写在 meta 里以便复现）

1. **距离**：按经纬度算大圆距离（Haversine），得到 26×26 距离矩阵。
2. **连边**：每个县连最近的 `K_NEIGHBORS` 个县，再把距离超过 `MAX_EDGE_KM` 的边剪掉，
   最后对称化（保证无向图）。
3. **连通性**：检查连通分量；若被切成多块，把各分量按「最近的县对」串起来。
   —— 孤岛在气象上说不通（高原天气系统不认行政边界），所以必须强制连通，
   但这一步要在报告里如实披露。

输出里同时给出**未归一化邻接**与 **GCN 用归一化邻接**：`Â = D^-1/2 (A+I) D^-1/2`。

用法：
    python backend/region_graph.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
REGION_LIST = ROOT / "public_data" / "region_list.csv"
OUTPUT_FILE = BASE / "data_store" / "region_graph.json"

K_NEIGHBORS = 4          # 每县连最近 4 个县
MAX_EDGE_KM = 600.0      # 超过这个距离的边剪掉（避免把相距上千公里的县连成一条边）


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


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
            "latitude": float(r["latitude"]),
            "longitude": float(r["longitude"]),
            "altitude": float(r.get("altitude") or 0),
            "pasture_type": (r.get("pasture_type") or "").strip(),
        })
    return sorted(out, key=lambda x: x["region_id"])


def distance_matrix(regions: list[dict]) -> list[list[float]]:
    n = len(regions)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            km = haversine_km(regions[i]["latitude"], regions[i]["longitude"],
                              regions[j]["latitude"], regions[j]["longitude"])
            d[i][j] = d[j][i] = round(km, 2)
    return d


def build_adjacency(dist: list[list[float]]) -> tuple[list[list[int]], list[dict]]:
    n = len(dist)
    adj = [[0] * n for _ in range(n)]
    edges: dict[tuple[int, int], float] = {}

    for i in range(n):
        ranked = sorted((j for j in range(n) if j != i), key=lambda j: dist[i][j])
        for j in ranked[:K_NEIGHBORS]:
            if dist[i][j] <= MAX_EDGE_KM:
                a, b = min(i, j), max(i, j)
                adj[a][b] = adj[b][a] = 1
                edges[(a, b)] = dist[a][b]

    # KNN 图通常能连起来，但不保证；断开就按「最近县对」把分量串起来
    forced: list[list[int]] = []
    comps = _components(adj)
    while len(comps) > 1:
        best = None
        for comp in comps[1:]:
            for i in comps[0]:
                for j in comp:
                    if best is None or dist[i][j] < best[0]:
                        best = (dist[i][j], i, j)
        assert best is not None
        _, i, j = best
        a, b = min(i, j), max(i, j)
        adj[a][b] = adj[b][a] = 1
        edges[(a, b)] = dist[a][b]
        forced.append([a, b])
        comps = _components(adj)

    return adj, [
        {"source": i, "target": j, "distance_km": km} for (i, j), km in sorted(edges.items())
    ], forced


def _components(adj: list[list[int]]) -> list[list[int]]:
    n = len(adj)
    seen = [False] * n
    comps: list[list[int]] = []
    for s in range(n):
        if seen[s]:
            continue
        comp, stack = [], [s]
        seen[s] = True
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in range(n):
                if adj[v][w] and not seen[w]:
                    seen[w] = True
                    stack.append(w)
        comps.append(sorted(comp))
    return comps


def normalized_adjacency(adj: list[list[int]]) -> list[list[float]]:
    """GCN 用：Â = D^-1/2 (A+I) D^-1/2，自带自环。"""
    n = len(adj)
    a_hat = [[adj[i][j] + (1 if i == j else 0) for j in range(n)] for i in range(n)]
    deg = [sum(row) for row in a_hat]
    inv_sqrt = [1.0 / math.sqrt(d) if d > 0 else 0.0 for d in deg]
    return [[round(a_hat[i][j] * inv_sqrt[i] * inv_sqrt[j], 6) for j in range(n)]
            for i in range(n)]


def build_all() -> dict:
    regions = load_regions()
    dist = distance_matrix(regions)
    adj, edges, forced = build_adjacency(dist)
    norm = normalized_adjacency(adj)

    deg = [sum(row) for row in adj]
    comps = _components(adj)
    neighbor_km = [dist[i][j] for i in range(len(regions))
                   for j in range(len(regions)) if adj[i][j]]

    return {
        "meta": {
            "source": REGION_LIST.name,
            "n_regions": len(regions),
            "method": f"Haversine 距离 + 最近 {K_NEIGHBORS} 邻接 + {MAX_EDGE_KM:.0f}km 截断 + 对称化",
            "undirected": True,
            "k_neighbors": K_NEIGHBORS,
            "max_edge_km": MAX_EDGE_KM,
            "n_edges": len(edges),
            "n_components": len(comps),
            "forced_bridge_edges": [
                {"source": regions[i]["region_id"], "target": regions[j]["region_id"],
                 "distance_km": dist[i][j]} for i, j in forced
            ],
            "degree": {"min": min(deg), "max": max(deg),
                       "mean": round(sum(deg) / len(deg), 2)},
            "edge_distance_km": {"min": min(neighbor_km), "max": max(neighbor_km),
                                 "mean": round(sum(neighbor_km) / len(neighbor_km), 1)},
            "elevation_range_m": [min(r["altitude"] for r in regions),
                                  max(r["altitude"] for r in regions)],
            "caveats": [
                "连边基于行政中心的直线距离，不等于地形连通性（未用 DEM）。",
                "KNN 图不保证连通；若出现多分量，已按最近县对强制串接，见 forced_bridge_edges。",
                "归一化邻接为 GCN 用 Â = D^-1/2 (A+I) D^-1/2。",
            ],
        },
        "region_ids": [r["region_id"] for r in regions],
        "regions": regions,
        "distance_km": dist,
        "adjacency": adj,
        "adjacency_normalized": norm,
        "edges": edges,
    }


def main() -> None:
    payload = build_all()
    OUTPUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                           encoding="utf-8")
    m = payload["meta"]
    print(f"已写出 {OUTPUT_FILE}")
    print(f"  {m['n_regions']} 个县 / {m['n_edges']} 条边 / 连通分量 {m['n_components']}")
    print(f"  度数 min {m['degree']['min']} max {m['degree']['max']} mean {m['degree']['mean']}")
    print(f"  边长 km：min {m['edge_distance_km']['min']} max {m['edge_distance_km']['max']} "
          f"mean {m['edge_distance_km']['mean']}")
    print(f"  点位高程范围 {m['elevation_range_m'][0]:.0f} ~ {m['elevation_range_m'][1]:.0f} m")
    if m["forced_bridge_edges"]:
        print("  ⚠️ 强制桥接的边：", [f"{e['source']}--{e['target']}" for e in m["forced_bridge_edges"]])
    print("\n  最近邻示例（前 3 个县）：")
    ids = payload["region_ids"]
    for i in range(min(3, len(ids))):
        nb = [(ids[j], payload["distance_km"][i][j]) for j in range(len(ids))
              if payload["adjacency"][i][j]]
        nb.sort(key=lambda x: x[1])
        print(f"    {ids[i]}: " + ", ".join(f"{n}({d:.0f}km)" for n, d in nb))


if __name__ == "__main__":
    main()
