"""前端渲染自检：打开 /meteo.html，等数据加载完，收集报错并验证真的画出了东西。

沿用项目已沉淀的坑：多开把 HOME 重定向 → 必须显式指定
`PLAYWRIGHT_BROWSERS_PATH`，否则会报 "Executable doesn't exist"。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BROWSERS = r"C:\Users\WH\AppData\Local\ms-playwright"
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", BROWSERS)

from playwright.sync_api import sync_playwright   # noqa: E402

URL = "http://127.0.0.1:8100/meteo.html"
SHOT = Path(__file__).resolve().parent / "meteo_page.png"


def main() -> int:
    errors: list[str] = []
    warns: list[str] = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1600, "height": 1200})
        pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        pg.on("console", lambda m: (errors if m.type == "error" else warns).append(
            f"console.{m.type}: {m.text}"))
        pg.goto(URL, wait_until="domcontentloaded")
        try:
            pg.wait_for_function(
                "() => { const s=document.getElementById('status');"
                " return s && (s.textContent.includes('已加载') || s.textContent.includes('失败')); }",
                timeout=90000)
        except Exception as exc:
            errors.append(f"等待加载超时: {exc}")

        status = pg.inner_text("#status")
        rows = pg.eval_on_selector_all("#tbl tbody tr", "els => els.length")
        cells = pg.eval_on_selector_all("#heat .cell", "els => els.length")
        bars = pg.eval_on_selector_all("#attrBars .bar", "els => els.length")
        attr = pg.inner_text("#attrWho")

        # 画布是否真的画了东西：抽样统计非背景像素比例
        painted = pg.evaluate("""() => {
            const out = {};
            for (const id of ['cTemp','cRisk']) {
                const c = document.getElementById(id);
                const g = c.getContext('2d');
                const d = g.getImageData(0,0,c.width,c.height).data;
                let n=0, tot=0;
                for (let i=0;i<d.length;i+=4*997){ tot++; if (d[i+3] > 10) n++; }
                out[id] = +(n/Math.max(1,tot)).toFixed(3);
            }
            return out;
        }""")
        pg.screenshot(path=str(SHOT), full_page=True)
        b.close()

    print(f"状态栏        : {status}")
    print(f"明细表行数    : {rows}（应为 26）")
    print(f"热力图格子    : {cells}（应为 26 × 5 = 130）")
    print(f"归因条数      : {bars}（应为 7 个通道）")
    print(f"归因对象      : {attr}")
    print(f"画布覆盖比例  : {painted}")
    print(f"截图          : {SHOT}")
    print(f"pageerror     : {len(errors)} 条")
    for e in errors[:8]:
        print("   ", e)
    print(f"警告          : {len(warns)} 条")
    for w in warns[:3]:
        print("   ", w)

    ok = (not errors and rows == 26 and cells == 130 and bars >= 5
          and all(v > 0.5 for v in painted.values()) and "失败" not in status)
    print("\n结论:", "✅ 页面渲染通过" if ok else "❌ 有问题，见上")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
