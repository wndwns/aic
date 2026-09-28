"""银行版控制台「去金融化」后的渲染自检：7 个页面逐页打开，收集报错并抓图。"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", r"C:\Users\WH\AppData\Local\ms-playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

BASE = "http://127.0.0.1:8100"
OUT = Path(__file__).resolve().parent
PAGES = ["dashboard", "pool", "profile", "ledger", "postloan", "region", "insurance"]


def main() -> int:
    errors: list[str] = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1560, "height": 1000})
        pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        pg.on("console", lambda m: errors.append(f"console.error: {m.text}") if m.type == "error" else None)
        pg.goto(f"{BASE}/bank", wait_until="domcontentloaded")
        pg.wait_for_timeout(3500)

        nav = pg.eval_on_selector_all(".sidebar .nav-item, .sidebar a, .nav-group a",
                                      "els => els.map(e=>e.textContent.trim()).filter(Boolean)")
        groups = pg.eval_on_selector_all(".nav-group .nav-group-title, .nav-group h3, .nav-group h4",
                                        "els => els.map(e=>e.textContent.trim())")
        title = pg.title()
        brand = pg.inner_text(".brand")
        print(f"页面标题 : {title}")
        print(f"品牌区   : {brand.splitlines()[0] if brand else '—'}")
        print(f"导航分组 : {groups}")
        print(f"导航项({len(nav)}) : {nav}")

        for name in PAGES:
            try:
                pg.evaluate(f"() => window.__nav && window.__nav('{name}')")
            except Exception:
                pass
            # 直接点侧边栏对应项
            try:
                pg.click(f".sidebar >> text={name}", timeout=2500)
            except Exception:
                pass
            pg.wait_for_timeout(700)
            shot = OUT / f"bank_{name}.png"
            pg.screenshot(path=str(shot))
            txt = pg.inner_text("body")[:60].replace("\n", " ")
            print(f"  {name:10s} 已截图 {shot.name}  首屏片段: {txt}")

        b.close()

    print(f"\npageerror / console.error：{len(errors)} 条")
    for e in errors[:10]:
        print("   ", e)
    ok = not errors and len(nav) >= 7 and "工银" not in brand and "牧融" not in brand
    print("\n结论:", "✅ 7 页渲染无报错、品牌已替换" if ok else "❌ 见上")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
