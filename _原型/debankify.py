"""把银行版控制台（/bank，7 个页面）做「去金融化」：只改**文案 + 配色**，不动结构与数据流。

## 作用范围（用户裁定）

- 只改**展示文案**与**配色**；页面结构、路由 key（`dashboard/pool/profile/ledger/postloan/region/insurance`）、
  页内 Tab key（`ov/doc/asset/credit/pl/ev`）一律保留 —— 换 key 要连带改 JS 逻辑，超出「只改文案」的边界。
- 不改数据流：`finance_credit.json` 等信贷字段仍照旧流入，只是**标签换成了风险/暴露的语言**。
  ⚠️ 这一点必须在报告里写明，不能让人以为底层数据已经换掉。

## 替换策略

按**从长到短**顺序替换，避免「授信额度」被先替换成「风险敞口额度」后又被「授信」二次命中。
所有 `from` 都是中文展示串，不可能命中代码标识符（标识符是英文），因此不会破坏逻辑。

用法：
    python _原型/debankify.py            # 预览（不写文件）
    python _原型/debankify.py --apply    # 落盘
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = [ROOT / "frontend" / "bank.html", ROOT / "frontend" / "bank.js"]
CSS = ROOT / "frontend" / "bank.css"

# ---- 后端展示串：这些字符串会渲染到页面上，但定义在 Python 里 ----
# 只改**值**（展示文案），不改 dict 的 key（key 是英文标识符）。
BACKEND_TARGETS = [ROOT / "backend" / "bank_view.py"]
BACKEND_MAP: list[tuple[str, str]] = [
    ("授信资料", "敞口资料"),
    ("统一授信额度", "可用敞口额度"),
    ("授信额度", "风险敞口额度"),
    ("已用额度", "已用敞口"),
    ("还款状态", "履约状态"),
    ("还款类", "履约类"),
    ("组织成银行信贷视角", "组织成气象灾害风险视角"),
    ("银行信贷视角", "气象灾害风险视角"),
    ("贷后任务", "灾后任务"),
    ("贷后", "灾后"),
]
# 测试里断言了资料分组名，改名后必须同步，否则测的是旧标签
TEST_FILE = ROOT / "backend" / "test_bank_api.py"

# ---- 文案映射：从长到短 ----
TEXT_MAP: list[tuple[str, str]] = [
    # 修复上一版非幂等脚本造成的叠词，并清掉最后的「额度链」残留
    # （放在最前，长串优先）
    ("预警预警工作台", "预警工作台"),
    ("额度链", "测算链"),
    # 品牌：工银牧融 -> 融天气象（必须最先，否则会被后面的「工银」「牧融」拆坏）
    ("工银牧融", "融天气象"),
    ("工行业务数据", "机构业务数据"),
    ("工银", "融天"),
    ("牧融", "气象"),
    # 额度池等第二轮补充
    ("集中度与额度池", "集中度与敞口池"),
    ("投放与额度池占用", "投放与敞口池占用"),
    ("额度池", "敞口池"),
    ("增信额度", "保障额度"),
    ("分行额度政策", "区域敞口政策"),
    ("同额度看", "同敞口看"),
    ("建议金额", "预警额度"),
    # 品牌与标题
    ("高原畜牧活体资产授信与贷后风控", "高原气象灾害预警与牧区暴露风险管理"),
    ("行内已有授信额度不等同于建议金额", "系统内存量敞口不等同于预警结论"),
    ("行内已有授信额度", "系统内存量敞口"),
    ("行内已有授信记录", "系统内存量业务记录"),
    ("行内授信记录", "系统内存量业务记录"),
    ("统一授信可用额度", "可用敞口额度"),
    ("按行内已有授信额度排序", "按存量敞口排序"),
    ("主体与授信字段来自", "主体与敞口字段来自"),
    ("主体名称、存栏、授信额度", "主体名称、存栏、存量敞口"),
    ("已有授信额度", "存量风险敞口"),
    ("授信额度", "风险敞口额度"),
    ("授信准入", "预警纳入"),
    ("授信要素", "敞口要素"),
    ("额度测算", "敞口测算"),
    ("建议金额只由唯一额度链测算给出", "预警额度只由唯一测算链给出"),
    ("建议金额只能由唯一额度链", "预警额度只能由唯一测算链"),
    ("唯一额度链", "唯一测算链"),
    ("额度链口径", "测算链口径"),
    ("自助测额度留资", "自助测算留资"),
    ("已用额度", "已用敞口"),
    ("用信率", "敞口使用率"),
    ("还款保障倍数", "偿债保障倍数"),
    ("还款状态", "履约状态"),
    ("还款类”, “履约", "履约类"),
    ("还款类：逾期", "履约类：逾期"),
    ("还款类", "履约类"),
    ("贷款用途", "资金用途"),
    ("有授信户数", "有敞口户数"),
    ("暂不放款", "暂不纳入"),
    ("活体资产台账", "暴露资产台账"),
    ("贷后待办", "灾后核查"),
    ("贷后", "灾后"),
    ("获客与授信", "监测与预警"),
    ("客户池", "监测对象"),
    ("客户档案", "对象档案"),
    ("客户经理", "预警值班员"),
    ("那曲市牧区预警值班员 2 组", "那曲市牧区预警值班 2 组"),
    ("看待营销名单", "看待核查名单"),
    ("待营销名单", "待核查名单"),
    ("资产与风控", "暴露与风险"),
    ("区域与集中度", "区域与暴露度"),
    ("保险协同", "保险联动"),
    ("绿色信贷余额", "暴露资产规模"),
    ("抵押物在栏", "在栏牲畜"),
    ("保险待核验", "投保待核验"),
    ("处理待办", "处理任务"),
    ("主体与授信", "主体与敞口"),
    ("授信", "敞口"),
    ("信贷", "金融"),
    ("贷款", "融资"),
    ("银行", "机构"),
    ("工作台", "预警工作台"),
]

# ---- 配色：工行红 -> 气象蓝（浅色科技风，与全站浅色主题一致）----
PALETTE: list[tuple[str, str]] = [
    ("--red: #C7000B", "--accent: #0B63C7"),
    ("--red-bg: #FFF5F4", "--accent-bg: #EEF5FF"),
    ("--red-line: #FFCCC7", "--accent-line: #CFE2FA"),
    ("--red-soft: #FFE8E5", "--accent-soft: #E3EEFC"),
    ("var(--red-bg)", "var(--accent-bg)"),
    ("var(--red-line)", "var(--accent-line)"),
    ("var(--red-soft)", "var(--accent-soft)"),
    ("var(--red)", "var(--accent)"),
]


def apply_text() -> dict[str, int]:
    """单遍替换：把命中先换成哨兵再统一还原，避免「A 的产物又被 B 命中」。

    幂等保护：若替换词 `to` 里包含 `from`（例如 工作台 → 预警工作台），
    第二次运行会命中自己产出的字符串，产生「预警预警工作台」。
    对这类规则改用负向后视，只替换前面不是 `to` 前缀的那些。
    """
    counts: dict[str, int] = {}
    for path in TARGETS:
        text = path.read_text(encoding="utf-8")
        sentinels: list[tuple[str, str]] = []
        for frm, to in TEXT_MAP:
            if frm in to:
                prefix = to[:to.index(frm)]
                pattern = re.compile(rf"(?<!{re.escape(prefix)}){re.escape(frm)}")
            else:
                pattern = re.compile(re.escape(frm))
            n = len(pattern.findall(text))
            if not n:
                continue
            token = f"\x00{len(sentinels)}\x00"
            text = pattern.sub(token, text)
            sentinels.append((token, to))
            counts[f"{frm} → {to}"] = counts.get(f"{frm} → {to}", 0) + n
        for token, to in sentinels:
            text = text.replace(token, to)
        path.write_text(text, encoding="utf-8")
    return counts


def apply_palette() -> dict[str, int]:
    text = CSS.read_text(encoding="utf-8")
    counts: dict[str, int] = {}
    for frm, to in PALETTE:
        n = text.count(frm)
        if n:
            text = text.replace(frm, to)
            counts[f"{frm} → {to}"] = n
    text = text.replace(
        "工银牧融 · 银行版控制台样式（/bank）",
        "融天气象 · 高原气象灾害预警控制台样式（/bank）")
    CSS.write_text(text, encoding="utf-8")
    return counts


def apply_backend() -> dict[str, int]:
    """改后端里的**展示串**（以及测试里对应的断言），不动 dict key。"""
    counts: dict[str, int] = {}
    for path in BACKEND_TARGETS + [TEST_FILE]:
        text = path.read_text(encoding="utf-8")
        for frm, to in BACKEND_MAP:
            n = text.count(frm)
            if n:
                text = text.replace(frm, to)
                counts[f"{path.name}: {frm} → {to}"] = n
        path.write_text(text, encoding="utf-8")
    return counts


def main(argv: list[str]) -> int:
    do_apply = "--apply" in argv
    if not do_apply:
        print("预览模式（不写文件）—— 用 --apply 落盘\n")
        for path in TARGETS:
            text = path.read_text(encoding="utf-8")
            hit = [f"  {text.count(f):3d} × {f} → {t}" for f, t in TEXT_MAP if text.count(f)]
            print(f"== {path.name}（{len(hit)} 条命中）")
            print("\n".join(hit) or "  （无）")
        print()
        return 0

    tc = apply_text()
    pc = apply_palette()
    bc = apply_backend()
    print(f"文案替换：{len(tc)} 条规则命中，共 {sum(tc.values())} 处")
    for k, v in sorted(tc.items(), key=lambda x: -x[1]):
        print(f"  {v:3d}  {k}")
    print(f"\n后端展示串替换：{len(bc)} 条规则命中，共 {sum(bc.values())} 处")
    for k, v in sorted(bc.items(), key=lambda x: -x[1]):
        print(f"  {v:3d}  {k}")
    print(f"\n配色替换：{len(pc)} 条规则")
    for k, v in pc.items():
        print(f"  {v:3d}  {k}")

    # 残留检查：还有没有明显的金融词
    left: dict[str, int] = {}
    for path in TARGETS + [CSS]:
        text = path.read_text(encoding="utf-8")
        for w in ("授信", "额度", "放款", "贷款", "信贷", "还款", "用信", "客户经理", "工银"):
            n = text.count(w)
            if n:
                left[f"{path.name}:{w}"] = n
    print("\n残留金融词：" + ("无 ✅" if not left else str(left)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
