#!/usr/bin/env python
"""レポートの Markdown を Chrome Headless で PDF に変換する。

既存の汎用スクリプトは画像と生 HTML ブロックを落としてしまうので、
このレポートの記法（見出し・段落・箇条書き・表・コード・画像・生 HTML）だけを
扱う小さな変換器をここに置く。HTML は Markdown と同じディレクトリに書き出し、
画像の相対パスがそのまま解決できるようにする。
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm 15mm; }
body {
  font-family: "Hiragino Sans", "Hiragino Kaku Gothic ProN", sans-serif;
  font-size: 10.2pt; line-height: 1.75; color: #1d2228; margin: 0;
}
h1 { font-size: 19pt; margin: 0 0 4px; letter-spacing: .01em; }
h1 + p { color: #5b6470; font-size: 10.5pt; margin: 0 0 2px; }
h2 {
  font-size: 14pt; margin: 26px 0 10px; padding-bottom: 5px;
  border-bottom: 2px solid #2b6cb0; page-break-after: avoid;
}
h3 { font-size: 11.6pt; margin: 18px 0 7px; color: #1a3a5c; page-break-after: avoid; }
p { margin: 7px 0; }
ul, ol { margin: 7px 0 7px 1.3em; padding: 0; }
li { margin: 3px 0; }
hr { border: 0; border-top: 1px solid #dfe3e8; margin: 20px 0; }
strong { color: #10253c; }
code {
  font-family: "SF Mono", Menlo, monospace; font-size: 8.8pt;
  background: #f1f3f6; padding: 1px 4px; border-radius: 3px;
}
pre {
  background: #f7f8fa; border: 1px solid #e2e6ec; border-radius: 5px;
  padding: 10px 12px; overflow-x: auto; page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 8.4pt; line-height: 1.55; }
table {
  border-collapse: collapse; width: 100%; margin: 10px 0;
  font-size: 9.4pt; page-break-inside: avoid;
}
th, td { border: 1px solid #d7dce3; padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #eef2f7; font-weight: 600; }
tbody tr:nth-child(even) { background: #fafbfc; }
img { max-width: 100%; display: block; margin: 12px auto; border: 1px solid #e2e6ec; border-radius: 4px; }
figcaption { text-align: center; color: #6b7480; font-size: 8.8pt; margin: -6px 0 14px; }

/* 構成図（本文中の生 HTML） */
.diagram { margin: 14px 0; page-break-inside: avoid; }
.diagram .row { display: flex; align-items: stretch; justify-content: center; gap: 6px; margin: 6px 0; }
.diagram .row.down { margin: 2px 0; }
.diagram .box {
  flex: 1; border: 1px solid #c3cbd6; border-radius: 6px; padding: 8px 9px;
  background: #fbfcfe; font-size: 9pt; text-align: center; font-weight: 600;
}
.diagram .box.accent { background: #eaf2fd; border-color: #8fb4e3; }
.diagram .box.accent2 { background: #fdf2e8; border-color: #e3b68f; }
.diagram .box .sub { display: block; font-weight: 400; color: #5b6470; font-size: 8.1pt; margin-top: 3px; }
.diagram .arrow { align-self: center; color: #7a8290; font-size: 13pt; flex: 0 0 auto; }
"""


def inline(text: str) -> str:
    """段落内の記法（コード・強調・リンク）を HTML にする。"""
    out = html.escape(text, quote=False)
    out = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", out)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', out)
    return out


def convert(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    list_tag: str | None = None

    def close_list() -> None:
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 生 HTML ブロック（<div ...> から対応する </div> まで素通し）
        if stripped.startswith("<div"):
            close_list()
            depth = 0
            while i < len(lines):
                out.append(lines[i])
                depth += lines[i].count("<div")
                depth -= lines[i].count("</div>")
                i += 1
                if depth <= 0:
                    break
            continue

        # コードブロック
        if stripped.startswith("```"):
            close_list()
            i += 1
            body = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body.append(html.escape(lines[i], quote=False))
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(body) + "</code></pre>")
            continue

        # 表
        if stripped.startswith("|") and stripped.endswith("|"):
            close_list()
            header = [c.strip() for c in stripped.strip("|").split("|")]
            i += 1
            if i < len(lines) and re.match(r"^\|[\s:\-|]+\|$", lines[i].strip()):
                i += 1
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append("<table><thead><tr>"
                       + "".join(f"<th>{inline(c)}</th>" for c in header)
                       + "</tr></thead><tbody>")
            for row in rows:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
            out.append("</tbody></table>")
            continue

        # 画像（単独行）
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", stripped)
        if m:
            close_list()
            alt, src = m.group(1), m.group(2)
            out.append(f'<img src="{src}" alt="{html.escape(alt)}">')
            if alt:
                out.append(f"<figcaption>{html.escape(alt)}</figcaption>")
            i += 1
            continue

        # 見出し
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            close_list()
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            i += 1
            continue

        # 水平線
        if re.match(r"^-{3,}$", stripped):
            close_list()
            out.append("<hr>")
            i += 1
            continue

        # 箇条書き
        m = re.match(r"^[-*]\s+(.*)$", stripped)
        if m:
            if list_tag != "ul":
                close_list()
                out.append("<ul>")
                list_tag = "ul"
            out.append(f"<li>{inline(m.group(1))}</li>")
            i += 1
            continue
        m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m:
            if list_tag != "ol":
                close_list()
                out.append("<ol>")
                list_tag = "ol"
            out.append(f"<li>{inline(m.group(2))}</li>")
            i += 1
            continue

        if not stripped:
            close_list()
            i += 1
            continue

        close_list()
        out.append(f"<p>{inline(stripped)}</p>")
        i += 1

    close_list()
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", type=str, default="")
    args = ap.parse_args()

    md = args.src.read_text(encoding="utf-8")
    body = convert(md)
    title = args.title or args.src.stem
    doc = (
        "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )
    # 画像の相対パスを解決するため、Markdown と同じディレクトリに置く
    html_path = args.src.with_suffix(".html")
    html_path.write_text(doc, encoding="utf-8")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        CHROME, "--headless", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer", "--virtual-time-budget=6000",
        f"--print-to-pdf={args.out.resolve()}", html_path.resolve().as_uri(),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if not args.out.exists():
        print(res.stderr[-1500:], file=sys.stderr)
        return 1
    print(f"PDF: {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
