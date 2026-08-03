from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CSS_FILE = "probability_statistics.css"
EXCLUDED_MARKDOWN = {"outline.md"}
TITLE_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
ORDERED_RE = re.compile(r"^(\s*)(\d+)\.\s+(.+)$")
UNORDERED_RE = re.compile(r"^(\s*)[-*]\s+(.+)$")
TASK_RE = re.compile(r"^\[([ xX])\]\s+(.+)$")


@dataclass(frozen=True)
class Heading:
    level: int
    text: str
    anchor: str


@dataclass(frozen=True)
class ListMarker:
    indent: int
    kind: str
    content: str


@dataclass(frozen=True)
class Page:
    source: Path
    output: Path
    title: str
    subtitle: str
    label: str
    headings: list[Heading]
    body: str
    word_count: int
    formula_count: int
    image_count: int
    mermaid_count: int


def markdown_files() -> list[Path]:
    return [
        path
        for path in sorted(ROOT.glob("*.md"))
        if path.name not in EXCLUDED_MARKDOWN
    ]


def plain_text(value: str) -> str:
    value = IMAGE_RE.sub(r"\1", value)
    value = LINK_RE.sub(r"\1", value)
    value = INLINE_CODE_RE.sub(r"\1", value)
    value = BOLD_RE.sub(r"\1", value)
    value = ITALIC_RE.sub(r"\1", value)
    return value.replace("\\", "").strip()


def slug_for(index: int, text: str) -> str:
    stem = re.sub(r"\s+", "-", plain_text(text).lower())
    stem = re.sub(r"[^a-z0-9\u4e00-\u9fff-]+", "", stem)
    stem = stem.strip("-")
    if not stem:
        stem = "section"
    return f"{index:02d}-{stem}"


def escape_attr(value: str) -> str:
    return html.escape(value, quote=True)


def convert_inline(value: str) -> str:
    placeholders: list[tuple[str, str]] = []

    def stash(fragment: str) -> str:
        key = f"\u0000{len(placeholders)}\u0000"
        placeholders.append((key, fragment))
        return key

    def image_sub(match: re.Match[str]) -> str:
        alt = escape_attr(match.group(1))
        src = escape_attr(match.group(2))
        return stash(
            f'<figure class="figure"><img src="{src}" alt="{alt}">'
            f"<figcaption>{html.escape(match.group(1))}</figcaption></figure>"
        )

    def code_sub(match: re.Match[str]) -> str:
        return stash(f"<code>{html.escape(match.group(1))}</code>")

    def link_sub(match: re.Match[str]) -> str:
        text = convert_inline(match.group(1))
        href = escape_attr(rewrite_link(match.group(2)))
        return stash(f'<a href="{href}">{text}</a>')

    value = IMAGE_RE.sub(image_sub, value)
    value = INLINE_CODE_RE.sub(code_sub, value)
    escaped = html.escape(value)
    escaped = BOLD_RE.sub(r"<strong>\1</strong>", escaped)
    escaped = ITALIC_RE.sub(r"<em>\1</em>", escaped)
    escaped = LINK_RE.sub(link_sub, escaped)

    for key, fragment in placeholders:
        escaped = escaped.replace(key, fragment)
    return escaped


def rewrite_link(href: str) -> str:
    if href.startswith(("http://", "https://", "#", "mailto:")):
        return href
    if href.endswith(".md"):
        target = Path(href)
        if target.name in EXCLUDED_MARKDOWN:
            return href
        return str(target.with_suffix(".html"))
    return href


def split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    first = lines[index].strip()
    second = lines[index + 1].strip()
    if not (first.startswith("|") and first.endswith("|")):
        return False
    cells = split_table_row(second)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def render_table(lines: list[str]) -> str:
    headers = split_table_row(lines[0])
    rows = [split_table_row(line) for line in lines[2:]]
    header_html = "".join(f"<th>{convert_inline(cell)}</th>" for cell in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{convert_inline(cell)}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "\n".join(body_rows)
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{header_html}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def list_marker(line: str) -> ListMarker | None:
    ordered_match = ORDERED_RE.match(line)
    if ordered_match:
        return ListMarker(
            indent=len(ordered_match.group(1).replace("\t", "    ")),
            kind="ol",
            content=ordered_match.group(3),
        )
    unordered_match = UNORDERED_RE.match(line)
    if unordered_match:
        return ListMarker(
            indent=len(unordered_match.group(1).replace("\t", "    ")),
            kind="ul",
            content=unordered_match.group(2),
        )
    return None


def render_list_item_content(content: str) -> tuple[str, bool]:
    task_match = TASK_RE.match(content)
    if not task_match:
        return convert_inline(content), False
    checked = " checked" if task_match.group(1).lower() == "x" else ""
    label = convert_inline(task_match.group(2))
    return f'<input type="checkbox" disabled{checked}>{label}', True


def render_list_block(lines: list[str], index: int, base_indent: int | None = None) -> tuple[str, int]:
    first_marker = list_marker(lines[index])
    if first_marker is None:
        return "", index
    indent = first_marker.indent if base_indent is None else base_indent
    kind = first_marker.kind
    items: list[str] = []
    has_task = False

    while index < len(lines):
        marker = list_marker(lines[index])
        if marker is None or marker.indent < indent or marker.indent != indent or marker.kind != kind:
            break

        content_html, is_task = render_list_item_content(marker.content)
        has_task = has_task or is_task
        item_parts = [content_html]
        index += 1

        while index < len(lines):
            next_marker = list_marker(lines[index])
            if next_marker is None:
                break
            if next_marker.indent > indent:
                nested_html, index = render_list_block(lines, index, next_marker.indent)
                item_parts.append(nested_html)
                continue
            break

        items.append("<li>" + "".join(item_parts) + "</li>")

    class_attr = ' class="task-list"' if has_task and kind == "ul" else ""
    return f"<{kind}{class_attr}>" + "\n".join(items) + f"</{kind}>", index


def render_blockquote(lines: list[str]) -> str:
    cleaned = []
    for line in lines:
        text = line[1:]
        if text.startswith(" "):
            text = text[1:]
        cleaned.append(text)
    text = "\n".join(cleaned).strip()
    class_name = "callout"
    plain = plain_text(text)
    if any(keyword in plain for keyword in ("误区", "注意", "风险", "危险", "不能")):
        class_name += " warn"
    elif any(keyword in plain for keyword in ("好习惯", "提醒", "关键", "核心")):
        class_name += " tip"
    elif any(keyword in plain for keyword in ("总结", "一句话", "本质")):
        class_name += " ok"
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    rendered = []
    for paragraph in paragraphs:
        rendered.append("<p>" + "<br>".join(convert_inline(line) for line in paragraph.splitlines()) + "</p>")
    return f'<blockquote class="{class_name}">' + "".join(rendered) + "</blockquote>"


def render_paragraph(lines: list[str]) -> str:
    parts = []
    for line in lines:
        if line.endswith("  "):
            parts.append(convert_inline(line.rstrip()) + "<br>")
        else:
            parts.append(convert_inline(line.strip()))
    return "<p>" + " ".join(parts) + "</p>"


def render_code_block(language: str, code_lines: list[str]) -> str:
    code = "\n".join(code_lines)
    if language == "mermaid":
        return (
            '<div class="mermaid-wrap"><pre class="mermaid">'
            f"{html.escape(code)}</pre></div>"
        )
    class_attr = f' class="language-{escape_attr(language)}"' if language else ""
    return f"<pre><code{class_attr}>{html.escape(code)}</code></pre>"


def extract_subtitle(lines: list[str]) -> str:
    quote_lines = []
    collecting = False
    for line in lines:
        if line.startswith(">"):
            collecting = True
            text = line[1:].strip()
            if text:
                quote_lines.append(plain_text(text))
            if len(" ".join(quote_lines)) > 80:
                break
        elif collecting:
            break
    subtitle = " ".join(quote_lines).strip()
    if not subtitle:
        return "把 Markdown 教程整理成可直接阅读的 HTML 页面, 保留公式, 图表和章节结构."
    if len(subtitle) > 118:
        subtitle = subtitle[:116].rstrip() + "..."
    return subtitle


def collect_headings(lines: list[str]) -> list[Heading]:
    headings: list[Heading] = []
    for line in lines:
        match = TITLE_RE.match(line)
        if not match:
            continue
        level = len(match.group(1))
        if level == 1:
            continue
        text = plain_text(match.group(2))
        anchor = slug_for(len(headings) + 1, text)
        headings.append(Heading(level=level, text=text, anchor=anchor))
    return headings


def render_markdown(lines: list[str], headings: list[Heading]) -> str:
    output: list[str] = []
    heading_index = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if stripped == "---":
            output.append('<hr class="divider">')
            index += 1
            continue

        if stripped.startswith("```"):
            language = stripped[3:].strip()
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            output.append(render_code_block(language, code_lines))
            index += 1
            continue

        if stripped == "$$":
            formula_lines: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != "$$":
                formula_lines.append(lines[index])
                index += 1
            formula = html.escape("\n".join(formula_lines))
            output.append(f'<div class="math-block">$$\n{formula}\n$$</div>')
            index += 1
            continue

        match = TITLE_RE.match(line)
        if match:
            level = len(match.group(1))
            text = match.group(2)
            if level == 1:
                output.append(
                    '<div class="source-title"><h2>'
                    f"{convert_inline(text)}</h2></div>"
                )
            else:
                anchor = headings[heading_index].anchor
                heading_index += 1
                output.append(f'<h{level} id="{anchor}">{convert_inline(text)}</h{level}>')
            index += 1
            continue

        if is_table_start(lines, index):
            table_lines = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|") and lines[index].strip().endswith("|"):
                table_lines.append(lines[index])
                index += 1
            output.append(render_table(table_lines))
            continue

        if stripped.startswith(">"):
            quote_lines = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(lines[index].strip())
                index += 1
            output.append(render_blockquote(quote_lines))
            continue

        if list_marker(line):
            list_html, index = render_list_block(lines, index)
            output.append(list_html)
            continue

        if IMAGE_RE.fullmatch(stripped):
            output.append(convert_inline(stripped))
            index += 1
            continue

        paragraph_lines = [line]
        index += 1
        while index < len(lines):
            next_line = lines[index]
            next_stripped = next_line.strip()
            if not next_stripped:
                break
            if (
                next_stripped == "---"
                or next_stripped.startswith("```")
                or next_stripped == "$$"
                or next_stripped.startswith(">")
                or TITLE_RE.match(next_line)
                or list_marker(next_line)
                or is_table_start(lines, index)
                or IMAGE_RE.fullmatch(next_stripped)
            ):
                break
            paragraph_lines.append(next_line)
            index += 1
        output.append(render_paragraph(paragraph_lines))

    return "\n".join(output)


def page_label(path: Path) -> str:
    match = re.search(r"chapter(\d+)", path.stem)
    if match:
        return f"Chapter {int(match.group(1)):02d}"
    return "Progress"


def build_page(path: Path) -> Page:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    title = path.stem
    for line in lines:
        match = TITLE_RE.match(line)
        if match and len(match.group(1)) == 1:
            title = plain_text(match.group(2))
            break
    headings = collect_headings(lines)
    body = render_markdown(lines, headings)
    return Page(
        source=path,
        output=path.with_suffix(".html"),
        title=title,
        subtitle=extract_subtitle(lines),
        label=page_label(path),
        headings=headings,
        body=body,
        word_count=len(re.findall(r"[\w\u4e00-\u9fff]+", text)),
        formula_count=text.count("$$") // 2 + len(re.findall(r"(?<!\\)\$[^$\n]+\$", text)),
        image_count=len(IMAGE_RE.findall(text)),
        mermaid_count=text.count("```mermaid"),
    )


def toc_html(page: Page) -> str:
    if not page.headings:
        return '<div class="toc-title">本页目录</div><a href="#main">正文</a>'
    links = ['<div class="toc-title">本页目录</div>']
    for heading in page.headings:
        if heading.level <= 3:
            links.append(f'<a href="#{heading.anchor}">{html.escape(heading.text)}</a>')
    return "\n".join(links)


def route_html(page: Page) -> str:
    picked = [heading for heading in page.headings if heading.level == 2][:4]
    if not picked:
        picked = page.headings[:4]
    if not picked:
        return ""
    items = []
    for index, heading in enumerate(picked, start=1):
        items.append(
            '<div class="route-step">'
            f'<div class="route-index">{index:02d}</div>'
            '<div class="route-body">'
            f'<strong>{html.escape(heading.text)}</strong>'
            '<span>本节是页面阅读路径中的关键节点.</span>'
            "</div></div>"
        )
    return '<div class="route">' + "\n".join(items) + "</div>"


def nav_link(page: Page | None, rel: str) -> str:
    if page is None:
        return '<span></span>'
    cls = "prev" if rel == "prev" else "next"
    label = "上一页" if rel == "prev" else "下一页"
    return (
        f'<a class="{cls}" href="{page.output.name}">'
        f"<span>{label}</span><strong>{html.escape(page.label)}: {html.escape(page.title)}</strong>"
        "</a>"
    )


def full_html(page: Page, previous_page: Page | None, next_page: Page | None) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{escape_attr(page.title)} HTML 教程">
  <title>{html.escape(page.title)}</title>
  <link rel="stylesheet" href="{CSS_FILE}">
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [["\\\\(", "\\\\)"], ["$", "$"]],
        displayMath: [["\\\\[", "\\\\]"], ["$$", "$$"]]
      }},
      svg: {{ fontCache: "global" }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <script>
    window.addEventListener("DOMContentLoaded", () => {{
      if (window.mermaid) {{
        window.mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});
      }}
    }});
  </script>
</head>
<body>
  <a class="skip-link" href="#main">跳到正文</a>
  <header class="topbar">
    <div class="topbar-inner">
      <a class="brand" href="outline.md">
        <span class="brand-mark">P</span>
        <span>概率统计教程</span>
      </a>
      <nav class="top-actions" aria-label="章节导航">
        <a class="nav-link" href="outline.md">总纲</a>
        <a class="nav-link" href="todo.html">进度</a>
      </nav>
    </div>
  </header>

  <section class="hero">
    <div class="hero-grid">
      <div>
        <p class="eyebrow">{html.escape(page.label)}</p>
        <h1>{html.escape(page.title)}</h1>
        <p class="lead">{html.escape(page.subtitle)}</p>
        <div class="hero-actions">
          <a class="button primary" href="#main">进入正文</a>
          <a class="button" href="#route-panel">阅读路径</a>
        </div>
      </div>
      <aside id="route-panel" class="hero-panel" aria-label="页面概览">
        <div class="panel-head">
          <strong>页面概览</strong>
          <span class="tag">公式 + 图示 + 主线</span>
        </div>
        <div class="stat-grid">
          <div class="stat-card"><span>小节</span><strong>{len(page.headings)}</strong></div>
          <div class="stat-card"><span>公式</span><strong>{page.formula_count}</strong></div>
          <div class="stat-card"><span>配图</span><strong>{page.image_count}</strong></div>
        </div>
        {route_html(page)}
      </aside>
    </div>
  </section>

  <main id="main" class="layout">
    <aside class="toc" aria-label="页面目录">
      {toc_html(page)}
    </aside>
    <div class="content">
      <article class="article">
        <div class="chapter-note">
          <strong>HTML 阅读版</strong>
          本页由同名 Markdown 自动转换生成, 保留原文结构, 并增强目录, 图表, 公式和重点提示的阅读体验.
        </div>
        {page.body}
      </article>
      <nav class="chapter-nav" aria-label="前后页面">
        {nav_link(previous_page, "prev")}
        {nav_link(next_page, "next")}
      </nav>
    </div>
  </main>

  <footer class="footer">
    由 build_probability_statistics_html.py 生成. 样式文件: {CSS_FILE}.
  </footer>
</body>
</html>
"""


def main() -> None:
    pages = [build_page(path) for path in markdown_files()]
    for index, page in enumerate(pages):
        previous_page = pages[index - 1] if index > 0 else None
        next_page = pages[index + 1] if index + 1 < len(pages) else None
        page.output.write_text(full_html(page, previous_page, next_page), encoding="utf-8")
        print(f"generated {page.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
