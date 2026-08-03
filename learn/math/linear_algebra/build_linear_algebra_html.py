from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CSS_FILE = "linear_algebra.css"
THEME_SCRIPT = "../../../docs/html/assets/xdl-theme.js"


@dataclass(frozen=True)
class Section:
    title: str
    kicker: str
    intro: str
    body: str


@dataclass(frozen=True)
class Chapter:
    number: int
    title: str
    subtitle: str
    focus: str
    output: str
    sections: list[Section]


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def slugify(index: int, title: str) -> str:
    stem = re.sub(r"\s+", "-", title.lower())
    stem = re.sub(r"[^a-z0-9\u4e00-\u9fff-]+", "", stem)
    stem = stem.strip("-") or "section"
    return f"{index:02d}-{stem}"


def figure(src: str, alt: str, caption: str) -> str:
    return (
        '<figure>'
        f'<img src="{esc(src)}" alt="{esc(alt)}">'
        f"<figcaption>{esc(caption)}</figcaption>"
        "</figure>"
    )


def figure_grid(*figures: str) -> str:
    return '<div class="figure-grid">' + "".join(figures) + "</div>"


def flow(items: list[tuple[str, str]]) -> str:
    cells = []
    for title, text in items:
        cells.append(
            '<div class="flow-item">'
            f"<strong>{esc(title)}</strong>"
            f"<span>{esc(text)}</span>"
            "</div>"
        )
    return '<div class="flow">' + "\n".join(cells) + "</div>"


def count_formulas(chapter: Chapter) -> int:
    return sum(section.body.count('class="formula"') for section in chapter.sections)


def count_figures(chapter: Chapter) -> int:
    return sum(section.body.count("<figure>") for section in chapter.sections)


def section_anchor(index: int, section: Section) -> str:
    return slugify(index, section.title)


def render_toc(chapter: Chapter) -> str:
    links = ['<div class="toc-title">本页目录</div>']
    for index, section in enumerate(chapter.sections, start=1):
        links.append(f'<a href="#{section_anchor(index, section)}">{esc(section.title)}</a>')
    return "\n".join(links)


def render_route(chapter: Chapter) -> str:
    steps = []
    for index, section in enumerate(chapter.sections[:4], start=1):
        steps.append(
            '<div class="route-step">'
            f'<div class="route-index">{index:02d}</div>'
            '<div class="route-body">'
            f"<strong>{esc(section.title)}</strong>"
            f"<span>{esc(section.intro)}</span>"
            "</div></div>"
        )
    return '<div class="route">' + "\n".join(steps) + "</div>"


def render_sections(chapter: Chapter) -> str:
    rendered = []
    for index, section in enumerate(chapter.sections, start=1):
        anchor = section_anchor(index, section)
        rendered.append(
            f'<section id="{anchor}" class="section">'
            '<div class="section-header">'
            f'<p class="section-kicker">{esc(section.kicker)}</p>'
            f"<h2>{esc(section.title)}</h2>"
            f"<p>{esc(section.intro)}</p>"
            "</div>"
            '<div class="section-body wide">'
            f"{section.body}"
            "</div>"
            "</section>"
        )
    return "\n".join(rendered)


def pager(chapters: list[Chapter], chapter: Chapter) -> str:
    position = chapters.index(chapter)
    previous_chapter = chapters[position - 1] if position > 0 else None
    next_chapter = chapters[position + 1] if position + 1 < len(chapters) else None
    previous_html = (
        f'<a class="prev" href="{previous_chapter.output}"><span>上一章</span><strong>{esc(previous_chapter.title)}</strong></a>'
        if previous_chapter
        else '<a class="prev" href="index.html"><span>课程首页</span><strong>线性代数教程总览</strong></a>'
    )
    next_html = (
        f'<a class="next" href="{next_chapter.output}"><span>下一章</span><strong>{esc(next_chapter.title)}</strong></a>'
        if next_chapter
        else '<a class="next" href="index.html"><span>回到首页</span><strong>从应用回看整套工具链</strong></a>'
    )
    return f'<nav class="pager" aria-label="章节翻页">{previous_html}{next_html}</nav>'


def render_chapter(chapters: list[Chapter], chapter: Chapter) -> str:
    section_count = len(chapter.sections)
    formula_count = count_formulas(chapter)
    figure_count = count_figures(chapter)
    return f"""<!doctype html>
<html lang="zh-CN" data-accent="blue">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{esc(chapter.title)} HTML 教程">
  <title>{esc(chapter.title)}</title>
  <link rel="stylesheet" href="{CSS_FILE}">
  <script defer src="{THEME_SCRIPT}"></script>
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
</head>
<body class="xdl-style-ledger math-doc-page">
  <a class="skip-link" href="#main">跳到正文</a>
  <header class="topbar">
    <div class="topbar-inner">
      <a class="brand" href="index.html">
        <span class="brand-mark">L</span>
        <span>线性代数教程</span>
      </a>
      <nav class="top-actions" aria-label="章节导航">
        <a class="nav-link" href="index.html">首页</a>
        <a class="nav-link" href="outline.md">总纲</a>
      </nav>
    </div>
  </header>

  <section class="hero">
    <div class="hero-grid">
      <div>
        <p class="eyebrow">Session {chapter.number}</p>
        <h1>{esc(chapter.title)}</h1>
        <p class="lead">{esc(chapter.subtitle)}</p>
        <div class="hero-actions">
          <a class="button primary" href="#main">进入正文</a>
          <a class="button" href="#{section_anchor(1, chapter.sections[0])}">看主线</a>
          <a class="button" href="#{section_anchor(len(chapter.sections), chapter.sections[-1])}">实战清单</a>
        </div>
      </div>
      <aside class="hero-panel" aria-label="页面概览">
        <div class="panel-head">
          <strong>本章定位</strong>
          <span class="tag">{esc(chapter.focus)}</span>
        </div>
        <div class="stat-grid">
          <div class="stat-card"><span>小节</span><strong>{section_count}</strong></div>
          <div class="stat-card"><span>公式块</span><strong>{formula_count}</strong></div>
          <div class="stat-card"><span>配图</span><strong>{figure_count}</strong></div>
        </div>
        {render_route(chapter)}
      </aside>
    </div>
  </section>

  <main id="main" class="layout">
    <aside class="toc" aria-label="页面目录">
      {render_toc(chapter)}
    </aside>
    <div class="content">
      {render_sections(chapter)}
      {pager(chapters, chapter)}
    </div>
  </main>

  <footer class="footer-note">
    本页为 XDL learn/math/linear_algebra 的静态 HTML 教程页面. 数学公式由 MathJax 渲染, 配图由本目录 assets 下的 Python 脚本生成.
  </footer>
</body>
</html>
"""


def render_index(chapters: list[Chapter]) -> str:
    cards = []
    for chapter in chapters:
        cards.append(
            f'<a class="chapter-card" href="{chapter.output}">'
            f"<strong>Session {chapter.number}: {esc(chapter.title)}</strong>"
            f"<span>{esc(chapter.focus)}. {esc(chapter.subtitle)}</span>"
            "</a>"
        )
    route = flow(
        [
            ("方程组", "从约束和消元开始, 建立矩阵语言."),
            ("空间", "用向量空间解释方向, 张成, 基和维数."),
            ("变换", "把矩阵看成动作, 研究坐标与不变结构."),
            ("正交", "引入长度, 角度, 投影和最小二乘."),
            ("分解", "把工程计算落到稳定的矩阵工具链."),
        ]
    )
    return f"""<!doctype html>
<html lang="zh-CN" data-accent="blue">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="线性代数 HTML 教程首页">
  <title>线性代数教程</title>
  <link rel="stylesheet" href="{CSS_FILE}">
  <script defer src="{THEME_SCRIPT}"></script>
</head>
<body class="xdl-style-ledger math-doc-page">
  <a class="skip-link" href="#main">跳到正文</a>
  <header class="topbar">
    <div class="topbar-inner">
      <a class="brand" href="index.html">
        <span class="brand-mark">L</span>
        <span>线性代数教程</span>
      </a>
      <nav class="top-actions" aria-label="页面导航">
        <a class="nav-link" href="outline.md">总纲</a>
        <a class="nav-link" href="assets/gen_linear_algebra_figs.py">绘图脚本</a>
      </nav>
    </div>
  </header>

  <section class="hero">
    <div class="hero-grid">
      <div>
        <p class="eyebrow">Linear Algebra</p>
        <h1>线性代数教程</h1>
        <p class="lead">从线性方程组出发, 逐步进入矩阵, 向量空间, 线性变换, 特征结构, 正交投影, 二次型, 矩阵分解和工程应用. 每章都配有图示和物理问题, 重点是把公式背后的结构讲清楚.</p>
        <div class="hero-actions">
          <a class="button primary" href="session1.html">从第一章开始</a>
          <a class="button" href="#chapters">查看全部章节</a>
        </div>
      </div>
      <aside class="hero-panel" aria-label="课程路线">
        <div class="panel-head">
          <strong>课程路线</strong>
          <span class="tag">由浅入深</span>
        </div>
        {route}
      </aside>
    </div>
  </section>

  <main id="main" class="layout">
    <aside class="toc" aria-label="页面目录">
      <div class="toc-title">课程目录</div>
      <a href="#route">学习主线</a>
      <a href="#chapters">全部章节</a>
      <a href="#physics">物理问题线索</a>
      <a href="#workflow">使用方式</a>
    </aside>
    <div class="content">
      <section id="route" class="section">
        <div class="section-header">
          <p class="section-kicker">Map</p>
          <h2>先看整套教程的主线</h2>
          <p>线性代数不是一堆孤立公式, 而是一套处理线性结构的语言.</p>
        </div>
        <div class="section-body wide">
          {route}
          <div class="summary">
            <strong>一句话总纲</strong>
            先用消元解决线性约束, 再用空间和变换解释矩阵, 最后用正交, 特征值和分解服务真实工程计算.
          </div>
        </div>
      </section>

      <section id="chapters" class="section">
        <div class="section-header">
          <p class="section-kicker">Chapters</p>
          <h2>全部章节</h2>
          <p>每章都按 "问题 -> 直觉 -> 定义 -> 计算 -> 物理例子 -> 清单" 组织.</p>
        </div>
        <div class="section-body wide">
          <div class="chapter-grid">
            {''.join(cards)}
          </div>
        </div>
      </section>

      <section id="physics" class="section">
        <div class="section-header">
          <p class="section-kicker">Physics Thread</p>
          <h2>贯穿全课的物理问题线索</h2>
          <p>每章都把数学概念放回一个真实可解释的物理或工程场景.</p>
        </div>
        <div class="section-body">
          <div class="grid">
            <div class="card"><strong>静力平衡</strong>第一章把结点受力写成线性方程组, 解释消元为什么是处理约束.</div>
            <div class="card"><strong>电路与网络</strong>第二章用 KCL/KVL 说明矩阵和逆矩阵如何把输入电压映射成电流.</div>
            <div class="card"><strong>坐标系和传感器</strong>第四, 五章解释分量, 基变换, 机器人坐标和相机位姿.</div>
            <div class="card"><strong>振动, 能量, 稳定性</strong>第六, 八, 九章把特征值, 二次型和分解连接到弹簧系统, 模态和数值求解.</div>
          </div>
        </div>
      </section>

      <section id="workflow" class="section">
        <div class="section-header">
          <p class="section-kicker">Usage</p>
          <h2>建议阅读方式</h2>
          <p>先完整读一遍主线, 再回到公式和例题处做手算.</p>
        </div>
        <div class="section-body">
          <ol>
            <li>先读每章的 "全章主线", 知道本章解决什么问题.</li>
            <li>遇到公式先看它的对象和维度, 不要马上代数展开.</li>
            <li>看完物理例子后, 尝试把同类问题写成矩阵形式.</li>
            <li>最后用实战清单检查: 条件是否满足, 量纲是否一致, 结论是否能解释原问题.</li>
          </ol>
        </div>
      </section>
    </div>
  </main>
</body>
</html>
"""


def build_chapters() -> list[Chapter]:
    chapters = [
        Chapter(
            number=1,
            title="第一章: 线性方程组与消元法",
            subtitle="从多个线性约束能否同时满足开始, 学会用增广矩阵和初等行变换把问题化成容易读解的标准形.",
            focus="约束与求解",
            output="session1.html",
            sections=[
                Section(
                    title="0. 全章主线",
                    kicker="Route",
                    intro="先把方程组看成约束, 再把消元看成保持解集不变的整理过程.",
                    body=flow(
                        [
                            ("真实问题", "多个条件同时成立."),
                            ("方程组", "把条件写成线性等式."),
                            ("增广矩阵", "只保留系数和常数项."),
                            ("行变换", "保持解集不变地整理约束."),
                            ("读解", "从阶梯形判断解的个数."),
                        ]
                    )
                    + r"""
                    <div class="summary"><strong>本章核心</strong>消元法不是技巧表演, 它是在不改变解集的前提下, 把一组复杂约束改写成更透明的约束.</div>
                    """,
                ),
                Section(
                    title="1. 线性方程组先代表什么",
                    kicker="Problem",
                    intro="每一条线性方程都是一个限制条件, 解就是所有限制同时满足的位置.",
                    body=r"""
                    <p>二元方程 $a x+b y=c$ 在平面中是一条直线. 三元方程 $a x+b y+c z=d$ 在空间中是一个平面. 多个方程放在一起, 就是在问这些直线或平面有没有公共交集.</p>
                    <div class="formula">$$
                    \begin{cases}
                    a_{11}x_1+a_{12}x_2+\cdots+a_{1n}x_n=b_1\\
                    a_{21}x_1+a_{22}x_2+\cdots+a_{2n}x_n=b_2\\
                    \cdots\\
                    a_{m1}x_1+a_{m2}x_2+\cdots+a_{mn}x_n=b_m
                    \end{cases}
                    $$</div>
                    <div class="grid three">
                      <div class="card"><strong>唯一解</strong>约束刚好交于一个点.</div>
                      <div class="card"><strong>无解</strong>约束彼此冲突, 没有公共位置.</div>
                      <div class="card"><strong>无穷多解</strong>约束有冗余, 公共交集是一条线, 一个平面或更高维对象.</div>
                    </div>
                    """,
                ),
                Section(
                    title="2. 增广矩阵与初等行变换",
                    kicker="Algebra",
                    intro="增广矩阵把方程组压缩成数字表, 行变换对应对方程做等价改写.",
                    body=r"""
                    <p>把未知数按固定顺序排好, 方程组就可以写成 $Ax=b$. 计算时常把系数矩阵和右端向量放在一起:</p>
                    <div class="formula">$$
                    \left[
                    \begin{array}{cccc|c}
                    a_{11}&a_{12}&\cdots&a_{1n}&b_1\\
                    a_{21}&a_{22}&\cdots&a_{2n}&b_2\\
                    \vdots&\vdots&&\vdots&\vdots\\
                    a_{m1}&a_{m2}&\cdots&a_{mn}&b_m
                    \end{array}
                    \right]
                    $$</div>
                    <div class="table-wrap"><table><thead><tr><th>行操作</th><th>方程视角</th><th>为什么可用</th></tr></thead><tbody>
                    <tr><td>交换两行</td><td>交换两条方程顺序</td><td>约束集合没有变化</td></tr>
                    <tr><td>某行乘非零数</td><td>等式两边同乘非零数</td><td>同一条约束的等价写法</td></tr>
                    <tr><td>一行加另一行的倍数</td><td>用约束的线性组合替换约束</td><td>新旧方程组互相推出</td></tr>
                    </tbody></table></div>
                    <div class="note"><strong>边界</strong>初等行变换保持解集不变. 初学阶段不要随意做列变换, 因为列变换会改变未知量的含义.</div>
                    """,
                ),
                Section(
                    title="3. 高斯消元怎样读解",
                    kicker="Elimination",
                    intro="消元的目标是得到阶梯形, 让主元, 自由变量和矛盾行一眼可见.",
                    body=r"""
                    <p>阶梯形矩阵里, 每个非零行最左侧的非零元素叫主元. 主元所在列对应主变量, 其他未知数通常是自由变量.</p>
                    <div class="formula">$$
                    \left[
                    \begin{array}{ccc|c}
                    1&2&-1&3\\
                    0&1&4&5\\
                    0&0&0&0
                    \end{array}
                    \right]
                    \Rightarrow
                    x_3=t,\quad x_2=5-4t,\quad x_1=-7+9t
                    $$</div>
                    <ol>
                      <li>先找主元列, 判断哪些变量被约束住.</li>
                      <li>再看是否出现 $0=非零数$ 的矛盾行.</li>
                      <li>若没有矛盾行, 用自由变量参数化全部解.</li>
                    </ol>
                    <div class="warning"><strong>常见误区</strong>出现一整行零不代表无解, 它通常表示这条方程是冗余约束. 真正的无解信号是 $[0\ 0\ \cdots\ 0\ |\ c]$ 且 $c\ne0$.</div>
                    """,
                ),
                Section(
                    title="4. 几何图像: 交点就是解",
                    kicker="Geometry",
                    intro="图像能帮助我们理解解的个数, 也能提醒代数运算背后是什么对象.",
                    body=figure_grid(
                        figure(
                            "assets/la_ch01_line_intersection.png",
                            "两条线性约束相交于唯一解",
                            "两个二元一次方程对应两条直线. 交点就是两个约束同时成立的唯一解.",
                        ),
                        figure(
                            "assets/la_ch01_force_equilibrium.png",
                            "结点受力平衡转化为线性方程组",
                            "静力学中结点不动意味着横向和纵向合力都为零. 两个平衡方程可以求出两根绳子的张力.",
                        ),
                    )
                    + r"""
                    <p>如果两条直线平行且不同, 方程组无解. 如果两条直线重合, 方程组有无穷多解. 在三维中, 同样的逻辑会变成平面之间的相交, 平行和重合.</p>
                    """,
                ),
                Section(
                    title="5. 物理问题: 受力平衡就是线性约束",
                    kicker="Physics",
                    intro="静力平衡给出本章最自然的物理模型: 总力为零.",
                    body=r"""
                    <div class="physics"><strong>问题</strong>一个重物挂在两根斜绳连接的结点上. 若结点保持静止, 两根绳子的张力 $T_1,T_2$ 必须让水平和竖直方向的合力都为零.</div>
                    <div class="formula">$$
                    \begin{cases}
                    T_1\cos\alpha - T_2\cos\beta = 0\\
                    T_1\sin\alpha + T_2\sin\beta = mg
                    \end{cases}
                    $$</div>
                    <p>这不是把物理问题"套公式", 而是把守恒或平衡规律翻译成线性约束. 当结构里有更多结点和杆件时, 方程数量会增加, 但核心仍是 $Ax=b$.</p>
                    """,
                ),
                Section(
                    title="6. 实战清单与易错点",
                    kicker="Checklist",
                    intro="做方程组题时, 先判断结构, 再计算, 最后回到原问题解释.",
                    body=r"""
                    <div class="grid">
                      <div class="card"><strong>实战清单</strong><ol><li>未知量顺序是否固定?</li><li>每行代表哪条约束?</li><li>行变换是否只对方程做等价改写?</li><li>主元列和自由变量是否读全?</li><li>结论能否解释原问题?</li></ol></div>
                      <div class="card"><strong>高频错误</strong><ol><li>把列变换当成普通消元步骤.</li><li>无穷多解时漏写参数.</li><li>看到零行就判断无解.</li><li>回代时把主变量和自由变量混掉.</li></ol></div>
                    </div>
                    <div class="summary"><strong>记住</strong>线性方程组的本质是约束系统. 消元法的本质是把约束改写到更容易读解的形态.</div>
                    """,
                ),
            ],
        ),
        Chapter(
            number=2,
            title="第二章: 矩阵运算与逆矩阵",
            subtitle="把矩阵从数字表升级为线性规则, 理解矩阵乘法是变换复合, 逆矩阵是把变换倒回去.",
            focus="矩阵作为变换",
            output="session2.html",
            sections=[
                Section(
                    title="0. 全章主线",
                    kicker="Route",
                    intro="矩阵运算看似规则多, 主线其实是变换如何组合和反解.",
                    body=flow(
                        [
                            ("向量输入", "状态或坐标写成列向量."),
                            ("矩阵作用", "矩阵把输入变成输出."),
                            ("乘法", "多个变换连续执行."),
                            ("单位矩阵", "什么都不改变的基准."),
                            ("逆矩阵", "把输出还原成输入."),
                        ]
                    )
                    + r"""<div class="summary"><strong>本章核心</strong>矩阵乘法不交换, 因为先旋转再拉伸和先拉伸再旋转通常不是同一个动作.</div>""",
                ),
                Section(
                    title="1. 矩阵不是一张表",
                    kicker="Meaning",
                    intro="矩阵最重要的解释是线性映射: 输入一个向量, 输出另一个向量.",
                    body=r"""
                    <p>当 $A$ 是 $m\times n$ 矩阵时, 它接受 $n$ 维向量并输出 $m$ 维向量:</p>
                    <div class="formula">$$
                    A:\mathbb{R}^n\to\mathbb{R}^m,\quad x\mapsto Ax
                    $$</div>
                    <p>第 $j$ 列 $a_j$ 可以理解为标准基向量 $e_j$ 被矩阵送到的位置. 因此</p>
                    <div class="formula">$$
                    Ax=x_1a_1+x_2a_2+\cdots+x_na_n
                    $$</div>
                    <p>矩阵乘向量的本质是用输入坐标对矩阵列向量做线性组合.</p>
                    """,
                ),
                Section(
                    title="2. 乘法为什么是行乘列",
                    kicker="Multiplication",
                    intro="矩阵乘法来自变换复合, 不是任意定义出来的记号游戏.",
                    body=r"""
                    <p>若 $B$ 先把 $x$ 变成 $Bx$, 再由 $A$ 作用, 总效果是 $A(Bx)=(AB)x$. 这要求乘积 $AB$ 的每一列等于 $A$ 作用在 $B$ 的对应列上.</p>
                    <div class="formula">$$
                    AB=\left[Ab_1\ Ab_2\ \cdots\ Ab_k\right]
                    $$</div>
                    <div class="note"><strong>维度检查</strong>$A_{m\times n}B_{n\times k}$ 才能相乘, 结果是 $m\times k$. 中间维度代表前一个输出必须能喂给后一个输入.</div>
                    """,
                ),
                Section(
                    title="3. 逆矩阵的意义",
                    kicker="Inverse",
                    intro="可逆意味着变换没有丢信息, 每个输出都能唯一追溯回输入.",
                    body=r"""
                    <div class="formula">$$
                    A^{-1}A=AA^{-1}=I,\quad Ax=b\Rightarrow x=A^{-1}b
                    $$</div>
                    <p>但实际计算中, 不应该把"求逆再相乘"当成默认数值方法. 求解线性系统通常更适合用消元, LU 或 QR. 逆矩阵更重要的价值是表达结构: 这个变换是否能倒回去.</p>
                    <div class="grid three">
                      <div class="card"><strong>可逆</strong>旋转, 非零缩放, 不把空间压扁的剪切.</div>
                      <div class="card"><strong>不可逆</strong>投影, 把平面压成线, 把多维信息合并掉.</div>
                      <div class="card"><strong>单位矩阵</strong>所有向量都保持原样, 是变换复合中的中性元素.</div>
                    </div>
                    """,
                ),
                Section(
                    title="4. 几何图像: 网格怎样被矩阵移动",
                    kicker="Geometry",
                    intro="观察网格能快速看出矩阵的旋转, 拉伸, 剪切和压缩效果.",
                    body=figure_grid(
                        figure(
                            "assets/la_ch02_grid_transform.png",
                            "矩阵作用在平面网格上的效果",
                            "矩阵把标准基送到两列向量的位置, 整个网格随之被线性地拉伸和剪切.",
                        ),
                        figure(
                            "assets/la_ch02_circuit_linear_system.png",
                            "简单电路方程写成矩阵形式",
                            "电路中的 KCL/KVL 会产生线性方程组. 网络矩阵可逆时, 电压输入能唯一决定电流.",
                        ),
                    )
                    + r"""
                    <p>图中的网格线仍保持直线和平行, 这是线性变换的重要特征. 如果一个变换会把直线弯成曲线, 它就不是线性代数这一章研究的对象.</p>
                    """,
                ),
                Section(
                    title="5. 物理问题: 电路网络中的矩阵",
                    kicker="Physics",
                    intro="线性电阻网络把电压, 电流和电阻连接成矩阵方程.",
                    body=r"""
                    <div class="physics"><strong>问题</strong>在由线性电阻组成的电路里, Kirchhoff 电流定律和电压定律给出一组线性约束. 未知量可以是支路电流, 节点电压或网孔电流.</div>
                    <div class="formula">$$
                    Gv=i,\quad G_{ij}\ \text{由电导和连接关系决定}
                    $$</div>
                    <p>若矩阵 $G$ 在选定边界条件下可逆, 给定注入电流 $i$ 就能唯一求出节点电压 $v$. 若不可逆, 常见原因是参考电位没有固定, 或网络中存在无法区分的自由漂移模式.</p>
                    """,
                ),
                Section(
                    title="6. 实战清单与易错点",
                    kicker="Checklist",
                    intro="矩阵题优先检查维度, 顺序和信息是否丢失.",
                    body=r"""
                    <div class="grid">
                      <div class="card"><strong>实战清单</strong><ol><li>每个矩阵的输入维度和输出维度是什么?</li><li>乘法顺序是否对应真实执行顺序?</li><li>是否需要可逆, 还是只需要求解 $Ax=b$?</li><li>特殊矩阵的性质是否满足条件?</li></ol></div>
                      <div class="card"><strong>高频错误</strong><ol><li>把矩阵乘法当成对应元素相乘.</li><li>默认 $AB=BA$.</li><li>把有解和可逆混为一谈.</li><li>忘记检查 $A$ 是否方阵就谈逆矩阵.</li></ol></div>
                    </div>
                    <div class="summary"><strong>记住</strong>矩阵是线性规则, 乘法是规则复合, 逆矩阵是规则可反向执行.</div>
                    """,
                ),
            ],
        ),
        Chapter(
            number=3,
            title="第三章: 行列式",
            subtitle="把行列式理解为有向面积或体积的缩放因子, 用它判断线性变换是否把空间压扁.",
            focus="面积体积缩放",
            output="session3.html",
            sections=[
                Section(
                    title="0. 全章主线",
                    kicker="Route",
                    intro="行列式不是为了机械展开, 而是为了度量线性变换对空间体积的影响.",
                    body=flow(
                        [
                            ("二维面积", "两列向量围成平行四边形."),
                            ("有向缩放", "符号记录方向是否翻转."),
                            ("性质", "行变换如何改变面积."),
                            ("可逆性", "体积不为零才没压扁."),
                            ("应用", "从坐标变换到物理量守恒."),
                        ]
                    ),
                ),
                Section(
                    title="1. 从二阶行列式开始",
                    kicker="Definition",
                    intro="二阶行列式就是两个列向量围成的有向面积.",
                    body=r"""
                    <div class="formula">$$
                    \det\begin{bmatrix}a&b\\c&d\end{bmatrix}=ad-bc
                    $$</div>
                    <p>如果把矩阵列向量看成 $u=(a,c)^T$ 和 $v=(b,d)^T$, 那么 $|\det A|$ 就是它们张成的平行四边形面积. 符号则表示方向是否被翻转.</p>
                    <div class="note"><strong>直觉</strong>两列越接近共线, 平行四边形越瘦, 行列式绝对值越小. 完全共线时面积为零, 变换不可逆.</div>
                    """,
                ),
                Section(
                    title="2. 行列式的核心性质",
                    kicker="Properties",
                    intro="行列式性质都可以从有向体积的变化来理解.",
                    body=r"""
                    <div class="table-wrap"><table><thead><tr><th>操作</th><th>行列式变化</th><th>几何解释</th></tr></thead><tbody>
                    <tr><td>交换两行</td><td>变号</td><td>空间方向翻转</td></tr>
                    <tr><td>某行乘 $k$</td><td>乘以 $k$</td><td>一个方向伸缩 $k$ 倍</td></tr>
                    <tr><td>一行加另一行倍数</td><td>不变</td><td>剪切不改变面积或体积</td></tr>
                    <tr><td>两行相同或相关</td><td>为 0</td><td>体积被压扁</td></tr>
                    </tbody></table></div>
                    <div class="formula">$$
                    \det(AB)=\det(A)\det(B)
                    $$</div>
                    <p>复合变换的体积缩放因子等于每一步缩放因子的乘积.</p>
                    """,
                ),
                Section(
                    title="3. 行列式和可逆性",
                    kicker="Invertibility",
                    intro="方阵可逆的一个关键判据是行列式不为零.",
                    body=r"""
                    <div class="formula">$$
                    A\ \text{可逆}\quad\Longleftrightarrow\quad \det A\ne0
                    $$</div>
                    <p>如果 $\det A=0$, 说明某些非零方向被压到低维空间里. 信息已经丢失, 就不可能存在真正的逆变换把所有输出唯一还原.</p>
                    <div class="warning"><strong>边界</strong>行列式只对方阵定义. 对非方阵问题, 可逆性要改成单射, 满射, 秩, 伪逆等语言.</div>
                    """,
                ),
                Section(
                    title="4. 几何图像: 面积和塌缩",
                    kicker="Geometry",
                    intro="把列向量画出来, 行列式的大小和零值会很直观.",
                    body=figure_grid(
                        figure(
                            "assets/la_ch03_determinant_area.png",
                            "行列式对应平行四边形面积",
                            "二维中, 矩阵两列向量张成的平行四边形面积就是行列式绝对值.",
                        ),
                        figure(
                            "assets/la_ch03_determinant_collapse.png",
                            "行列式为零时面积塌缩",
                            "当两列向量线性相关时, 平行四边形退化成线段, 变换把平面压扁.",
                        ),
                    ),
                ),
                Section(
                    title="5. 物理问题: 坐标变换中的体积因子",
                    kicker="Physics",
                    intro="物理积分换元时, 行列式给出面积或体积元素的缩放.",
                    body=r"""
                    <div class="physics"><strong>问题</strong>计算一个非规则区域中的质量, 常把坐标从 $(u,v)$ 变换到 $(x,y)$. 面密度积分必须乘上面积元素缩放因子.</div>
                    <div class="formula">$$
                    \iint_R \rho(x,y)\,dx\,dy
                    =
                    \iint_S \rho(x(u,v),y(u,v))
                    \left|\det\frac{\partial(x,y)}{\partial(u,v)}\right|\,du\,dv
                    $$</div>
                    <p>在线性变换中, Jacobian 就是矩阵本身. 因此行列式直接告诉我们每个微小面积块被放大或缩小了多少.</p>
                    """,
                ),
                Section(
                    title="6. 实战清单与易错点",
                    kicker="Checklist",
                    intro="行列式计算要服务结构判断, 不要陷入无意义展开.",
                    body=r"""
                    <div class="grid">
                      <div class="card"><strong>实战清单</strong><ol><li>矩阵是否方阵?</li><li>能否用行变换化成三角形?</li><li>每次行操作对行列式影响是否记录?</li><li>结果是否能解释面积, 体积或可逆性?</li></ol></div>
                      <div class="card"><strong>高频错误</strong><ol><li>交换行忘记变号.</li><li>把矩阵乘法和行列式乘法混淆.</li><li>把 $\det A$ 当成矩阵本身的大小.</li><li>余子式符号因子 $(-1)^{i+j}$ 漏掉.</li></ol></div>
                    </div>
                    <div class="summary"><strong>记住</strong>行列式是有向体积缩放因子. 为零表示空间被压扁, 非零表示方阵没有丢掉维度.</div>
                    """,
                ),
            ],
        ),
        Chapter(
            number=4,
            title="第四章: 向量与向量空间",
            subtitle="从具体箭头走向抽象空间, 理解线性组合, 张成, 线性无关, 基, 维数和四个基本子空间.",
            focus="空间与维数",
            output="session4.html",
            sections=[
                Section(
                    title="0. 全章主线",
                    kicker="Route",
                    intro="向量空间回答哪些对象可以加, 可以数乘, 可以用坐标描述.",
                    body=flow(
                        [
                            ("向量", "有加法和数乘的对象."),
                            ("线性组合", "用已有方向拼出新对象."),
                            ("张成", "所有能拼出的集合."),
                            ("无关", "没有冗余方向."),
                            ("基和维数", "最小且完整的坐标系统."),
                        ]
                    ),
                ),
                Section(
                    title="1. 向量可以不只是箭头",
                    kicker="Object",
                    intro="只要满足加法和数乘规则, 函数, 多项式, 信号片段也可以成为向量.",
                    body=r"""
                    <p>初学时向量常被画成箭头, 但真正重要的是线性结构. 例如一段离散传感器信号可以写成 $x=(x_1,\dots,x_n)$, 一个多项式可以写成系数向量, 一张灰度图可以拉平成很长的向量.</p>
                    <div class="formula">$$
                    c_1v_1+c_2v_2+\cdots+c_kv_k
                    $$</div>
                    <p>这个表达式叫线性组合. 线性代数的大量问题都可以转化为: 目标对象能否由一组已知对象线性组合出来?</p>
                    """,
                ),
                Section(
                    title="2. 张成, 无关, 基",
                    kicker="Basis",
                    intro="张成表示覆盖能力, 无关表示没有冗余, 基要求两者同时成立.",
                    body=r"""
                    <div class="grid three">
                      <div class="card"><strong>张成</strong>$\operatorname{span}\{v_1,\dots,v_k\}$ 是所有线性组合的集合.</div>
                      <div class="card"><strong>线性无关</strong>只有所有系数都为零时, 线性组合才等于零向量.</div>
                      <div class="card"><strong>基</strong>既能张成整个空间, 又线性无关. 每个向量坐标唯一.</div>
                    </div>
                    <div class="formula">$$
                    c_1v_1+\cdots+c_kv_k=0\Rightarrow c_1=\cdots=c_k=0
                    $$</div>
                    <p>基不是"随便挑几个向量", 而是一个没有冗余且表达能力完整的方向系统.</p>
                    """,
                ),
                Section(
                    title="3. 四个基本子空间",
                    kicker="Subspaces",
                    intro="一个矩阵天然带着列空间, 零空间, 行空间和左零空间.",
                    body=r"""
                    <div class="table-wrap"><table><thead><tr><th>子空间</th><th>定义</th><th>回答的问题</th></tr></thead><tbody>
                    <tr><td>列空间</td><td>$C(A)=\{Ax:x\in\mathbb{R}^n\}$</td><td>哪些 $b$ 可以被 $Ax$ 达到?</td></tr>
                    <tr><td>零空间</td><td>$N(A)=\{x:Ax=0\}$</td><td>哪些输入会被压成零?</td></tr>
                    <tr><td>行空间</td><td>$C(A^T)$</td><td>方程真正约束了哪些输入方向?</td></tr>
                    <tr><td>左零空间</td><td>$N(A^T)$</td><td>输出空间里哪些方向永远到不了?</td></tr>
                    </tbody></table></div>
                    <div class="formula">$$
                    \dim N(A)+\operatorname{rank}(A)=n
                    $$</div>
                    <p>这就是秩-零度定理的初步形态: 输入维数被分成"能被看见的方向"和"会消失的方向".</p>
                    """,
                ),
                Section(
                    title="4. 几何图像: 张成和分量",
                    kicker="Geometry",
                    intro="把向量画出来, 冗余方向和基的意义会自然出现.",
                    body=figure_grid(
                        figure(
                            "assets/la_ch04_span_basis.png",
                            "一个方向张成线, 两个无关方向张成平面",
                            "线性组合的几何意义是沿若干方向伸缩后相加. 独立方向越多, 可到达的空间维度越高.",
                        ),
                        figure(
                            "assets/la_ch04_incline_components.png",
                            "斜面上的重力分解",
                            "同一个重力向量可以换到沿斜面和垂直斜面的基下表示, 这就是分量和基的物理意义.",
                        ),
                    ),
                ),
                Section(
                    title="5. 物理问题: 斜面分解和约束方向",
                    kicker="Physics",
                    intro="受力分解本质上是在选择适合问题的基.",
                    body=r"""
                    <div class="physics"><strong>问题</strong>物体在斜面上运动时, 把重力 $mg$ 分解成沿斜面方向和法向方向, 会比用水平竖直坐标更直接.</div>
                    <div class="formula">$$
                    mg_{\parallel}=mg\sin\theta,\quad mg_{\perp}=mg\cos\theta
                    $$</div>
                    <p>这说明"坐标"不是向量本身. 坐标只是向量在某组基下的数字表示. 选对基, 问题会明显变简单.</p>
                    """,
                ),
                Section(
                    title="6. 实战清单与易错点",
                    kicker="Checklist",
                    intro="空间题要同时检查生成能力和是否冗余.",
                    body=r"""
                    <div class="grid">
                      <div class="card"><strong>实战清单</strong><ol><li>对象是否在同一个向量空间里?</li><li>要证明的是张成, 无关, 还是基?</li><li>子空间是否包含零向量并对加法, 数乘封闭?</li><li>秩和零空间维数是否对应未知数个数?</li></ol></div>
                      <div class="card"><strong>高频错误</strong><ol><li>把向量个数当成维数.</li><li>只证明张成就说是基.</li><li>把列空间和零空间放在同一个空间里比较.</li><li>忘记零向量不能加入线性无关组.</li></ol></div>
                    </div>
                    <div class="summary"><strong>记住</strong>向量空间研究的是可线性组合的对象. 基给出最小完整坐标系统, 维数表示需要多少个独立方向.</div>
                    """,
                ),
            ],
        ),
        Chapter(
            number=5,
            title="第五章: 线性变换与坐标变换",
            subtitle="把矩阵看成线性变换在某组基下的表示, 分清对象本身和坐标数字, 理解核, 像和相似.",
            focus="变换与坐标",
            output="session5.html",
            sections=[
                Section(
                    title="0. 全章主线",
                    kicker="Route",
                    intro="线性变换是动作, 矩阵是这个动作在某组基下的坐标表达.",
                    body=flow(
                        [
                            ("变换", "把一个向量送到另一个向量."),
                            ("线性", "保持加法和数乘."),
                            ("矩阵表示", "由基向量的像决定."),
                            ("核与像", "研究消失方向和可达方向."),
                            ("换基", "同一动作换一种坐标写法."),
                        ]
                    ),
                ),
                Section(
                    title="1. 线性变换的两个条件",
                    kicker="Definition",
                    intro="线性变换必须保持线性组合结构.",
                    body=r"""
                    <div class="formula">$$
                    T(u+v)=T(u)+T(v),\quad T(cv)=cT(v)
                    $$</div>
                    <p>这两个条件合起来意味着 $T(c_1v_1+\cdots+c_kv_k)=c_1T(v_1)+\cdots+c_kT(v_k)$. 所以只要知道一组基向量被送到哪里, 就能确定整个线性变换.</p>
                    <div class="note"><strong>判断技巧</strong>如果一个变换包含平移 $x\mapsto Ax+t$ 且 $t\ne0$, 它一般不是线性变换, 因为零向量没有被送到零向量.</div>
                    """,
                ),
                Section(
                    title="2. 核, 像和秩-零度",
                    kicker="Structure",
                    intro="核告诉我们哪些输入被压没, 像告诉我们哪些输出能达到.",
                    body=r"""
                    <div class="formula">$$
                    \ker T=\{v:T(v)=0\},\quad \operatorname{im}T=\{T(v):v\in V\}
                    $$</div>
                    <p>核越大, 说明变换丢掉的信息越多. 像越小, 说明输出空间中可达到的范围越受限制.</p>
                    <div class="formula">$$
                    \dim V=\dim\ker T+\dim\operatorname{im}T
                    $$</div>
                    <p>这条公式把输入空间分解成"消失的自由度"和"真正进入输出的自由度".</p>
                    """,
                ),
                Section(
                    title="3. 坐标变换不是向量变了",
                    kicker="Coordinates",
                    intro="同一个向量在不同基下有不同坐标, 但几何对象本身没有改变.",
                    body=r"""
                    <p>若 $P=[b_1\ b_2\ \cdots\ b_n]$ 的列是新基向量在旧坐标下的表示, 那么新坐标 $[x]_B$ 和旧坐标 $[x]_E$ 满足:</p>
                    <div class="formula">$$
                    [x]_E=P[x]_B,\quad [x]_B=P^{-1}[x]_E
                    $$</div>
                    <p>同一线性变换 $T$ 在不同基下的矩阵会改变. 若旧基下矩阵是 $A$, 新基下矩阵是 $B$, 常见关系为:</p>
                    <div class="formula">$$
                    B=P^{-1}AP
                    $$</div>
                    <p>这就是相似矩阵的来源: 数字矩阵变了, 但它们描述的是同一个线性变换.</p>
                    """,
                ),
                Section(
                    title="4. 几何图像: 同一对象, 不同坐标",
                    kicker="Geometry",
                    intro="坐标系换了, 向量和传感器位置的几何关系没有换.",
                    body=figure_grid(
                        figure(
                            "assets/la_ch05_coordinate_change.png",
                            "同一个向量在不同基下的坐标",
                            "红色向量是同一个几何对象. 灰色标准基和旋转后的新基会给它不同的坐标数字.",
                        ),
                        figure(
                            "assets/la_ch05_robot_frame.png",
                            "机器人世界坐标和相机坐标",
                            "机器人和相机标定中常用 $p_{world}=Rp_{camera}+t$. 旋转矩阵描述坐标轴方向, 平移描述原点位置.",
                        ),
                    ),
                ),
                Section(
                    title="5. 物理问题: 刚体姿态和坐标框架",
                    kicker="Physics",
                    intro="测量系统里, 物理点不变, 变的是观察它的坐标框架.",
                    body=r"""
                    <div class="physics"><strong>问题</strong>相机看到一个点的坐标 $p_c$, 机器人控制系统需要世界坐标 $p_w$. 如果两套坐标轴之间只有旋转和平移, 就可以写成刚体变换.</div>
                    <div class="formula">$$
                    p_w=Rp_c+t,\quad R^TR=I
                    $$</div>
                    <p>严格说 $p\mapsto Rp+t$ 是仿射变换, 不是线性变换. 但齐次坐标可以把它写成矩阵乘法, 这也是图形学和机器人学里常见的工程做法.</p>
                    """,
                ),
                Section(
                    title="6. 实战清单与易错点",
                    kicker="Checklist",
                    intro="换基题最大的风险是把对象和坐标混在一起.",
                    body=r"""
                    <div class="grid">
                      <div class="card"><strong>实战清单</strong><ol><li>当前矩阵表示哪个变换?</li><li>基向量是按列还是按行放入换基矩阵?</li><li>公式是在旧坐标转新坐标, 还是新坐标转旧坐标?</li><li>核和像分别属于哪个空间?</li></ol></div>
                      <div class="card"><strong>高频错误</strong><ol><li>把 $P^{-1}AP$ 写成 $PAP^{-1}$.</li><li>把坐标数字变化理解成向量本身变化.</li><li>忘记平移不是线性变换.</li><li>核空间和解空间的关系说不清.</li></ol></div>
                    </div>
                    <div class="summary"><strong>记住</strong>线性变换是结构保持的动作. 矩阵是动作在某组基下的坐标说明书.</div>
                    """,
                ),
            ],
        ),
        Chapter(
            number=6,
            title="第六章: 特征值, 特征向量与对角化",
            subtitle="寻找变换中方向不变的特殊向量, 用合适的基把复杂耦合变成按方向独立缩放.",
            focus="不变方向",
            output="session6.html",
            sections=[
                Section(
                    title="0. 全章主线",
                    kicker="Route",
                    intro="特征向量揭示线性变换最自然的方向, 对角化则是在找最适合变换的坐标系.",
                    body=flow(
                        [
                            ("不变方向", "方向不变, 只缩放."),
                            ("特征方程", "求出可能的缩放倍数."),
                            ("特征空间", "同一特征值下的全部方向."),
                            ("对角化", "凑齐一组特征向量基."),
                            ("应用", "递推, 振动, 稳定性."),
                        ]
                    ),
                ),
                Section(
                    title="1. 特征值和特征向量",
                    kicker="Definition",
                    intro="特征向量是变换后仍落在原方向上的非零向量.",
                    body=r"""
                    <div class="formula">$$
                    Av=\lambda v,\quad v\ne0
                    $$</div>
                    <p>$\lambda$ 是伸缩倍数, $v$ 是不变方向. 这个定义不允许 $v=0$, 因为零向量在任何变换下都无法给出方向信息.</p>
                    <div class="formula">$$
                    (A-\lambda I)v=0\Rightarrow \det(A-\lambda I)=0
                    $$</div>
                    <p>特征多项式给出特征值候选, 再代回零空间求特征向量.</p>
                    """,
                ),
                Section(
                    title="2. 对角化为什么有用",
                    kicker="Diagonalization",
                    intro="如果能用特征向量作为基, 矩阵就会变成对角矩阵.",
                    body=r"""
                    <div class="formula">$$
                    A=PDP^{-1},\quad D=\operatorname{diag}(\lambda_1,\dots,\lambda_n)
                    $$</div>
                    <p>这意味着 $A$ 的复杂作用可以分成三步: 先换到特征向量坐标, 再按坐标独立缩放, 最后换回原坐标.</p>
                    <div class="formula">$$
                    A^k=PD^kP^{-1}
                    $$</div>
                    <p>矩阵幂, 线性递推和线性微分方程因此会大幅简化.</p>
                    """,
                ),
                Section(
                    title="3. 什么时候不能对角化",
                    kicker="Condition",
                    intro="有特征值不等于一定能对角化, 关键是能否凑齐一组特征向量基.",
                    body=r"""
                    <p>$n\times n$ 矩阵可对角化的核心条件是存在 $n$ 个线性无关的特征向量. 代数重数告诉一个特征值在特征多项式中重复几次, 几何重数告诉对应特征空间维数.</p>
                    <div class="formula">$$
                    1\le \dim E_\lambda \le \text{代数重数}(\lambda)
                    $$</div>
                    <div class="warning"><strong>常见误区</strong>重复特征值不是问题本身. 真正的问题是重复特征值下是否有足够多的独立特征向量.</div>
                    """,
                ),
                Section(
                    title="4. 几何图像: 不变方向和振动模态",
                    kicker="Geometry",
                    intro="特征向量在几何上是不变方向, 在物理上常对应系统的自然模态.",
                    body=figure_grid(
                        figure(
                            "assets/la_ch06_eigenvectors.png",
                            "特征向量在变换后保持方向",
                            "蓝色和绿色方向被矩阵作用后仍沿原方向伸缩. 普通向量通常会改变方向.",
                        ),
                        figure(
                            "assets/la_ch06_vibration_modes.png",
                            "双质量弹簧系统的两种振动模态",
                            "耦合振动系统的特征向量对应自然振型. 每个模态可以独立振荡, 这是对角化在物理中的含义.",
                        ),
                    ),
                ),
                Section(
                    title="5. 物理问题: 正常模态和稳定性",
                    kicker="Physics",
                    intro="很多振动问题都可以化成广义特征值问题.",
                    body=r"""
                    <div class="physics"><strong>问题</strong>小振幅机械系统常写成 $M\ddot x+Kx=0$. 令 $x(t)=u\cos(\omega t)$, 可得到广义特征值问题.</div>
                    <div class="formula">$$
                    Ku=\omega^2Mu
                    $$</div>
                    <p>$u$ 是振型, $\omega$ 是自然频率. 工程上需要避开外部激励频率接近自然频率的情况, 否则可能出现强烈共振.</p>
                    """,
                ),
                Section(
                    title="6. 实战清单与易错点",
                    kicker="Checklist",
                    intro="特征值题要把方程, 空间维数和应用含义连起来.",
                    body=r"""
                    <div class="grid">
                      <div class="card"><strong>实战清单</strong><ol><li>矩阵是否方阵?</li><li>特征多项式是否算对?</li><li>每个特征值的特征空间维数是多少?</li><li>特征向量能否组成基?</li><li>物理问题里特征值代表频率, 增长率还是能量尺度?</li></ol></div>
                      <div class="card"><strong>高频错误</strong><ol><li>允许零向量做特征向量.</li><li>混淆代数重数和几何重数.</li><li>看到 $n$ 个特征值就不检查独立性.</li><li>忘记复特征值可能对应旋转或振荡.</li></ol></div>
                    </div>
                    <div class="summary"><strong>记住</strong>特征向量是变换的自然方向. 对角化是用这些自然方向重写矩阵.</div>
                    """,
                ),
            ],
        ),
        Chapter(
            number=7,
            title="第七章: 内积, 正交与最小二乘",
            subtitle="给向量空间加入长度, 角度和距离, 用投影解释最小二乘和工程数据拟合.",
            focus="距离与投影",
            output="session7.html",
            sections=[
                Section(
                    title="0. 全章主线",
                    kicker="Route",
                    intro="有了内积, 线性代数就能谈最近, 最垂直和最佳近似.",
                    body=flow(
                        [
                            ("内积", "定义长度和角度."),
                            ("正交", "方向互不干扰."),
                            ("投影", "找最近的子空间点."),
                            ("最小二乘", "无精确解时找最佳近似."),
                            ("QR", "用正交基稳定计算."),
                        ]
                    ),
                ),
                Section(
                    title="1. 内积, 范数和距离",
                    kicker="Metric",
                    intro="内积把代数空间变成几何空间.",
                    body=r"""
                    <div class="formula">$$
                    \langle x,y\rangle=x^Ty,\quad \|x\|=\sqrt{\langle x,x\rangle},\quad d(x,y)=\|x-y\|
                    $$</div>
                    <p>正交的定义是 $\langle x,y\rangle=0$. 如果一组非零向量两两正交, 它们一定线性无关, 因为每个方向都无法由其他方向拼出来.</p>
                    <div class="note"><strong>边界</strong>正交依赖内积. 换一个内积, "垂直" 的含义可能随之改变. 这在带权最小二乘和物理能量内积中很常见.</div>
                    """,
                ),
                Section(
                    title="2. 正交投影",
                    kicker="Projection",
                    intro="投影点是子空间中离目标最近的点, 残差与子空间正交.",
                    body=r"""
                    <p>若 $S$ 是一个子空间, $y$ 到 $S$ 的投影 $\hat y$ 满足 $y-\hat y$ 与 $S$ 中所有向量正交. 若 $S$ 由矩阵 $A$ 的列空间给出, 则最小二乘解满足法方程:</p>
                    <div class="formula">$$
                    A^T(A\hat x-y)=0\Rightarrow A^TA\hat x=A^Ty
                    $$</div>
                    <p>这不是凭空出现的公式, 而是"残差垂直于所有可调整方向"的代数表达.</p>
                    """,
                ),
                Section(
                    title="3. Gram-Schmidt 和 QR",
                    kicker="Orthogonalization",
                    intro="把一组独立向量改造成标准正交基, 能显著简化投影和求解.",
                    body=r"""
                    <div class="formula">$$
                    q_k=\frac{v_k-\sum_{j<k}\langle v_k,q_j\rangle q_j}{\left\|v_k-\sum_{j<k}\langle v_k,q_j\rangle q_j\right\|}
                    $$</div>
                    <p>把矩阵列向量正交化, 就得到 $A=QR$. 其中 $Q$ 的列标准正交, $R$ 是上三角矩阵. 数值计算里, QR 通常比直接解法方程更稳定.</p>
                    """,
                ),
                Section(
                    title="4. 几何图像: 投影和传感器拟合",
                    kicker="Geometry",
                    intro="最小二乘本质上是把无法精确命中的目标投影到可达子空间.",
                    body=figure_grid(
                        figure(
                            "assets/la_ch07_projection.png",
                            "向量投影和残差",
                            "蓝色投影是子空间中离目标最近的向量. 黄色残差与子空间正交.",
                        ),
                        figure(
                            "assets/la_ch07_sensor_fit.png",
                            "物理传感器数据的线性校准",
                            "带噪声的温度和电压样本通常不能被一条直线完全穿过. 最小二乘选择残差平方和最小的校准线.",
                        ),
                    ),
                ),
                Section(
                    title="5. 物理问题: 测量校准和误差最小化",
                    kicker="Physics",
                    intro="实验数据有噪声时, 精确解通常不存在, 最小二乘给出可解释的最佳近似.",
                    body=r"""
                    <div class="physics"><strong>问题</strong>传感器输出电压 $v$ 与温度 $T$ 近似满足 $v=aT+b$, 但每次测量都有噪声. 需要从多组样本估计 $a,b$.</div>
                    <div class="formula">$$
                    \min_{a,b}\sum_{i=1}^m (aT_i+b-v_i)^2
                    $$</div>
                    <p>这个目标函数正是在最小化残差向量的长度. 当噪声近似独立同分布且方差相同, 最小二乘还有统计意义上的合理性.</p>
                    """,
                ),
                Section(
                    title="6. 实战清单与易错点",
                    kicker="Checklist",
                    intro="投影问题要先识别子空间, 再写正交条件.",
                    body=r"""
                    <div class="grid">
                      <div class="card"><strong>实战清单</strong><ol><li>目标向量是什么?</li><li>可达子空间由哪些列向量张成?</li><li>残差应与哪些方向正交?</li><li>列满秩条件是否满足?</li><li>是否应使用 QR 而不是法方程?</li></ol></div>
                      <div class="card"><strong>高频错误</strong><ol><li>把投影公式用于非标准正交基.</li><li>把最小二乘解当成精确解.</li><li>忽略 $A^TA$ 可能病态.</li><li>忘记残差垂直的是列空间, 不是原始数据轴.</li></ol></div>
                    </div>
                    <div class="summary"><strong>记住</strong>最小二乘就是投影. 残差正交是公式背后的核心句子.</div>
                    """,
                ),
            ],
        ),
        Chapter(
            number=8,
            title="第八章: 二次型与实对称矩阵",
            subtitle="用矩阵表达二次能量, 曲率和稳定性, 理解正定, 不定, 谱定理和 Hessian 的关系.",
            focus="能量与曲率",
            output="session8.html",
            sections=[
                Section(
                    title="0. 全章主线",
                    kicker="Route",
                    intro="二次型把二阶信息压缩进矩阵, 对称矩阵让这种信息有清晰几何意义.",
                    body=flow(
                        [
                            ("二次型", "$x^TAx$ 表示二阶量."),
                            ("对称化", "只保留真正影响二次型的部分."),
                            ("正定性", "判断能量是否总为正."),
                            ("谱定理", "正交特征基给出主轴."),
                            ("优化", "Hessian 描述局部曲率."),
                        ]
                    ),
                ),
                Section(
                    title="1. 二次型是什么",
                    kicker="Definition",
                    intro="二次型是所有二次项的统一矩阵表达.",
                    body=r"""
                    <div class="formula">$$
                    q(x)=x^TAx
                    $$</div>
                    <p>若 $A$ 不是对称矩阵, 二次型只依赖它的对称部分:</p>
                    <div class="formula">$$
                    x^TAx=x^T\left(\frac{A+A^T}{2}\right)x
                    $$</div>
                    <p>因此研究实二次型时, 可以把焦点放在实对称矩阵上.</p>
                    """,
                ),
                Section(
                    title="2. 正定, 负定和不定",
                    kicker="Definiteness",
                    intro="正定性判断一个二次型在所有非零方向上的符号.",
                    body=r"""
                    <div class="table-wrap"><table><thead><tr><th>类型</th><th>条件</th><th>几何图像</th></tr></thead><tbody>
                    <tr><td>正定</td><td>$x^TAx>0,\ x\ne0$</td><td>能量碗, 原点是严格极小</td></tr>
                    <tr><td>负定</td><td>$x^TAx<0,\ x\ne0$</td><td>倒扣能量碗</td></tr>
                    <tr><td>不定</td><td>有正方向也有负方向</td><td>鞍面</td></tr>
                    <tr><td>半正定</td><td>$x^TAx\ge0$</td><td>某些方向平坦</td></tr>
                    </tbody></table></div>
                    <p>常用判据包括特征值符号, 顺序主子式和 Cholesky 分解是否存在.</p>
                    """,
                ),
                Section(
                    title="3. 谱定理和主轴",
                    kicker="Spectral",
                    intro="实对称矩阵有一组标准正交特征向量, 这让二次型可以被旋转到主轴方向.",
                    body=r"""
                    <div class="formula">$$
                    A=Q\Lambda Q^T,\quad Q^TQ=I
                    $$</div>
                    <p>令 $y=Q^Tx$, 则</p>
                    <div class="formula">$$
                    x^TAx=y^T\Lambda y=\lambda_1y_1^2+\cdots+\lambda_ny_n^2
                    $$</div>
                    <p>这说明特征向量给出二次型的主轴, 特征值给出各主轴方向上的曲率或能量刚度.</p>
                    """,
                ),
                Section(
                    title="4. 几何图像: 曲率, 主轴和能量",
                    kicker="Geometry",
                    intro="等高线能直接显示正定和不定二次型的差别.",
                    body=figure_grid(
                        figure(
                            "assets/la_ch08_quadratic_contours.png",
                            "正定二次型和不定二次型等高线",
                            "正定二次型的等高线围绕原点闭合. 不定二次型会出现鞍形结构, 沿不同方向符号不同.",
                        ),
                        figure(
                            "assets/la_ch08_energy_bowl.png",
                            "耦合弹簧系统的二次能量",
                            "弹性势能常写成 $E=\\frac12 x^TKx$. 刚度矩阵正定意味着平衡点稳定.",
                        ),
                    ),
                ),
                Section(
                    title="5. 物理问题: 势能, 刚度和稳定性",
                    kicker="Physics",
                    intro="小位移系统的稳定性通常由二次能量决定.",
                    body=r"""
                    <div class="physics"><strong>问题</strong>弹簧结构在平衡点附近的势能可近似为 $E(x)=\frac12x^TKx$. 若 $K$ 正定, 任意非零小位移都会增加能量, 平衡点稳定.</div>
                    <div class="formula">$$
                    K\succ0\Rightarrow E(x)>0\quad (x\ne0)
                    $$</div>
                    <p>优化中的 Hessian 也扮演类似角色. 对函数 $f$ 在临界点附近做二阶展开, Hessian 的正定性决定局部极小, 负定性决定局部极大, 不定则常对应鞍点.</p>
                    """,
                ),
                Section(
                    title="6. 实战清单与易错点",
                    kicker="Checklist",
                    intro="二次型题要区分相似, 合同和正交对角化.",
                    body=r"""
                    <div class="grid">
                      <div class="card"><strong>实战清单</strong><ol><li>矩阵是否对称?</li><li>研究的是二次型 $x^TAx$ 还是线性变换 $Ax$?</li><li>需要判断正定性还是求标准形?</li><li>特征值符号是否解释了几何图像?</li></ol></div>
                      <div class="card"><strong>高频错误</strong><ol><li>把合同变换和相似变换混淆.</li><li>只看对角元就判断正定.</li><li>忘记半正定有平坦方向.</li><li>把 Hessian 和梯度混为一谈.</li></ol></div>
                    </div>
                    <div class="summary"><strong>记住</strong>二次型是能量和曲率的矩阵语言. 实对称矩阵的特征值告诉每个主方向上的弯曲程度.</div>
                    """,
                ),
            ],
        ),
        Chapter(
            number=9,
            title="第九章: 矩阵分解与数值视角",
            subtitle="把矩阵拆成更稳定, 更可解释, 更易计算的部分, 理解 LU, QR, Cholesky, SVD, 伪逆和条件数.",
            focus="稳定计算",
            output="session9.html",
            sections=[
                Section(
                    title="0. 全章主线",
                    kicker="Route",
                    intro="分解不是为了形式漂亮, 而是为了求解, 投影, 压缩和控制误差.",
                    body=flow(
                        [
                            ("LU", "把消元过程保存下来."),
                            ("QR", "用正交基做稳定投影."),
                            ("Cholesky", "正定系统的高效分解."),
                            ("SVD", "任何矩阵的主方向分解."),
                            ("条件数", "误差会被放大多少."),
                        ]
                    ),
                ),
                Section(
                    title="1. 常见分解各自解决什么",
                    kicker="Factorization",
                    intro="不同分解服务不同问题, 不能只背名字.",
                    body=r"""
                    <div class="table-wrap"><table><thead><tr><th>分解</th><th>形式</th><th>典型用途</th></tr></thead><tbody>
                    <tr><td>LU</td><td>$A=LU$</td><td>多次求解同一个系数矩阵的方程组</td></tr>
                    <tr><td>QR</td><td>$A=QR$</td><td>最小二乘, 正交化, 稳定求解</td></tr>
                    <tr><td>Cholesky</td><td>$A=LL^T$</td><td>对称正定系统, 协方差, 能量矩阵</td></tr>
                    <tr><td>SVD</td><td>$A=U\Sigma V^T$</td><td>低秩近似, 伪逆, 降维, 病态诊断</td></tr>
                    </tbody></table></div>
                    """,
                ),
                Section(
                    title="2. SVD 的核心图像",
                    kicker="SVD",
                    intro="SVD 把任意矩阵拆成输入正交方向, 单轴缩放和输出正交方向.",
                    body=r"""
                    <div class="formula">$$
                    A=U\Sigma V^T
                    $$</div>
                    <p>$V^T$ 先把输入转到右奇异向量坐标, $\Sigma$ 按奇异值缩放, $U$ 再转到输出方向. 奇异值越小, 对应方向越容易放大误差.</p>
                    <div class="formula">$$
                    A_k=\sum_{i=1}^k\sigma_i u_i v_i^T
                    $$</div>
                    <p>截断 SVD 给出 Frobenius 范数和谱范数意义下的最佳低秩近似.</p>
                    """,
                ),
                Section(
                    title="3. 条件数和数值稳定性",
                    kicker="Conditioning",
                    intro="数学上有唯一解, 不代表数值上容易求.",
                    body=r"""
                    <div class="formula">$$
                    \kappa_2(A)=\frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}
                    $$</div>
                    <p>条件数大表示某些方向被矩阵压得很扁. 求逆或求解时, 这些方向上的微小测量误差会被巨大放大.</p>
                    <div class="warning"><strong>常见误区</strong>行列式很小不等于条件数一定大, 行列式不小也不保证条件数好. 条件数关注最大和最小伸缩比例.</div>
                    """,
                ),
                Section(
                    title="4. 几何图像: SVD 和病态问题",
                    kicker="Geometry",
                    intro="SVD 把单位圆变成椭圆, 条件数就是长短轴比例.",
                    body=figure_grid(
                        figure(
                            "assets/la_ch09_svd_ellipse.png",
                            "单位圆经过矩阵变成椭圆",
                            "奇异值给出椭圆主轴长度. 右奇异向量是输入方向, 左奇异向量是输出方向.",
                        ),
                        figure(
                            "assets/la_ch09_condition_number.png",
                            "近乎平行的约束导致病态求解",
                            "两条线几乎平行时, 右端项的微小变化会让交点明显漂移. 这是病态线性系统的几何直觉.",
                        ),
                    ),
                ),
                Section(
                    title="5. 物理问题: 有限元刚度矩阵",
                    kicker="Physics",
                    intro="大型结构求解需要分解和条件数视角, 不能只看理论公式.",
                    body=r"""
                    <div class="physics"><strong>问题</strong>有限元静力分析常得到 $Ku=f$. $K$ 是大型稀疏刚度矩阵, $u$ 是节点位移, $f$ 是外力. 直接求逆既慢又不稳定.</div>
                    <div class="formula">$$
                    Ku=f,\quad K=K^T,\quad K\succ0\ \text{after boundary conditions}
                    $$</div>
                    <p>工程上会利用稀疏 Cholesky, 共轭梯度或预条件器. 预条件的目标是降低有效条件数, 让迭代更快收敛.</p>
                    """,
                ),
                Section(
                    title="6. 实战清单与易错点",
                    kicker="Checklist",
                    intro="数值线性代数先问目标和矩阵结构, 再选分解.",
                    body=r"""
                    <div class="grid">
                      <div class="card"><strong>实战清单</strong><ol><li>目标是求解, 拟合, 压缩还是诊断?</li><li>矩阵是否对称, 正定, 稀疏, 低秩?</li><li>是否需要多次求解同一矩阵?</li><li>条件数是否会放大误差?</li><li>能否避免显式求逆?</li></ol></div>
                      <div class="card"><strong>高频错误</strong><ol><li>把特征分解和 SVD 当成同一件事.</li><li>对病态矩阵直接求逆.</li><li>忽略浮点误差和数据噪声.</li><li>不知道伪逆只给某种最小范数或最小二乘解.</li></ol></div>
                    </div>
                    <div class="summary"><strong>记住</strong>分解是工程计算的接口. 它让线性代数从"可解"走向"可稳定求解".</div>
                    """,
                ),
            ],
        ),
        Chapter(
            number=10,
            title="第十章: 工程与机器学习应用",
            subtitle="把前九章工具汇总到真实问题: 回归, PCA, 图像压缩, 神经网络线性层, 图算法和物理系统仿真.",
            focus="应用整合",
            output="session10.html",
            sections=[
                Section(
                    title="0. 全章主线",
                    kicker="Route",
                    intro="应用不是另起炉灶, 而是把问题翻译成向量, 矩阵, 投影, 特征值和分解.",
                    body=flow(
                        [
                            ("建模", "决定向量每一维表示什么."),
                            ("矩阵化", "把关系写成线性或局部线性形式."),
                            ("优化", "用投影, 梯度和二次近似求解."),
                            ("分解", "提取主方向, 模态或低秩结构."),
                            ("决策", "解释结果并回到现实约束."),
                        ]
                    ),
                ),
                Section(
                    title="1. 线性回归和最小二乘",
                    kicker="Regression",
                    intro="回归是最小二乘在数据建模中的基本形态.",
                    body=r"""
                    <div class="formula">$$
                    \min_w \|Xw-y\|_2^2
                    $$</div>
                    <p>$X$ 的每一行是一条样本, 每一列是一种特征. 权重 $w$ 不只是参数表, 它给出每个特征方向对预测的线性贡献.</p>
                    <div class="note"><strong>工程提醒</strong>特征量纲差异很大时, 最小二乘会受到尺度影响. 标准化, 正则化和条件数检查都很重要.</div>
                    """,
                ),
                Section(
                    title="2. PCA, 协方差和降维",
                    kicker="PCA",
                    intro="PCA 寻找数据方差最大的正交方向, 本质上是协方差矩阵的特征分解或数据矩阵的 SVD.",
                    body=r"""
                    <div class="formula">$$
                    C=\frac{1}{m}X^TX,\quad Cv=\lambda v
                    $$</div>
                    <p>最大特征值对应最大方差方向. 保留前 $k$ 个主成分, 就是在尽量少损失方差信息的前提下降维.</p>
                    <div class="formula">$$
                    X\approx U_k\Sigma_kV_k^T
                    $$</div>
                    """,
                ),
                Section(
                    title="3. 深度学习里的矩阵乘法",
                    kicker="Deep Learning",
                    intro="神经网络的线性层就是矩阵乘法, 但它通常嵌入非线性和批量张量计算中.",
                    body=r"""
                    <div class="formula">$$
                    Y=XW^T+b
                    $$</div>
                    <p>这里 $X$ 是批量输入, $W$ 的每一行是一组输出通道权重. 反向传播中的梯度同样是矩阵乘法和链式法则的组合.</p>
                    <div class="grid three">
                      <div class="card"><strong>表示</strong>向量编码样本, 词元, 像素块或特征.</div>
                      <div class="card"><strong>变换</strong>权重矩阵改变坐标和特征空间.</div>
                      <div class="card"><strong>优化</strong>Hessian 或近似二阶信息描述局部曲率.</div>
                    </div>
                    """,
                ),
                Section(
                    title="4. 图像, PCA 和物理系统工作流",
                    kicker="Visualization",
                    intro="同一套线性工具可以连接数据降维和工程仿真.",
                    body=figure_grid(
                        figure(
                            "assets/la_ch10_pca_scatter.png",
                            "PCA 主方向示意",
                            "散点数据的第一主成分是方差最大的方向. 这对应协方差矩阵最大特征值的特征向量.",
                        ),
                        figure(
                            "assets/la_ch10_mass_spring_pipeline.png",
                            "物理系统矩阵化工作流",
                            "工程问题常从物理规律出发, 形成矩阵方程, 再通过分解得到模态, 仿真或控制决策.",
                        ),
                    ),
                ),
                Section(
                    title="5. 物理问题: 从质量弹簧到仿真控制",
                    kicker="Physics",
                    intro="真实系统建模通常是线性代数工具链的组合使用.",
                    body=r"""
                    <div class="physics"><strong>问题</strong>多自由度质量弹簧阻尼系统可写成 $M\ddot x+C\dot x+Kx=f(t)$. 其中 $M,C,K$ 分别编码质量, 阻尼和刚度.</div>
                    <div class="formula">$$
                    M\ddot x+C\dot x+Kx=f(t)
                    $$</div>
                    <p>分析时会用特征值理解自然频率, 用分解求解线性系统, 用二次型解释能量, 用状态空间矩阵做控制. 这正是前九章工具在一个问题里的汇合.</p>
                    """,
                ),
                Section(
                    title="6. 实战清单与易错点",
                    kicker="Checklist",
                    intro="应用题的关键是把现实对象和矩阵维度说清楚.",
                    body=r"""
                    <div class="grid">
                      <div class="card"><strong>实战清单</strong><ol><li>向量的每一维代表什么?</li><li>矩阵的行和列分别对应什么对象?</li><li>是精确线性, 近似线性, 还是局部线性?</li><li>用到的分解是否满足前提条件?</li><li>输出能否回到物理量或业务量解释?</li></ol></div>
                      <div class="card"><strong>高频错误</strong><ol><li>只会调库, 不知道矩阵维度含义.</li><li>把相关方向误解为因果关系.</li><li>忽略数据尺度导致数值病态.</li><li>把局部线性近似当成全局真理.</li></ol></div>
                    </div>
                    <div class="summary"><strong>记住</strong>线性代数应用的第一步不是选算法, 而是把现实问题翻译成清晰的向量和矩阵对象.</div>
                    """,
                ),
            ],
        ),
    ]
    return chapters


def main() -> None:
    chapters = build_chapters()
    (ROOT / "index.html").write_text(render_index(chapters), encoding="utf-8")
    for chapter in chapters:
        (ROOT / chapter.output).write_text(render_chapter(chapters, chapter), encoding="utf-8")


if __name__ == "__main__":
    main()
