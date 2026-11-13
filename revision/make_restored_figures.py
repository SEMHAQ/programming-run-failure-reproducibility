"""生成正文用的 12 张图。矢量(PDF+SVG) + 300dpi PNG。

设计约定(按顶刊图的通行做法):
  * 字体 Arial —— 期刊图的通用无衬线体,缩到单栏后仍然清晰。
  * 配色取自 dataviz 参考调色板,已过 validate_palette.js 校验。
    2019 一律 blue、2021 一律 orange —— 颜色跟实体走,不跟排序走。
  * **不用装饰性箭头**。承载意义的定位用参考线 + 直接标注,不用箭头指来指去。
  * **图例不压数据**。能直接标注就直接标;必须用图例时放在空白区或图外。
  * 无顶/右边框;网格只留 y 向且极淡;每个数据点直接标数值。
  * **一律粗体**。轴标题、刻度、图例、数值、注释全部 bold,细体在单栏宽度下读不清。
  * **图内不放解释性长句**。图注里已经写了的,不要把同一句再印在图下方。
  * **按最终印刷尺寸定字号(2026-09-23)。** 字号写在代码里是"标称值",
    读者看到的是"印刷值":

        印刷 pt = 标称 pt × (includegraphics 的宽度 pt / 图 PDF 的自然宽度 pt)

    正文是 10pt(mdpi.cls 的 \\LoadClass[10pt])。用 pdffonts/pdfinfo 量出
    paper.tex 每一张图的缩放比之后发现:fig10 的最小字只印到 **4.11pt**,
    fig04 5.23、fig11 5.77、fig12 5.81、fig06 5.84、fig03 5.87、fig05 5.89、
    fig09 6.15、fig02 6.42、fig07 6.56 —— 12 张图里有 10 张的最小字低于 6.5pt。
    这是可测量的缺陷,不需要引任何期刊规范来证明。

    定下的规则:
      1. 自然宽度 **小于 \\textwidth(394.36pt)的图,paper.tex 里不加 width=**,
         按原始尺寸插入(缩放比 1.0),标称值就是印刷值 —— fig03/04/06/07/08/12。
      2. 自然宽度 **大于 \\textwidth 的图,一律 width=\\textwidth**
         (缩放比 0.82~0.97)—— fig01/02/05/09/10/11。
      3. 于是**印刷最小字号 ≥ 7.0pt** 成了唯一要守的数:改字号前先算
         `7.0 / 缩放比`,只对小字号的元素提到该值,不动版式。

    核对不需要另写脚本:**`python make_figs.py` 画完图会自己跑
    `check_print_size()`**,逐张打印缩放比和最小印刷字号,低于 7.0pt 的
    标 `!`。宽度从 paper.tex 读、自然宽度从 pdfinfo 读,标称字号扫本文件 ——
    三处都是事实,不重述,于是改了 tex 忘了改别的也不会漏。

    注意 rules 1/2 依赖"自然宽度"这个量,而它由 bbox_inches='tight' 裁出来,
    会随字号变。所以**改字号之后必须重跑一次核对**:字变大 → 自然变宽 →
    缩放比变小,是要迭代一两轮才收敛。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.colors import to_rgba
BLUE, ORANGE, AQUA, YELLOW = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')
INK, INK2, MUTED, FAINT = ('#0b0b0b', '#3a3936', '#6e6c66', '#d8d6d0')
SURF = '#ffffff'
from pathlib import Path
FIGDIR = str(Path(__file__).resolve().parent / 'figures')
Path(FIGDIR).mkdir(exist_ok=True)
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 9, 'svg.fonttype': 'none', 'hatch.linewidth': 0.5, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'axes.edgecolor': MUTED, 'axes.linewidth': 0.7, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'xtick.major.size': 2.6, 'ytick.major.size': 2.6, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False, 'figure.dpi': 300, 'lines.solid_capstyle': 'round'})

def save(fig, name):
    for ext in ('pdf', 'svg'):
        fig.savefig('%s/%s.%s' % (FIGDIR, name, ext), bbox_inches='tight')
    fig.savefig('%s/%s.png' % (FIGDIR, name), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  ' + name)

def bare(ax, xlab='', ylab=''):
    if xlab:
        ax.set_xlabel(xlab, color=INK, fontsize=8.8, labelpad=4)
    if ylab:
        ax.set_ylabel(ylab, color=INK, fontsize=8.8, labelpad=5)

def ygrid(ax):
    ax.grid(axis='y', color=FAINT, linewidth=0.7)
    ax.set_axisbelow(True)

def xgrid(ax):
    ax.grid(axis='x', color=FAINT, linewidth=0.7)
    ax.set_axisbelow(True)

def fig1():
    """整篇论文的流程图。

    2026-09-23 重做,第三版。第二版被指出「全是文字和方框、东西挤在一起,
    1→2 那里还出现双箭头」。对着 corpus/pdfs 里四张同类 Figure 1
    (sym18081386 评测流水线、app16125902 纵向流程、app16136455 系统架构、
    app16126156 同题材流水线)重看,它们是这么做的:

      * 框里每行都很短,标题一行、说明最多两三行,长句子留给图注;
      * 面板之间有肉眼可见的空白,不靠边框贴边框来分隔;
      * 合流只画一个箭头 —— app16125902 两路并行的框各自下来,汇成一个点
        再往下走一个箭头,不是每个汇入口都画箭头。

    这一版因此:每框最多 4 行正文、行距 2.7、阶段间距 3.4~5.2;
    合流的两条竖线不带箭头,只有中间那根落下的箭头带箭头。

    **第四版(用户:「你有些框里的字乱七八糟的…我完全读不懂这张图」)。**
    第三版框里塞的是实现细节:特征名清单(edits_since_last_run、
    secs_since_last_edit、blk_prior_errs…)、一行 Python
    (buf = buf[:i] + InsertText + buf[i+len(DeleteText):])、字段名
    (X-Checkpoint、loc = 0)、统计缩写(GroupKFold、ECE、P(Y))。作者自己
    读得懂,别人读不懂 —— 那不是图,是源码摘录。

    规则:**框里每一行都必须是一句人能读的话**;标识符、代码、字段名一律
    移到图注和表里(Tables 1~3 本来就有)。只有两类例外可以留在框里:
    本研究的具体数字(98.281%、242 students、75,112 runs、3B/7B/14B/32B)
    和谁都认识的记号(Δ AUC)。

    **第五版(用户:「可读性还是很低 感觉你在乱用标点符号 特别是 · 我没看到
    这个符号是干嘛 而且字体很小」)。** 两条规则:

      * **不用 `·`。** 那个点是中文排版习惯带进来的,英文读者不知道它是
        「和」還是「或」。并列用逗号,二选一用斜杠,句中不放装饰性符号;
        「Fall 2019 · cs1_2019」→「2019 release, cs1_2019」、
        「242 students · 75,112 runs」→「487 students, 75,112 runs」、
        「all runs · syntax-clean」→「all runs, and recorded-clean runs」。
      * **字号 6.8/7.6 → 8.2/8.8**(印出来 7.98/8.57pt)。代价是每行必须
        短到 ~43 字符,于是文字再砍一轮 —— 把例子挪进图注(「e.g. edits and
        pauses」整句删掉),只留结论句。
    """
    fig, ax = plt.subplots(figsize=(7.0, 6.45))
    ax.set_xlim(0, 100)
    ax.set_ylim(-8.6, 81.2)
    ax.axis('off')
    PAD = 0.35
    BPAD = 0.4

    def box_h(n):
        return 6.95 + (n - 1) * 2.9

    def band(y0, y1, label):
        """阶段面板。标题短(≤20 单位),躲开 x=25.5/74.5 两条竖链。"""
        ax.add_patch(FancyBboxPatch((1.0, y0), 98.0, y1 - y0, boxstyle='round,pad=0.4,rounding_size=0.9', linewidth=0.6, edgecolor='#cfcdc7', facecolor='#f5f4f2', alpha=1.0))
        ax.text(3.4, y1 + 0.4, label, ha='left', va='bottom', fontsize=8.8, color=INK)
        bands.append((1.0 - BPAD, 99.0 + BPAD, y0 - BPAD, y1 + BPAD))
    rects = []
    segs = []
    bands = []

    def box(x0, x1, ytop, col, title, lines, fs=8.2, fill=0.09, ls='solid', hfix=None):
        """从 ytop 往下画;高度按行数算,返回**看得见的**底边 y。

        `hfix` 给一个更大的高度时,框长到那个高度,文字块在框内竖直居中。
        阶段 2 的左列靠它跟右列等高 —— 否则左列的右边框比右列短,横箭头
        没有落脚点,只能挤在两个圆角之间(见下面 ag_my 的推导)。
        """
        n = len(lines)
        h = box_h(n)
        y0 = ytop - h
        H = h if hfix is None else hfix
        dy = (H - h) / 2.0
        ax.add_patch(FancyBboxPatch((x0, ytop - H), x1 - x0, H, boxstyle='round,pad=%g,rounding_size=0.9' % PAD, linewidth=1.0, edgecolor=col, facecolor=to_rgba(col, fill), linestyle=ls))
        mid = (x0 + x1) / 2.0
        tobjs = [ax.text(mid, ytop - dy - 1.9, title, ha='center', va='top', fontsize=8.8, color=INK)]
        for i, ln in enumerate(lines):
            tobjs.append(ax.text(mid, ytop - dy - 4.8 - i * 2.9, ln, ha='center', va='top', fontsize=fs, color=INK))
        rects.append((x0 - PAD, x1 + PAD, ytop - H - PAD, ytop + PAD, tobjs, title))
        return ytop - H - PAD

    def check():
        """把每个框里的 Text 量成 bbox,越界就报 —— 单栏宽度下眼睛看不出来。"""
        fig.canvas.draw()
        inv = ax.transData.inverted()
        bad = 0
        tight = (9000000000.0, '')
        for rx0, rx1, ry0, ry1, tobjs, name in rects:
            for t in tobjs:
                bb = t.get_window_extent()
                (a, b), (c, d) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
                if a < rx0 or c > rx1 or b < ry0 or (d > ry1):
                    bad += 1
                    print('    !! 出界 [%s] %r  x %.2f..%.2f / 框 %.2f..%.2f  y %.2f..%.2f / 框 %.2f..%.2f' % (name, t.get_text(), a, c, rx0, rx1, b, d, ry0, ry1))
                s = min(a - rx0, rx1 - c, b - ry0, ry1 - d)
                if s < tight[0]:
                    tight = (s, '%s / %r' % (name, t.get_text()))
        print('    fig01 溢出检查:%s;最挤处余量 %.2f 单位(%s)' % ('全部在框内' if not bad else '%d 条出界' % bad, tight[0], tight[1]))
        TOL = 0.05
        rects_all = [(r[0], r[1], r[2], r[3]) for r in rects] + bands

        def on_rect(p):
            x, y = p
            for rx0, rx1, ry0, ry1 in rects_all:
                on_x = rx0 - TOL <= x <= rx1 + TOL
                on_y = ry0 - TOL <= y <= ry1 + TOL
                if on_x and on_y and (abs(x - rx0) < TOL or abs(x - rx1) < TOL or abs(y - ry0) < TOL or (abs(y - ry1) < TOL)):
                    return True
            return False

        def on_seg(p, skip):
            x, y = p
            for k, ((ax_, ay), (bx, by)) in enumerate(segs):
                if k == skip:
                    continue
                dx, dy = (bx - ax_, by - ay)
                L2 = dx * dx + dy * dy
                if L2 == 0:
                    t = 0.0
                else:
                    t = max(0.0, min(1.0, ((x - ax_) * dx + (y - ay) * dy) / L2))
                d = ((x - (ax_ + t * dx)) ** 2 + (y - (ay + t * dy)) ** 2) ** 0.5
                if d < TOL:
                    return True
            return False
        loose = 0
        for k, (p0, p1) in enumerate(segs):
            for tag, p in (('尾', p0), ('头', p1)):
                if not (on_rect(p) or on_seg(p, k)):
                    loose += 1
                    print('    !! 悬空 %s端 (%.2f, %.2f)' % (tag, p[0], p[1]))
        print('    fig01 连线检查:%s(%d 条线段)' % ('端点全部接住' if not loose else '%d 个端点悬空' % loose, len(segs)))

    def arrow(x0, y0, x1, y1, col=INK2, ls='solid'):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=col, linewidth=0.9, linestyle=ls, shrinkA=0, shrinkB=0))
        segs.append(((x0, y0), (x1, y1)))

    def link(x0, y0, x1, y1, col=INK2):
        """不带箭头的连线:合流处只在最后一段画箭头,免得出现两个箭头。"""
        ax.plot([x0, x1], [y0, y1], color=col, linewidth=0.9, solid_capstyle='round')
        segs.append(((x0, y0), (x1, y1)))

    def edge(x, y0, y1, label, lx=1.5, ha='left', fs=8.2, col=INK2, ly=None):
        """竖直边 + 边上的标签。ly 显式给出标签高度,免得落在带标题那一行上。"""
        arrow(x, y0, x, y1, col=col)
        ax.text(x + lx, (y0 + y1) / 2.0 if ly is None else ly, label, ha=ha, va='center', fontsize=fs, color=INK)
    b1_top = 78.0
    b1_bot = b1_top - (2.4 + box_h(3) + 1.3)
    band(b1_bot, b1_top, '1. Corpora')
    cb = b1_top - 2.4
    b1a_bot = box(3.0, 48.0, cb, BLUE, '2019 release, cs1_2019', ['logs from the Phanon plug-in', '487 students, 75,112 runs', 'a failed run is marked in the log'])
    b1b_bot = box(52.0, 97.0, cb, ORANGE, 'Fall 2021, cs1', ['PyCharm logs (ProgSnap2 format)', '44 students, 54,561 runs', 'the outcome sits in a separate file'])
    bus = b1_bot - 2.8
    link(25.5, b1a_bot, 25.5, bus)
    link(74.5, b1b_bot, 74.5, bus)
    link(25.5, bus, 74.5, bus)
    b2_top = b1_bot - 5.4
    h_xc, h_ag = (box_h(1), box_h(1))
    GAP2 = 4.4
    h_col = h_xc + GAP2 + h_ag
    b2_bot = b2_top - (2.4 + h_col + 1.6)
    band(b2_bot, b2_top, '2. Reconstruction')
    arrow(50.0, bus, 50.0, b2_top + BPAD)
    ax.text(52.0, (bus + b2_top) / 2.0, 'edit events, timestamped', ha='left', va='center', fontsize=8.2, color=INK)
    r_top = b2_top - 2.4
    r_bot = box(3.0, 52.0, r_top, INK, 'Rebuild the file at every run', ['replay the edits in timestamp order', 'each insert and delete at its caret', 'the same stream always gives the same file', '2019 has checkpoints, 2021 does not'], fill=0.045, hfix=h_col)
    xc_bot = box(58.0, 97.0, r_top, MUTED, 'X-Compilable', ['what the IDE itself reported'], fill=0.1, ls='dashed')
    ag_top = xc_bot - GAP2
    ag_bot = box(58.0, 97.0, ag_top, INK, 'Agreement with the rebuild', ['98.281% (2019), 99.873% (2021)'], fill=0.045)
    ag_my = ag_top + PAD - (h_ag + 2 * PAD) / 2.0
    arrow(52.0 + PAD, ag_my, 58.0 - PAD, ag_my)
    arrow(77.5, xc_bot, 77.5, ag_top + PAD, col=MUTED, ls='dashed')
    b3_top = b2_bot - 4.4
    b3_bot = b3_top - (2.4 + box_h(3) + 1.3)
    band(b3_bot, b3_top, '3. Predictors')
    p_top = b3_top - 2.4
    p_bot = box(3.0, 48.0, p_top, AQUA, 'Process features', ['8 pre-run process features,', 'including earlier failure counts', 'gradient-boosted trees'])
    box(52.0, 97.0, p_top, YELLOW, 'Source reader', ['Qwen2.5-Instruct, 3B / 7B / 14B / 32B', '4-bit, on one consumer GPU', 'reads the file; will this run fail?'])
    edge(25.5, r_bot, p_top + PAD, 'run events', ly=b2_bot - 1.4)
    edge(74.5, ag_bot, p_top + PAD, 'the rebuilt file', ly=b2_bot - 1.4)
    b4_top = b3_bot - 4.4
    b4_bot = b4_top - (2.4 + box_h(3) + 1.3)
    band(b4_bot, b4_top, '4. Evaluation')
    box(3.0, 97.0, b4_top - 2.4, INK, 'Scored on runs the model has never seen', ["every student's runs stay inside one fold", 'two arms: all runs, and recorded-clean runs', 'do the process features add anything? (Δ AUC)'], fill=0.035)
    edge(25.5, p_bot, b4_top + BPAD, '8 features', col=MUTED, ly=b3_bot - 1.4)
    edge(74.5, p_bot, b4_top + BPAD, 'scores', col=MUTED, ly=b3_bot - 1.4)
    check()
    save(fig, 'fig01_pipeline')

def fig2():
    d19 = [('p4f', 23.440860215053764, 43.164607413312076), ('p5f', 21.463414634146343, 37.109375), ('p6f', 19.48113958300175, 35.35186129525752), ('p7f', 14.846206425153794, 29.333626084092657), ('p8f', 23.58429160338468, 51.956303645078584)]
    d21 = [('Assign6', 11.446582614465827, 14.111705288175877), ('Assign7', 38.9028827315981, 39.74477958236659), ('Assign8', 46.34703196347032, 50.70603337612324), ('Assign9', 51.167582417582416, 54.71594043022614), ('Assign10', 65.63953488372093, 65.5311778290993), ('Assign11', 78.87197851387646, 78.27868852459017), ('Assign12', 47.944142746314974, 50.032916392363404), ('Assign13', 68.88633754305397, 70.2368692070031)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True)
    for ax, data, tag, col in ((axes[0], d19, 'Fall 2019', BLUE), (axes[1], d21, 'Fall 2021', ORANGE)):
        for i, (lab, comp, allr) in enumerate(data):
            y = len(data) - 1 - i
            ax.plot([comp, allr], [y, y], color=col, linewidth=1.8, alpha=0.32, zorder=1)
            ax.scatter([allr], [y], s=30, color=SURF, edgecolor=col, linewidth=1.5, zorder=3)
            ax.scatter([comp], [y], s=30, color=col, zorder=3)
            ax.text(allr + 3.5, y, '%.0f' % allr, va='center', fontsize=7.7, color=INK)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([d[0] for d in data][::-1], fontsize=7.8)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.7, len(data) - 0.3)
        xgrid(ax)
        bare(ax, 'run failure rate (%)')
        ax.set_title(tag, fontsize=9.2, color=INK, pad=6)
    h1 = axes[0].scatter([], [], s=30, color=MUTED, label='recorded-clean subset')
    h2 = axes[0].scatter([], [], s=30, color=SURF, edgecolor=MUTED, linewidth=1.5, label='all runs')
    fig.legend(handles=[h1, h2], loc='lower center', bbox_to_anchor=(0.5, -0.135), ncol=2, fontsize=7.7, handletextpad=0.3, columnspacing=1.8)
    save(fig, 'fig02_failure_rates')

def fig3():
    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    x = np.arange(2)
    w = 0.32
    a = [98.281, 99.873]
    b = [99.712, 99.097]
    ax.bar(x - w / 2 - 0.012, a, w, color=BLUE, label='vs. compile status')
    ax.bar(x + w / 2 + 0.012, b, w, color=AQUA, hatch='/////', label='per-edit DeleteText')
    for xi, v in zip(x - w / 2 - 0.012, a):
        ax.text(xi, v + 0.055, '%.2f' % v, ha='center', fontsize=7.2, color=INK)
    for xi, v in zip(x + w / 2 + 0.012, b):
        ax.text(xi, v + 0.055, '%.2f' % v, ha='center', fontsize=7.2, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(['Fall 2019', 'Fall 2021'], fontsize=8.5)
    ax.set_ylim(97.5, 100.5)
    ax.set_yticks([98, 99, 100])
    ygrid(ax)
    bare(ax, '', 'reconstruction agreement (%)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fontsize=7.4, ncol=2, handlelength=1.0, columnspacing=1.6)
    save(fig, 'fig03_reconstruction')

def fig5():
    d19 = [('p4f', 0.6168428190392743), ('p5f', 0.6079457227555054), ('p6f', 0.6380465410121352), ('p7f', 0.6985553965929575), ('p8f', 0.7030857947964875)]
    d21 = [('Assign6', 0.6037173994493948), ('Assign7', 0.5930694015561715), ('Assign8', 0.6698692193923301), ('Assign9', 0.6008986303438771), ('Assign10', 0.7148458048765135), ('Assign11', 0.48386848534985283), ('Assign12', 0.6298814984156381), ('Assign13', 0.6793111931119311)]
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 4.3), sharey=True)
    fig.subplots_adjust(hspace=0.62)
    for i, (ax, data, tag, col, mean) in enumerate(((axes[0], d19, 'Fall 2019', BLUE, 0.652895254839272), (axes[1], d21, 'Fall 2021', ORANGE, 0.6219327040619637))):
        x = np.arange(len(data))
        ax.axhline(0.5, color=MUTED, linewidth=0.7, linestyle=(0, (4, 3)))
        ax.axhline(mean, color=col, linewidth=1.0, linestyle=(0, (1.5, 2.5)), alpha=0.7)
        ax.scatter(x, [v for _, v in data], s=36, color=col, zorder=3)
        for xi, (lab, v) in zip(x, data):
            below = v < mean
            ax.text(xi, v - 0.014 if below else v + 0.018, '%.3f' % v, ha='center', va='top' if below else 'bottom', fontsize=7.9, color=ORANGE if v < 0.55 else INK, fontweight='bold' if v < 0.55 else 'normal')
        ax.set_xticks(x)
        ax.set_xticklabels([d[0].replace('Assign', 'Assign\n') for d in data], fontsize=7.9, rotation=0, ha='center', linespacing=1.25)
        ax.set_xlim(-0.7, len(data) - 0.3)
        ax.set_ylim(0.41, 0.78)
        ygrid(ax)
        ax.set_ylabel('AUC (syntax-free process)', color=INK, fontsize=8.8, labelpad=5)
        ax.set_title('%s, mean %.4f' % (tag, mean), fontsize=9.2, color=INK, pad=6)
        ax.text(0.02, 0.502, 'chance', fontsize=7.9, color=MUTED, va='bottom', transform=ax.get_yaxis_transform())
        if i == 1:
            bare(ax, 'assignment')
    save(fig, 'fig05_process_per_assignment')

def fig8():
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    params = [3, 7, 14, 32]
    r19 = [0.3089632222859874, 0.7756599052960509, 0.867502194614584, 0.8764261004836555]
    r21 = [0.498208554367674, 0.6610534386759894, 0.6737199409804584, 0.7639337126330057]
    ax.axhspan(0.25, 0.5, color=FAINT, alpha=0.35, zorder=0)
    ax.text(2.55, 0.268, 'below chance', fontsize=7.0, color=MUTED, va='bottom')
    ax.axhline(0.6716019422257944, color=BLUE, linewidth=1.0, linestyle=(0, (5, 3)), alpha=0.7)
    ax.text(2.55, 0.6786019422257944, 'process, 2019', fontsize=7.0, color=BLUE, va='bottom')
    ax.axhline(0.6085732967978108, color=ORANGE, linewidth=1.0, linestyle=(0, (5, 3)), alpha=0.7)
    ax.text(2.55, 0.6155732967978108, 'process, 2021', fontsize=7.0, color=ORANGE, va='bottom')
    for vals, col in ((r19, BLUE), (r21, ORANGE)):
        ax.plot(params, vals, color=col, linewidth=1.9, marker='o', markersize=6.5, markeredgecolor=SURF, markeredgewidth=1.1, zorder=3)
    for i, (a, b) in enumerate(zip(r19, r21)):
        ax.text(params[i], a + 0.026, '%.3f' % a, ha='center', fontsize=7.0, color=INK, fontweight='bold')
        ax.text(params[i], b + 0.026, '%.3f' % b, ha='center', fontsize=7.0, color=INK)
    ax.set_xscale('log')
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(params)
    ax.set_xticklabels(['3B', '7B', '14B', '32B'], fontsize=8.4)
    ax.set_xlim(2.4, 40)
    ax.set_ylim(0.25, 0.98)
    ygrid(ax)
    bare(ax, 'model size (Qwen2.5-Instruct, Q4_K_M)', 'AUC (recorded-clean, assignment-controlled)')
    ax.plot([], [], color=BLUE, linewidth=1.9, marker='o', markersize=6.5, label='Fall 2019')
    ax.plot([], [], color=ORANGE, linewidth=1.9, marker='o', markersize=6.5, label='Fall 2021')
    ax.legend(loc='lower right', fontsize=7.4, handlelength=1.8)
    save(fig, 'fig08_ladder')

def fig9():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={'width_ratios': [1, 1.3]})
    ax = axes[0]
    bars = ax.bar([0, 1], [np.float64(31.715771230502597), np.float64(45.294117647058826)], 0.46, color=[INK2, SURF], edgecolor=INK2, linewidth=0.9, hatch=['', '////'])
    for xi, v in zip([0, 1], [np.float64(31.715771230502597), np.float64(45.294117647058826)]):
        ax.text(xi, v + 1.0, '%.1f%%' % v, ha='center', fontsize=8.2, color=INK, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['score >= 0.5', 'score < 0.5'], fontsize=8)
    ax.set_ylim(0, 65)
    ygrid(ax)
    bare(ax, '', 'true failure rate (%)')
    ax.set_title('3B archived scores (exploratory)', fontsize=9.0, color=INK, pad=6)
    ax = axes[1]
    d19 = [0.19027777777777777, 0.5939849624060151, 0.26952085181898844, 0.2833670033670034, 0.48461028023071817]
    d21 = [0.4475210970464135, 0.5388614800759013, 0.4415979157620495, 0.21425576519916142, 0.4247658079625293, 0.29545454545454547, 0.49145299145299143, 0.611336032388664]
    ax.axhspan(0.12, 0.5, color=FAINT, alpha=0.35, zorder=0)
    ax.axhline(0.5, color=MUTED, linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
    ax.scatter(np.arange(len(d19)) - 0.11, d19, s=36, color=BLUE, zorder=3, label='Fall 2019 (5 assignments)')
    ax.scatter(np.arange(len(d21)) + 0.11, d21, s=36, color=ORANGE, zorder=3, label='Fall 2021 (8 assignments)')
    ax.set_xticks(range(8))
    ax.set_xticklabels(['%d' % (i + 1) for i in range(8)], fontsize=7.6)
    ax.set_xlim(-0.65, 7.65)
    ax.set_ylim(0.0, 0.85)
    ax.text(0.02, 0.505, 'chance', ha='left', va='bottom', fontsize=7.6, color=MUTED, transform=ax.get_yaxis_transform())
    ygrid(ax)
    bare(ax, 'assignment (indexed within cohort)', 'AUC (3B, assignment-controlled)')
    ax.legend(loc='lower right', fontsize=7.6, handletextpad=0.25, labelspacing=0.3)
    save(fig, 'fig09_3b_inverted')

def fig11():
    """校准:不用可靠性曲线。

    **为什么换掉**:可靠性曲线假设预测值是连续分布的。实测不是 —— 14B 把
    78.7% 的预测压在 >=0.95(32B 是 74.3%),所以曲线退化成 x=1 处的一根竖线
    加一堆点,读不出任何东西。图表的类型选错了,不是配色或排版的问题。

    改成两个面板,直接呈现真正要说的两件事:
      (a) 预测 vs 实际 —— 所有模型都高估
      (b) 预测值 >=0.95 的占比 —— 分数是判决,不是概率
    """
    import json as _json
    D = {'3B|2019': {'pred': 0.32230344923558646, 'obs': 0.3837471783295711, 'sat': 0.06772009029345373}, '7B|2019': {'pred': 0.6315905810068055, 'obs': 0.3837471783295711, 'sat': 0.5158013544018059}, '14B|2019': {'pred': 0.7305104499425615, 'obs': 0.3837471783295711, 'sat': 0.6997742663656885}, '32B|2019': {'pred': 0.7196905602890205, 'obs': 0.3837471783295711, 'sat': 0.6817155756207675}, '3B|2021': {'pred': 0.34923620224936563, 'obs': 0.4335981838819523, 'sat': 0.06923950056753689}, '7B|2021': {'pred': 0.7564439284297887, 'obs': 0.4335981838819523, 'sat': 0.5459704880817253}, '14B|2021': {'pred': 0.8523069493844887, 'obs': 0.4335981838819523, 'sat': 0.8070374574347332}, '32B|2021': {'pred': 0.8267324703952761, 'obs': 0.4335981838819523, 'sat': 0.7707150964812712}}
    models = ['3B', '7B', '14B', '32B']
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={'width_ratios': [1.35, 1.35, 1.0]})
    for ax, ds, ttl in ((axes[0], '2019', 'Fall 2019'), (axes[1], '2021', 'Fall 2021')):
        x = np.arange(4)
        w = 0.36
        pred = [100 * D['%s|%s' % (m, ds)]['pred'] for m in models]
        obs = [100 * D['%s|%s' % (m, ds)]['obs'] for m in models]
        ax.bar(x - w / 2 - 0.012, pred, w, color=ORANGE, label='predicted')
        ax.bar(x + w / 2 + 0.012, obs, w, color=BLUE, label='observed')
        for xi, v in zip(x - w / 2 - 0.012, pred):
            ax.text(xi, v + 1.6, '%.0f' % v, ha='center', fontsize=8.2, color=INK)
        for xi, v in zip(x + w / 2 + 0.012, obs):
            ax.text(xi, v + 1.6, '%.0f' % v, ha='center', fontsize=8.2, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8.2)
        ax.set_ylim(0, 100)
        ygrid(ax)
        bare(ax, '', 'run failure rate (%)' if ds == '2019' else '')
        ax.set_title(ttl, fontsize=9.0, color=INK, pad=5)
    axes[0].legend(loc='upper left', fontsize=8.2, handlelength=1.0, labelspacing=0.3)
    ax = axes[2]
    x = np.arange(4)
    w = 0.36
    s19 = [100 * D['%s|2019' % m]['sat'] for m in models]
    s21 = [100 * D['%s|2021' % m]['sat'] for m in models]
    ax.bar(x - w / 2 - 0.012, s19, w, color=BLUE, label='2019')
    ax.bar(x + w / 2 + 0.012, s21, w, color=ORANGE, label='2021')
    for xi, v in zip(x - w / 2 - 0.012, s19):
        ax.text(xi, v + 1.6, '%.0f' % v, ha='center', fontsize=8.2, color=INK)
    for xi, v in zip(x + w / 2 + 0.012, s21):
        ax.text(xi, v + 1.6, '%.0f' % v, ha='center', fontsize=8.2, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.2)
    ax.set_ylim(0, 100)
    ygrid(ax)
    ax.set_title('predictions ≥ 0.95 (%)', fontsize=8.2, color=INK, pad=5)
    ax.legend(loc='upper left', fontsize=8.2, handlelength=1.0, labelspacing=0.3)
    save(fig, 'fig11_calibration')

def fig12():
    m = [('3B', 4216, 0.3089632222859874, 0.498208554367674), ('7B', 5910, 0.7756599052960509, 0.6610534386759894), ('14B', 10943, 0.867502194614584, 0.6737199409804584), ('32B', 21776, 0.8764261004836555, 0.7639337126330057)]
    fig, ax = plt.subplots(figsize=(5.3, 3.1))
    ax.axhline(0.6716019422257944, color=BLUE, linewidth=1.0, linestyle=(0, (5, 3)), alpha=0.7)
    ax.text(23200, 0.6786019422257944, 'process, 2019', ha='right', fontsize=7.0, color=BLUE, va='bottom')
    ax.axhline(0.6085732967978108, color=ORANGE, linewidth=1.0, linestyle=(0, (5, 3)), alpha=0.7)
    ax.text(23200, 0.6155732967978108, 'process, 2021', ha='right', fontsize=7.0, color=ORANGE, va='bottom')
    for idx, col in ((2, BLUE), (3, ORANGE)):
        ax.plot([x[1] for x in m], [x[idx] for x in m], color=col, linewidth=1.9, marker='o', markersize=6.5, markeredgecolor=SURF, markeredgewidth=1.1, zorder=3)
    off = {'3B': (-180, 'right'), '7B': (-150, 'right')}
    for lab, v, a, b in m:
        dx, ha = off.get(lab, (0, 'center'))
        ax.text(v + dx, a + 0.026, lab, ha=ha, fontsize=7.2, color=INK, fontweight='bold')
    ax.axvline(5910, color=MUTED, linewidth=0.9, linestyle=(0, (2, 2.5)), zorder=1)
    ax.set_xlim(2200, 23500)
    ax.set_ylim(0.3, 0.98)
    ax.set_xticks([5000, 10000, 15000, 20000])
    ygrid(ax)
    bare(ax, 'device memory when loaded (MB)', 'AUC (recorded-clean, assignment-controlled)')
    ax.plot([], [], color=BLUE, linewidth=1.9, marker='o', markersize=6.5, label='Fall 2019')
    ax.plot([], [], color=ORANGE, linewidth=1.9, marker='o', markersize=6.5, label='Fall 2021')
    ax.legend(loc='lower right', fontsize=7.4, handlelength=1.8)
    save(fig, 'fig12_cost')
if __name__ == '__main__':
    for f in (fig1, fig3, fig2, fig5, fig8, fig9, fig11, fig12):
        f()
