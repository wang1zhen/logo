"""中音谱号（C 谱号）个人 Logo：两种手绘轮廓方案 + 渲染导出。

坐标约定：五线谱线距 = height / 4，第三线在 y = 0，五条线位于 y = k·height/4（k = -2..2）。
左侧一粗一细两根竖柱；右侧是一个整体轮廓：中腰实心楔形，向左的尖凹顶点落在第三线 y = 0
（中音谱号指示中央 C 的位置），上下各伸出一片尖叶，再接粗腹细回转的卷叶，卷端收成带钩的圆团。

所有图形都是闭合三次 Bézier 复合路径：外轮廓逆时针，留白（飞白）子路径顺时针，
按 nonzero 规则填充即可得到镂空，不依赖 even-odd。形态参考 SMuFL cClef（U+E05C）独立手绘，
不使用任何字体的控制点。

直接运行 `python clef.py` 把默认参数的全部图形导出到 out/。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath

OPENING_MAX = 0.8
SEED_MAX = 2**32 - 1
# 实测支持的范围（相对 height）：超出后路径会自交、部件互相穿插或卷叶留白被挤掉
WEIGHT_MIN, WEIGHT_MAX = 0.0375, 0.125
GAP_MIN, GAP_MAX = 0.025, 0.25
WIDTH_MAX = 0.9
WAIST_GAP = 0.3  # 腰部左端与细柱的距离 = WAIST_GAP · gap
SHEAR_B = 0.045  # B 主体的右倾斜势（x += SHEAR_B · y）


# ---------------------------------------------------------------- 参数

@dataclass(frozen=True)
class Params:
    height: float = 4.0  # 竖柱总高（= 五线谱上下两线的距离）
    width: float = 2.76  # 整体宽度：粗柱左缘到外腹最右点，默认约 0.69·height
    weight: float = 0.25  # 基础笔画粗细；粗柱为 2 倍，外腹约 2.6 倍
    opening: float = 0.25  # 卷端开口：0 = 圆团贴近尖叶，OPENING_MAX = 圆团上移、尖叶压低
    gap: float = 0.25  # 粗细柱间距；腰部与细柱之间留 WAIST_GAP·gap
    contrast: float = 0.6  # 粗细对比：越大细柱与回转发丝越细、外腹越厚
    seed: int = 7  # 飞白、墨晕与 B 方案手抖的随机种子（只用局部 RNG）
    dryness: float = 0.35  # 飞白量，0 = 无
    bleed: float = 0.3  # 墨晕量，0 = 无

    def __post_init__(self):
        for name in ("height", "width", "weight", "opening", "gap", "contrast", "dryness", "bleed"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{name} 必须是有限数值，得到 {v!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError(f"seed 必须是整数，得到 {self.seed!r}")
        if not 0 <= self.seed <= SEED_MAX:
            raise ValueError(f"seed 必须在 [0, {SEED_MAX}] 内")
        if self.height <= 0:
            raise ValueError("height 必须 > 0")
        h = self.height
        if not WEIGHT_MIN * h <= self.weight <= WEIGHT_MAX * h:
            raise ValueError(f"weight 必须在 [{WEIGHT_MIN}·height, {WEIGHT_MAX}·height] = "
                             f"[{WEIGHT_MIN * h:g}, {WEIGHT_MAX * h:g}] 内")
        if not 0 <= self.opening <= OPENING_MAX:
            raise ValueError(f"opening 必须在 [0, {OPENING_MAX}] 内")
        if not GAP_MIN * h <= self.gap <= GAP_MAX * h:
            raise ValueError(f"gap 必须在 [{GAP_MIN}·height, {GAP_MAX}·height] = [{GAP_MIN * h:g}, {GAP_MAX * h:g}] 内")
        for name in ("contrast", "dryness", "bleed"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} 必须在 [0, 1] 内")
        need = self.min_width()
        if self.width < need:
            raise ValueError(f"width 太小：当前粗细/间距下至少需要 {need:.3f}")
        if self.width > WIDTH_MAX * h:
            raise ValueError(f"width 不应超过 {WIDTH_MAX}·height = {WIDTH_MAX * h:g}（再宽圆团会撞上尖叶）")

    @property
    def half(self) -> float:
        return self.height / 2

    @property
    def thin(self) -> float:
        """细柱宽度。"""
        return self.weight * (1 - 0.6 * self.contrast)

    @property
    def hair(self) -> float:
        """卷叶上下回转处的发丝线粗细。"""
        return self.weight * (1 - 0.8 * self.contrast) * 0.95 + 0.004 * self.height

    @property
    def x_waist(self) -> float:
        """腰部左端的 x 坐标。"""
        return 2 * self.weight + self.gap + self.thin + WAIST_GAP * self.gap

    def min_width(self) -> float:
        return self.x_waist + max(4 * self.weight, 0.3 * self.height)


# ---------------------------------------------------------------- Bézier 工具

def _dir(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([math.cos(a), math.sin(a)])


def K(x, y, a_in, a_out=None, k_in=0.36, k_out=None):
    """锚点：位置、入/出切线方向（度，沿行进方向）、入/出控制柄长度（相对弦长）。"""
    return (float(x), float(y), float(a_in), float(a_in if a_out is None else a_out),
            float(k_in), float(k_in if k_out is None else k_out))


def chain(anchors) -> np.ndarray:
    """锚点 → 开放三次 Bézier 链的控制点 (3n+1, 2)：P0 C C P1 C C P2 …"""
    out = [np.array(anchors[0][:2])]
    for a, b in zip(anchors[:-1], anchors[1:]):
        p0, p3 = np.array(a[:2]), np.array(b[:2])
        d = float(np.hypot(*(p3 - p0)))
        out += [p0 + a[5] * d * _dir(a[3]), p3 - b[4] * d * _dir(b[2]), p3]
    return np.array(out)


def line(p0, p1) -> np.ndarray:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    return np.array([p0, p0 + (p1 - p0) / 3, p0 + 2 * (p1 - p0) / 3, p1])


def join(*chains) -> np.ndarray:
    """首尾相接的若干链 → 一条链（去掉重复的衔接点）。"""
    out = [chains[0]]
    for c in chains[1:]:
        if not np.allclose(out[-1][-1], c[0], atol=1e-9):
            raise ValueError("链没有首尾相接")
        out.append(c[1:])
    return np.vstack(out)


def closed(anchors) -> np.ndarray:
    return chain([*anchors, anchors[0]])


def mirror_rev(c: np.ndarray) -> np.ndarray:
    """关于 y=0 镜像并倒序，保持轮廓的环绕方向。"""
    return c[::-1] * [1.0, -1.0]


def catmull_closed(pts: np.ndarray) -> np.ndarray:
    """经过所有点的闭合 Catmull-Rom 样条 → 三次 Bézier 控制点。"""
    p = np.asarray(pts, float)
    pm, pn, pnn = np.roll(p, 1, 0), np.roll(p, -1, 0), np.roll(p, -2, 0)
    c1, c2 = p + (pn - pm) / 6, pn - (pnn - p) / 6
    out = np.empty((3 * len(p) + 1, 2))
    out[0] = p[0]
    out[1::3], out[2::3], out[3::3] = c1, c2, pn
    return out


def flatten(c: np.ndarray, n: int = 24) -> np.ndarray:
    """三次 Bézier 链 → 折线（不重复首点）。"""
    c = np.asarray(c, float)
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    p0, p1, p2, p3 = c[0:-1:3], c[1::3], c[2::3], c[3::3]
    seg = [(1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t**2 * d + t**3 * e
           for a, b, d, e in zip(p0, p1, p2, p3)]
    return np.vstack(seg)


def signed_area(c: np.ndarray) -> float:
    q = flatten(c)
    x, y = q[:, 0], q[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def winding(points, contours) -> np.ndarray:
    """各点相对一组闭合轮廓的 nonzero 环绕数之和（逆时针 +1）。"""
    q = np.atleast_2d(np.asarray(points, float))
    total = np.zeros(len(q), dtype=int)
    for c in contours:
        poly = flatten(c)
        idx = np.flatnonzero((q[:, 1] >= poly[:, 1].min()) & (q[:, 1] <= poly[:, 1].max()) & (q[:, 0] <= poly[:, 0].max()))
        if not len(idx):
            continue
        a, b = poly, np.roll(poly, -1, 0)
        ax_, ay, bx, by = a[:, 0][None], a[:, 1][None], b[:, 0][None], b[:, 1][None]
        px, py = q[idx, :1], q[idx, 1:]
        cross = (bx - ax_) * (py - ay) - (px - ax_) * (by - ay)
        up = (ay <= py) & (by > py) & (cross > 0)
        down = (ay > py) & (by <= py) & (cross < 0)
        total[idx] += up.sum(1) - down.sum(1)
    return total


def to_path(contours) -> MplPath:
    verts, codes = [], []
    for c in contours:
        verts += [*c, c[0]]
        codes += [MplPath.MOVETO] + [MplPath.CURVE4] * (len(c) - 1) + [MplPath.CLOSEPOLY]
    return MplPath(np.array(verts), codes)


def bbox_of(contours) -> tuple[float, float, float, float]:
    allp = np.vstack([flatten(c) for c in contours])
    return (*allp.min(axis=0), *allp.max(axis=0))


def rounded_rect(x0, y0, x1, y1, r) -> np.ndarray:
    """逆时针圆角矩形：直边与四分之一圆弧交替。"""
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = [(x0 + r, y0, 0), (x1 - r, y0, 0), (x1, y0 + r, 90), (x1, y1 - r, 90),
           (x1 - r, y1, 180), (x0 + r, y1, 180), (x0, y1 - r, 270), (x0, y0 + r, 270)]
    arc, straight = 0.552 / math.sqrt(2), 1 / 3
    return closed([K(x, y, a, a, straight if i % 2 else arc, arc if i % 2 else straight)
                   for i, (x, y, a) in enumerate(pts)])


# ---------------------------------------------------------------- 字形结构

class Bleed(NamedTuple):
    role: str  # 跟随所属笔画的颜色
    contour: np.ndarray
    alpha: float
    clip: np.ndarray  # 有界裁切区域（闭合轮廓）


class Glyph(NamedTuple):
    bar_thick: np.ndarray
    bar_thin: np.ndarray
    body: np.ndarray  # 中腰 + 上下卷叶，一条逆时针轮廓
    holes: tuple  # 飞白：顺时针子路径，位于粗柱或主体之内
    notch: tuple  # 中腰向左尖凹的顶点（应在第三线上）
    counters: tuple  # 上、下卷叶留白内各一个探测点
    terminals: tuple  # 上、下卷端圆团中心
    bleed: tuple = ()
    height: float = 4.0

    @property
    def outers(self):
        return (self.bar_thick, self.bar_thin, self.body)


def validate_glyph(g: Glyph, tol: float = 1e-6) -> None:
    """检查中音谱号的结构约束，不满足时抛 ValueError。"""
    half = g.height / 2
    named = {"bar_thick": g.bar_thick, "bar_thin": g.bar_thin, "body": g.body}
    named.update({f"hole{i}": h for i, h in enumerate(g.holes)})
    for name, c in named.items():
        c = np.asarray(c, float)
        if c.ndim != 2 or c.shape[1] != 2 or len(c) < 7 or (len(c) - 1) % 3:
            raise ValueError(f"{name} 不是三次 Bézier 链")
        if not np.isfinite(c).all():
            raise ValueError(f"{name} 含非有限坐标")
        if not np.allclose(c[0], c[-1], atol=tol):
            raise ValueError(f"{name} 没有闭合")
    for name in ("bar_thick", "bar_thin", "body"):
        if signed_area(named[name]) <= 0:
            raise ValueError(f"{name} 应为逆时针外轮廓")
    for i, h in enumerate(g.holes):
        if signed_area(h) >= 0:
            raise ValueError(f"飞白 {i} 应为顺时针子路径")
        if not (winding(flatten(h, 8), g.outers) == 1).all():
            raise ValueError(f"飞白 {i} 不在单层墨迹之内")
    boxes = {k: bbox_of([v]) for k, v in named.items() if not k.startswith("hole")}
    for name in ("bar_thick", "bar_thin"):
        x0, y0, x1, y1 = boxes[name]
        if abs(y0 + y1) > 0.05 * half or y1 - y0 < 1.8 * half:
            raise ValueError(f"{name} 应纵贯五线并以第三线 y=0 为中心")
        if y1 - y0 < 3 * (x1 - x0):
            raise ValueError(f"{name} 不是竖柱")
    if not boxes["bar_thick"][2] <= boxes["bar_thin"][0] + tol:
        raise ValueError("顺序应为：粗柱在细柱左侧")
    if not boxes["bar_thick"][2] - boxes["bar_thick"][0] > boxes["bar_thin"][2] - boxes["bar_thin"][0]:
        raise ValueError("左柱应比右柱粗")
    if not boxes["bar_thin"][2] <= boxes["body"][0] + tol:
        raise ValueError("主体应在细柱右侧")
    x0, y0, x1, y1 = boxes["body"]
    if y1 < 0.8 * half or y0 > -0.8 * half:
        raise ValueError("主体应有上下两片卷叶")
    span = max(x1 - x0, y1 - y0)
    for (cx, cy), sign in zip(g.counters, (1, -1)):
        if cy * sign <= 0:
            raise ValueError("留白探测点应分居第三线上下")
        if winding([(cx, cy)], g.outers)[0] != 0:
            raise ValueError("卷叶内部没有留白")
        for dx, dy in ((1, 0), (0, 1), (0, -1)):  # 右、上、下三面被墨包围，左侧可向尖叶开口
            ray = np.array([(cx + dx * t, cy + dy * t) for t in np.linspace(0, span, 300)])
            if not (winding(ray, [g.body]) > 0).any():
                raise ValueError("卷叶留白没有被外腹与上下回转包住")
    nx, ny = g.notch
    if abs(ny) > tol:
        raise ValueError("中腰尖凹顶点应在第三线 y=0")
    d = 0.02 * half
    if winding([(nx - d, 0.0)], [g.body])[0] != 1 or winding([(nx + d, 0.0)], [g.body])[0] != 0:
        raise ValueError("中腰应在第三线处向左尖凹：左侧是墨、右侧是留白")
    xs = np.linspace(boxes["bar_thick"][0], x1, 97) + 1e-4 * math.pi
    ys = np.linspace(-1.05 * half, 1.05 * half, 131) + 1e-4 * math.e
    grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    w = winding(grid, [*g.outers, *g.holes])
    if w.min() < 0 or w.max() > 1:
        raise ValueError("轮廓自交或互相重叠（nonzero 环绕数应只有 0/1）")


# ---------------------------------------------------------------- A 精致传统

def _a_upper(p: Params) -> tuple[np.ndarray, dict]:
    """上半：从第三线上的尖凹顶点出发，逆时针绕上卷叶，到腰部左端上角为止。"""
    w, H, W, xw = p.weight, p.half, p.width, p.x_waist
    L = W - xw
    X = lambda s: xw + s * L  # noqa: E731
    Y = lambda t: t * H  # noqa: E731
    belly = min(2.6 * w * (0.7 + 0.5 * p.contrast), 0.36 * L)
    hl = p.hair
    rb = min(1.1 * w, 0.15 * L) * (1 - 0.15 * p.opening)
    xb = X(0.13) + rb
    yb = Y(0.725 + 0.06 * p.opening)
    yb = min(yb, H - hl - 1.15 * rb - 0.03 * H)
    t_spike = 0.41 - 0.12 * p.opening
    y_cbot = Y(0.105) + 1.1 * hl  # 下回转内缘
    xi = W - belly  # 留白右缘
    tb = 0.7 * w  # 腰部左端半高
    c = 0.39  # 四分之一圆弧的控制柄（相对弦长）
    anchors = [
        K(X(0.29), 0, 62, 62, 0.3, 0.42),  # 尖凹顶点（与下半镜像后形成尖角）
        K(X(0.42), Y(0.115), 10, 10, 0.4, 0.4),
        K(X(0.62), Y(0.102), 0, 0, 0.36, 0.34),  # 下回转外缘最低点
        K(W, Y(0.56), 90, 90, 0.46, 0.42),  # 外腹最右
        K(X(0.53), H, 180, 180, 0.45, 0.36),  # 顶
        K(X(0.2), Y(0.93), 205, 205, 0.36, 0.4),
        K(xb - rb, yb, 270, 270, 0.4, c),  # 圆团最左
        K(xb, yb - rb, 0, 0, c, c),
        K(xb + rb, yb, 90, 90, c, c),
        K(xb + 0.6 * rb, yb + 0.8 * rb, 140, 140, 0.4, 0.4),
        K(xb - 0.12 * rb, yb + 1.12 * rb, 168, 12, 0.4, 0.3),  # 回钩尖
        K(X(0.45), H - hl, 0, 0, 0.4, 0.4),  # 上回转内缘
        K(xi, Y(0.6), 270, 270, 0.42, 0.42),  # 留白右缘
        K(X(0.5), y_cbot, 180, 180, 0.3, 0.42),  # 下回转内缘
        K(X(0.235), Y(t_spike), 108, 262, 0.55, 0.3),  # 尖叶
        K(xw, tb, 196, 270, 0.42, 0.3),  # 腰部左端上角
    ]
    info = dict(counter=((xb + rb + xi) / 2, (y_cbot + H - hl) / 2), ball=(xb, yb), rb=rb,
                belly_center=((xb + rb + xi) / 2, Y(0.5)), notch=(X(0.29), 0.0))
    return chain(anchors), info


def variant_a(p: Params) -> Glyph:
    """A 精致传统：上下对称，粗腹细回转，卷端圆团带回钩，中腰尖凹 + 尖叶。"""
    w, H = p.weight, p.half
    upper, info = _a_upper(p)
    xw, tb = p.x_waist, 0.7 * w
    body = join(upper, line((xw, tb), (xw, -tb)), mirror_rev(upper))
    thick = rounded_rect(0, -H, 2 * w, H, 0.22 * w)
    x_thin = 2 * w + p.gap
    thin = rounded_rect(x_thin, -H, x_thin + p.thin, H, 0.45 * p.thin)
    (bx, by), (cx, cy), cen = info["ball"], info["counter"], info["belly_center"]
    rng = np.random.default_rng([p.seed, 0xA])
    holes = _dry_holes(p, rng, thick, body, [cen, (cen[0], -cen[1])], strength=0.5)
    bleed = _ball_bleed(p, rng, [(bx, by, info["rb"]), (bx, -by, info["rb"])], strength=0.5)
    return Glyph(thick, thin, body, tuple(holes), info["notch"], ((cx, cy), (cx, -cy)),
                 ((bx, by), (bx, -by)), tuple(bleed), p.height)


# ---------------------------------------------------------------- B 自由水墨

def _jitter(rng, n, scale):
    return np.clip(rng.normal(0, scale, n), -2 * scale, 2 * scale)


def _shear(q) -> np.ndarray:
    q = np.array(q, float)
    q[..., 0] += SHEAR_B * q[..., 1]
    return q


def _b_body(p: Params, rng) -> tuple[np.ndarray, dict]:
    """B 主体：上卷叶大而重、圆团带尖钩；下卷叶收窄、末端为斜向泪滴；腰部左端是一个尖楔。"""
    w, H, W, xw = p.weight, p.half, p.width, p.x_waist
    L = W - xw
    j = _jitter(rng, 24, 0.010 * H)
    a = _jitter(rng, 12, 3.0)
    X = lambda s: xw + s * L  # noqa: E731
    Y = lambda t: t * H  # noqa: E731
    belly = min(2.9 * w * (0.7 + 0.5 * p.contrast), 0.38 * L)
    hl = p.hair
    rb = min(1.25 * w, 0.17 * L) * (1 - 0.15 * p.opening)
    xb, yb = X(0.10) + rb, Y(0.69 + 0.06 * p.opening)
    yb = min(yb, H - hl - 1.2 * rb - 0.03 * H)
    y_spike = min(Y(0.52 - 0.14 * p.opening), yb - rb - 0.02 * H)  # 尖叶不顶进被压低的圆团
    nx = X(0.33)
    xi = W - belly
    y_cbot = Y(0.12) + 1.0 * hl + max(0.0, j[1], j[3])  # 内缘随外缘手抖上移，发丝不被抖穿
    c = 0.39
    upper = [
        K(nx, 0, 72 + a[0], 72 + a[0], 0.3, 0.45),  # 尖凹顶点
        K(X(0.47) + j[0], Y(0.135) + j[1], 8, 8, 0.4, 0.4),
        K(X(0.70) + j[2], Y(0.125) + j[3], 5 + a[1], 5 + a[1]),
        K(X(0.92), Y(0.30) + j[4], 50, 50, 0.36, 0.34),
        K(W - SHEAR_B * Y(0.60), Y(0.60) + j[5], 90, 90, 0.34, 0.36),  # 外腹最右（偏上，像转折处的顿笔）
        K(X(0.78) + j[6], Y(0.955), 132 + a[2], 132 + a[2], 0.36, 0.36),  # 肩
        K(X(0.45) + j[7], H, 180, 180, 0.38, 0.4),  # 顶
        K(X(0.14), Y(0.90), 215, 215, 0.36, 0.4),
        K(xb - rb, yb + j[8], 268, 268, 0.4, c),  # 圆团最左
        K(xb + 0.05 * rb, yb - rb, 3, 3, c, c),
        K(xb + rb, yb + 0.05 * rb, 88, 88, c, c),
        K(xb + 0.55 * rb, yb + 0.85 * rb, 140, 140, 0.4, 0.4),
        K(xb - 0.32 * rb, yb + 1.22 * rb, 168, 18, 0.4, 0.3),  # 尖钩（回锋）
        K(X(0.45) + j[9], H - 1.05 * hl, 2, 2, 0.4, 0.4),
        K(xi + j[10], Y(0.62), 272 + a[3], 272 + a[3], 0.42, 0.42),
        K(X(0.5) + j[11], y_cbot, 183, 183, 0.3, 0.42),
        K(X(0.17) + j[12], y_spike, 112, 248, 0.6, 0.32),  # 长尖叶，向左斜挑
        K(X(0.03), Y(0.10), 205, 205, 0.4, 0.3),
    ]
    # 下半在“上半坐标系”里设计，再镜像倒序接回；与上半刻意不同
    b_r = 0.86 * rb
    xl, yl = X(0.16) + b_r, Y(0.745 + 0.05 * p.opening)
    yl = min(yl, H - hl - 1.25 * b_r - 0.03 * H)
    W_lo = W - 0.06 * L
    xi_lo = W_lo - 0.92 * belly
    t_spike_lo = 0.33 - 0.10 * p.opening
    y_cbot_lo = Y(0.11) + 1.1 * hl + max(0.0, j[14])
    lower = [
        K(nx, 0, 52 + a[4], 52 + a[4], 0.3, 0.45),
        K(X(0.45) + j[13], Y(0.11) + j[14], 2, 2, 0.4, 0.4),
        K(X(0.76) + j[15], Y(0.15), 20, 20, 0.36, 0.4),
        K(W_lo, Y(0.52) + j[16], 90, 90, 0.42, 0.4),
        K(X(0.52) + j[17], H * 0.985, 180 + a[5], 180 + a[5], 0.45, 0.36),
        K(xl - 0.55 * b_r, yl + 0.9 * b_r, 222, 222, 0.4, 0.4),  # 泪滴：从发丝斜落下来
        K(xl - b_r, yl - 0.1 * b_r, 265, 265, c, c),
        K(xl + 0.1 * b_r, yl - b_r, 0, 0, c, c),
        K(xl + b_r, yl + 0.1 * b_r, 92, 92, c, 0.4),
        K(xl + 0.62 * b_r, yl + 1.02 * b_r, 122, 32, 0.4, 0.3),  # 钝钩：不回尖，直接接发丝
        K(X(0.52) + j[18], H * 0.985 - 1.1 * hl, 0, 0, 0.45, 0.4),
        K(xi_lo + j[19], Y(0.52), 270, 270, 0.42, 0.42),
        K(X(0.5) + j[20], y_cbot_lo, 180 + a[6], 180 + a[6], 0.3, 0.42),
        K(X(0.25) + j[21], Y(t_spike_lo), 118, 235, 0.5, 0.35),  # 短而钝的下尖叶
        K(X(0.035), Y(0.09), 200, 200, 0.4, 0.3),
    ]
    tip = K(xw, Y(0.012), 245, 295, 0.3, 0.3)  # 腰部左端尖楔
    up_c, lo_c = chain(upper), chain(lower)
    left = chain([upper[-1], tip, (lambda k: (k[0], -k[1], 180 - k[3], 180 - k[2], k[5], k[4]))(lower[-1])])
    body = _shear(join(up_c, left, mirror_rev(lo_c)))  # 整体右倾：y=0 上的点不动
    pt = lambda x, y: tuple(_shear([x, y]))  # noqa: E731
    info = dict(
        notch=(nx, 0.0), ball=pt(xb, yb), rb=rb, ball_lo=pt(xl, -yl), rb_lo=b_r,
        counter=pt((xb + rb + xi) / 2, (y_cbot + H - hl) / 2),
        counter_lo=pt((xl + b_r + xi_lo) / 2, -(y_cbot_lo + 0.985 * H - hl) / 2),
        belly_center=pt((xb + rb + xi) / 2, Y(0.52)), belly_center_lo=pt((xl + b_r + xi_lo) / 2, -Y(0.48)),
    )
    return body, info


def _b_bars(p: Params, rng) -> tuple[np.ndarray, np.ndarray]:
    """B 竖柱：粗柱斜切起笔、中段略收、下端顿笔；细柱顶端藏锋小顿、向下收成悬针。"""
    w, H = p.weight, p.half
    j = _jitter(rng, 8, 0.04 * w)
    thick = closed([
        K(0.95 * w, -0.995 * H, 0, 0, 0.4, 0.4),  # 收笔底
        K(2.0 * w - abs(j[0]), -0.90 * H, 70, 88, 0.45, 0.3),
        K(1.9 * w - abs(j[1]), 0.0, 92, 92, 0.4, 0.4),  # 中段略收
        K(1.98 * w - abs(j[2]), 0.86 * H, 96, 96, 0.3, 0.4),
        K(1.72 * w, 0.975 * H, 160, 168, 0.4, 0.4),
        K(-0.06 * w, 1.025 * H, 175, 255, 0.45, 0.35),  # 斜切起笔尖
        K(0.04 * w + j[3], 0.80 * H, 272, 272, 0.4, 0.4),
        K(0.10 * w + j[4], 0.0, 270, 270, 0.4, 0.4),
        K(0.0 + j[5], -0.88 * H, 262, 300, 0.3, 0.45),
    ])
    t, x = p.thin, 2 * w + p.gap
    thin = closed([
        K(x + 0.42 * t, -1.03 * H, 284, 80, 0.3, 0.3),  # 悬针尖
        K(x + 0.97 * t - abs(j[6]), -0.2 * H, 92, 92, 0.4, 0.4),
        K(x + 0.96 * t, 0.86 * H, 90, 90, 0.4, 0.3),
        K(x + 0.5 * t, 1.0 * H, 172, 180, 0.3, 0.3),  # 顶端小顿
        K(x + 0.03 * t, 0.9 * H, 262, 270, 0.5, 0.4),
        K(x + 0.06 * t + abs(j[7]), -0.2 * H, 270, 270, 0.4, 0.4),
        K(x + 0.25 * t, -0.85 * H, 285, 285, 0.1, 0.3),
    ])
    return thick, thin


def variant_b(p: Params) -> Glyph:
    """B 非对称自由水墨：独立的笔锋起收、上重下轻的卷叶、斜挑尖叶，按 seed 有轻微手抖。"""
    rng = np.random.default_rng([p.seed, 0xB])
    body, info = _b_body(p, rng)
    thick, thin = _b_bars(p, rng)
    holes = _dry_holes(p, rng, thick, body, [info["belly_center"], info["belly_center_lo"]], strength=1.0)
    balls = [(*info["ball"], info["rb"]), (*info["ball_lo"], info["rb_lo"])]
    bleed = _ball_bleed(p, rng, balls, strength=1.0) + _bar_bleed(p, rng, thick)
    return Glyph(thick, thin, body, tuple(holes), info["notch"], (info["counter"], info["counter_lo"]),
                 (info["ball"], info["ball_lo"]), tuple(bleed), p.height)


# ---------------------------------------------------------------- 飞白与墨晕

def _ray_band(poly: np.ndarray, origin, direction):
    """射线与多边形的前两个交点距离（进入墨迹、离开墨迹）。"""
    a, b = poly, np.roll(poly, -1, 0)
    d = np.asarray(direction, float)
    e = b - a
    den = d[0] * e[:, 1] - d[1] * e[:, 0]
    ok = np.abs(den) > 1e-12
    oa = a - np.asarray(origin, float)
    t = np.where(ok, (oa[:, 0] * e[:, 1] - oa[:, 1] * e[:, 0]) / np.where(ok, den, 1), -1)
    u = np.where(ok, (oa[:, 0] * d[1] - oa[:, 1] * d[0]) / np.where(ok, den, 1), -1)
    hit = np.sort(t[ok & (t > 1e-9) & (u >= 0) & (u < 1)])
    return (hit[0], hit[1]) if len(hit) >= 2 else None


def _streaks(rng, poly, rays, n, length, width, lanes=(0.15, 0.85)) -> list[np.ndarray]:
    """沿一族射线定义的笔带生成一束顺笔向的飞白丝（顺时针闭合 Bézier）。

    rays(u) → (origin, direction)，u ∈ [0, 1] 沿运笔方向；射线与墨迹的进/出点给出笔带横向坐标 0..1。
    飞白丝落在 lanes 范围内互不重叠的若干条“车道”上，两端尖、中段宽，长短不一，都收在运笔后段。
    """
    lo, hi = lanes
    step = (hi - lo) / n
    out = []
    for i in range(n):
        wf = min(width * rng.uniform(0.5, 1.0), 0.6 * step)
        f0 = lo + (i + 0.5) * step + rng.uniform(-0.15, 0.15) * (step - wf)
        u1 = rng.uniform(0.82, 1.0)
        u0 = max(0.0, u1 - length * rng.uniform(0.45, 1.0))
        us = np.linspace(u0, u1, 22)
        left, right = [], []
        for k, u in enumerate(us):
            o, d = (np.asarray(v, float) for v in rays(u))
            band = _ray_band(poly, o, d)
            if band is None:
                break
            sk = k / (len(us) - 1)
            half = 0.5 * wf * math.sin(math.pi * sk) ** 0.7
            r_in, r_out = band
            left.append(o + (r_in + (f0 - half) * (r_out - r_in)) * d)
            right.append(o + (r_in + (f0 + half) * (r_out - r_in)) * d)
        if len(left) < len(us):
            continue
        c = catmull_closed(np.vstack([left, right[-2:0:-1]]))
        if signed_area(c) > 0:
            c = c[::-1]
        if abs(signed_area(c)) > 1e-6:
            out.append(c)
    return out


def _dry_holes(p: Params, rng, thick, body, bellies, strength: float) -> list[np.ndarray]:
    """飞白：粗柱下段一束、上下外腹的运笔后段各一束，都偏笔画外侧；strength 让 A 比 B 收敛。"""
    d = p.dryness * strength
    if d <= 0:
        return []
    H, w = p.half, p.weight
    holes = _streaks(rng, flatten(thick), lambda u: ((-w, (0.25 - 1.15 * u) * H), (1.0, 0.0)),
                     2 + round(7 * d), length=0.3 + 0.5 * d, width=0.03 + 0.05 * d, lanes=(0.5, 0.9))
    poly_body = flatten(body)
    for (cx, cy), a0, a1 in ((bellies[0], -50, 75), (bellies[1], 50, -75)):
        def rays(u, cx=cx, cy=cy, a0=a0, a1=a1):
            a = math.radians(a0 + (a1 - a0) * u)
            return (cx, cy), (math.cos(a), math.sin(a))

        holes += _streaks(rng, poly_body, rays, 2 + round(8 * d), length=0.3 + 0.45 * d,
                          width=0.03 + 0.05 * d, lanes=(0.45, 0.9))
    outers = (thick, body)
    return [h for h in holes if (winding(flatten(h, 8), outers) == 1).all()]


def _blob(rng, cx, cy, rx, ry, wobble=0.05, n=14) -> np.ndarray:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k = 1 + wobble * (np.sin(2 * th + rng.uniform(0, 6.3)) + 0.6 * np.sin(3 * th + rng.uniform(0, 6.3)))
    return catmull_closed(np.column_stack([cx + rx * k * np.cos(th), cy + ry * k * np.sin(th)]))


def _ball_bleed(p: Params, rng, balls, strength: float) -> list[Bleed]:
    """卷端圆团处的墨晕：两圈低透明度外扩，只在圆团偏外下侧的有界圆盘内可见。"""
    b = p.bleed * strength
    if b <= 0:
        return []
    out = []
    for x, y, r in balls:
        sy = 1 if y > 0 else -1
        clip = _blob(rng, x - 0.35 * r, y - sy * 0.25 * r, 1.5 * r, 1.5 * r, wobble=0.0, n=12)
        for k in (1, 2):
            grow = r * (1 + (0.12 + 0.16 * k) * b)
            out.append(Bleed("fg", _blob(rng, x, y, grow, grow), 0.05 + 0.10 * b / k, clip))
    return out


def _bar_bleed(p: Params, rng, thick) -> list[Bleed]:
    """粗柱起笔处的墨晕，裁切在柱顶附近。"""
    if p.bleed <= 0:
        return []
    w, H = p.weight, p.half
    clip = rounded_rect(-0.5 * w, 0.84 * H, 2.4 * w, 1.1 * H, 0.3 * w)
    return [Bleed("fg", _blob(rng, 0.85 * w, 0.975 * H, w * (1.0 + 0.12 * p.bleed * k), 0.035 * H * (1 + 0.6 * p.bleed * k)),
                  0.05 + 0.10 * p.bleed / k, clip) for k in (1, 2)]


# ---------------------------------------------------------------- 方案注册

VARIANTS = {
    "a": ("A refined", variant_a),
    "b": ("B free ink", variant_b),
}
REMOVED_VARIANTS = {"c": "旧方案 C（staff-fused 谱线融合）已删除，不再提供，也不映射到其他方案"}


def get_variant(key: str):
    if key in REMOVED_VARIANTS:
        raise ValueError(REMOVED_VARIANTS[key])
    if key not in VARIANTS:
        raise ValueError(f"未知方案 {key!r}，可选 {tuple(VARIANTS)}")
    return VARIANTS[key][1]


def silhouette(g: Glyph) -> Glyph:
    """去掉飞白与墨晕的纯剪影，用于检查轮廓本身。"""
    return g._replace(holes=(), bleed=())


# ---------------------------------------------------------------- 配色与渲染

class Palette(NamedTuple):
    name: str
    fg: str
    accent: str
    bg: str


PALETTES = {
    "ink": Palette("墨蓝·朱红·米白", fg="#1E3550", accent="#C8402A", bg="#F4EEE1"),
    "teal": Palette("深青·琥珀·暖白", fg="#0E4B48", accent="#D8962A", bg="#FAF5EC"),
    "cobalt": Palette("炭黑·钴蓝·白", fg="#25262A", accent="#2D5FD8", bg="#FFFFFF"),
    "sumi": Palette("墨黑·纸白·朱红", fg="#171513", accent="#B8322A", bg="#F7F3EA"),
}
MONO = Palette("单色", fg="#111111", accent="#111111", bg="#FFFFFF")


def get_palette(palette) -> Palette:
    return PALETTES[palette] if isinstance(palette, str) else palette


class Shape(NamedTuple):
    role: str  # "fg" 或 "accent"
    path: MplPath
    alpha: float = 1.0
    clip: MplPath | None = None


def glyph_shapes(g: Glyph) -> list[Shape]:
    """渲染顺序：墨晕（低透明度、裁切）→ 主墨（粗柱 + 主体 + 顺时针飞白）→ 细柱点缀色。"""
    shapes = [Shape(b.role, to_path([b.contour]), b.alpha, to_path([b.clip])) for b in g.bleed]
    shapes.append(Shape("fg", to_path([g.bar_thick, g.body, *g.holes])))
    shapes.append(Shape("accent", to_path([g.bar_thin])))
    return shapes


def glyph_bbox(g: Glyph) -> tuple[float, float, float, float]:
    return bbox_of(g.outers)


def square_bbox(bbox, pad: float = 0.12) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2


def union_bbox(boxes) -> tuple[float, float, float, float]:
    b = np.array(list(boxes))
    return (*b[:, :2].min(0), *b[:, 2:].max(0))


def draw_glyph(ax, g: Glyph, palette, bbox=None) -> None:
    pal = get_palette(palette)
    x0, y0, x1, y1 = square_bbox(glyph_bbox(g)) if bbox is None else bbox
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=pal.bg, lw=0, zorder=0))
    for s in glyph_shapes(g):
        patch = PathPatch(s.path, facecolor=getattr(pal, s.role), edgecolor="none", alpha=s.alpha,
                          zorder=1 if s.alpha < 1 else 2)
        ax.add_patch(patch)
        if s.clip is not None:
            patch.set_clip_path(s.clip, ax.transData)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")


def render(g: Glyph, palette="ink", size: float = 4.0, bbox=None) -> Figure:
    fig = Figure(figsize=(size, size), facecolor=get_palette(palette).bg)
    FigureCanvasAgg(fig)
    draw_glyph(fig.add_axes((0, 0, 1, 1)), g, palette, bbox)
    return fig


def rasterize(g: Glyph, palette="ink", px: int = 64) -> np.ndarray:
    """按精确像素尺寸栅格化，返回 (px, px, 4) uint8 数组，用于检查小尺寸效果。"""
    fig = render(g, palette, size=1.0)
    fig.set_dpi(px)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def common_bbox(p: Params) -> tuple[float, float, float, float]:
    return square_bbox(union_bbox(glyph_bbox(fn(p)) for _, fn in VARIANTS.values()))


def silhouette_figure(p: Params) -> Figure:
    """两方案纯黑剪影，同一取景框，叠五线谱与第三线。"""
    bbox = common_bbox(p)
    fig = Figure(figsize=(3.2 * len(VARIANTS), 3.6), facecolor="white")
    axes = fig.subplots(1, len(VARIANTS), squeeze=False)[0]
    for ax, (label, fn) in zip(axes, VARIANTS.values()):
        draw_glyph(ax, silhouette(fn(p)), MONO, bbox)
        for k in range(-2, 3):
            ax.axhline(k * p.height / 4, color="#C8402A" if k == 0 else "#9AA0A6", lw=1.0 if k == 0 else 0.5, zorder=3)
        ax.set_title(f"{label} silhouette", fontsize=10)
    fig.tight_layout()
    return fig


def compare_figure(p: Params, palettes=None) -> Figure:
    """行 = 两方案（同一尺度），列 = 四配色 + 单色。"""
    palettes = [*PALETTES.values(), MONO] if palettes is None else palettes
    bbox = common_bbox(p)
    fig = Figure(figsize=(2.4 * len(palettes), 2.6 * len(VARIANTS)), facecolor="white")
    axes = fig.subplots(len(VARIANTS), len(palettes), squeeze=False)
    keys = {pal: key for key, pal in PALETTES.items()}
    for row, (label, fn) in zip(axes, VARIANTS.values()):
        g = fn(p)
        for ax, pal in zip(row, palettes):
            draw_glyph(ax, g, pal, bbox)
            ax.set_title(f"{label} / {keys.get(pal, 'mono')}", fontsize=9)
    fig.tight_layout()
    return fig


def breakdown_figure(p: Params, palette="ink") -> Figure:
    """逐部件拆解：每行一个方案；粗柱、细柱、主体（标尖凹与留白）、飞白、组合 + 五线谱。"""
    pal = get_palette(palette)
    bbox = common_bbox(p)
    titles = ["thick bar", "thin bar", "body: notch + counters", "dry-brush holes", "assembled on staff"]
    fig = Figure(figsize=(2.9 * len(titles), 3.1 * len(VARIANTS)), facecolor="white")
    axes = fig.subplots(len(VARIANTS), len(titles), squeeze=False)
    for row, (label, fn) in zip(axes, VARIANTS.values()):
        g = fn(p)
        for i, (ax, title) in enumerate(zip(row, titles)):
            ax.add_patch(Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], color=pal.bg, lw=0, zorder=0))
            for k in range(-2, 3):
                ax.axhline(k * p.height / 4, color="#9AA0A6", lw=0.5, zorder=0.5)
            ghost = to_path(g.outers)
            ax.add_patch(PathPatch(ghost, fc="none", ec="#B9BEC5", lw=0.6, zorder=1))
            if i == 0:
                ax.add_patch(PathPatch(to_path([g.bar_thick]), fc=pal.fg, ec="none", zorder=2))
            elif i == 1:
                ax.add_patch(PathPatch(to_path([g.bar_thin]), fc=pal.accent, ec="none", zorder=2))
            elif i == 2:
                ax.add_patch(PathPatch(to_path([g.body]), fc=pal.fg, ec="none", zorder=2))
                ax.plot(*g.notch, "o", ms=5, color=pal.accent, mec="white", zorder=3)
                for c in g.counters:
                    ax.plot(*c, "x", ms=6, mew=1.6, color=pal.accent, zorder=3)
            elif i == 3:
                ax.add_patch(PathPatch(to_path([g.bar_thick, g.body]), fc="#D8D2C6", ec="none", zorder=2))
                if g.holes:
                    ax.add_patch(PathPatch(to_path(g.holes), fc=pal.accent, ec="none", zorder=3))
            else:
                draw_glyph(ax, g, pal, bbox)
                ax.axhline(0, color=pal.accent, lw=1.2, zorder=4)
            ax.set_xlim(bbox[0], bbox[2])
            ax.set_ylim(bbox[1], bbox[3])
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{label[0]} · {title}", fontsize=9)
    fig.tight_layout()
    return fig


def export_all(outdir="out", p: Params | None = None, palette="ink") -> dict[str, Path]:
    p = Params() if p is None else p
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = {}
    for key, (_, fn) in VARIANTS.items():
        fig = render(fn(p), palette)
        for ext in ("png", "svg"):
            path = outdir / f"variant_{key}.{ext}"
            fig.savefig(path, dpi=128, facecolor=fig.get_facecolor(), metadata={"Date": None} if ext == "svg" else None)
            written[path.name] = path
    for name, fig in (("compare.png", compare_figure(p)), ("strokes_breakdown.png", breakdown_figure(p, palette))):
        fig.savefig(outdir / name, dpi=110, facecolor="white")
        written[name] = outdir / name
    return written


if __name__ == "__main__":
    for path in export_all().values():
        print(path)
