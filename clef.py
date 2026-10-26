"""中音谱号（C 谱号）个人 Logo：共享基础笔画 + 三种方案 + 渲染导出。

坐标约定：五线谱线距 = height / 4，第三线在 y = 0，五条线位于 y = k·height/4（k = -2..2）。
左侧粗、细两条竖线，右侧上下两瓣弧线在 (x0, 0) 收束——这就是中音谱号把 C 指在第三线上的位置。

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
BEAK_RISE = 0.3  # 弧瓣碗底高出第三线的量（相对竖向半径），形成指向左侧的喙
LOBE_RX_RATIO = 0.62  # 椭圆横向半径占弧瓣总宽的比例；其余留给喙
BALL_RATIO = 0.8  # A/C 方案弧瓣末端圆点半径（相对 weight）
STROKE_NAMES = ("bar_thick", "bar_thin", "lobe_upper", "lobe_lower")


# ---------------------------------------------------------------- 参数

@dataclass(frozen=True)
class Params:
    height: float = 4.0  # 竖线总高（= 五线谱上下两线的距离）
    width: float = 3.0  # 整体宽度（含笔画粗细）
    weight: float = 0.3  # 基础笔画粗细；粗竖线为 2 倍
    opening: float = 0.25  # 弧瓣末端开口，0 = 闭合回到竖线侧，OPENING_MAX = 最大开口
    gap: float = 0.3  # 粗细竖线之间、细竖线与弧瓣之间的间距

    def __post_init__(self):
        for name in ("height", "width", "weight", "opening", "gap"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{name} 必须是有限数值，得到 {v!r}")
        if self.height <= 0:
            raise ValueError("height 必须 > 0")
        if not 0 < self.weight <= self.height / 8:
            raise ValueError(f"weight 必须在 (0, height/8 = {self.height / 8:g}] 内")
        if not 0 <= self.opening <= OPENING_MAX:
            raise ValueError(f"opening 必须在 [0, {OPENING_MAX}] 内")
        if not 0 <= self.gap <= self.height / 4:
            raise ValueError(f"gap 必须在 [0, height/4 = {self.height / 4:g}] 内")
        need = self.min_width()
        if self.width < need:
            raise ValueError(f"width 太小：当前粗细/间距下至少需要 {need:.3f}")
        if self.width > self.height:
            raise ValueError(f"width 不应超过 height = {self.height:g}（再宽弧瓣会被压扁）")

    @property
    def x0(self) -> float:
        """弧瓣收束点（喙点）的 x 坐标。"""
        return 3 * self.weight + 2 * self.gap

    @property
    def lobe_rx(self) -> float:
        return self.width - self.x0 - self.weight / 2

    def min_width(self) -> float:
        # 弧瓣总宽至少 2.5·weight，保证 B 方案内切后仍有实心
        return self.x0 + 3 * self.weight


# ---------------------------------------------------------------- 基础笔画

def base_strokes(p: Params, n: int = 121) -> dict[str, np.ndarray]:
    """四条骨架线（中心线），所有方案都从这里派生。"""
    w, half = p.weight, p.height / 2
    bars = {}
    for name, xc, bw in (("bar_thick", w, 2 * w), ("bar_thin", 2.5 * w + p.gap, w)):
        bars[name] = np.array([[xc, -half + bw / 2], [xc, half - bw / 2]])
    # 上弧瓣 = 从喙点 (x0, 0) 出发的切线 + 椭圆弧（反写的 C 形碗，开口朝左）
    cx, cy, rx, ry, phi0, phi1 = lobe_ellipse(p)
    m = max(n // 6, 4)
    line = np.linspace([p.x0, 0.0], [cx + rx * math.cos(phi0), cy + ry * math.sin(phi0)], m, endpoint=False)
    phi = np.linspace(phi0, phi1, n - m)
    arc = np.column_stack([cx + rx * np.cos(phi), cy + ry * np.sin(phi)])
    upper = np.vstack([line, arc])
    return {**bars, "lobe_upper": upper, "lobe_lower": upper * [1.0, -1.0]}


def lobe_ellipse(p: Params) -> tuple[float, float, float, float, float, float]:
    """上弧瓣骨架椭圆 (cx, cy, rx, ry) 及弧段起止参数角 (phi0 = 喙切点, phi1 = 末端)。"""
    w, half = p.weight, p.height / 2
    ry = (half - w / 2) / (2 + BEAK_RISE)  # 顶端外缘恰好落在 y = half
    rx = LOBE_RX_RATIO * p.lobe_rx  # 最右外缘恰好落在 x = width
    cx, cy = p.x0 + p.lobe_rx - rx, (1 + BEAK_RISE) * ry
    # 喙点到椭圆的下侧切点（在单位圆坐标里求）
    u, v = (p.x0 - cx) / rx, -cy / ry
    phi0 = math.atan2(v, u) + math.acos(1 / math.hypot(u, v))
    phi1 = math.radians(120 - 100 * p.opening)  # opening 越大，顶端越早结束
    return cx, cy, rx, ry, phi0, phi1


def stroke_width(name: str, p: Params) -> float:
    return 2 * p.weight if name == "bar_thick" else p.weight


def validate_strokes(strokes: dict[str, np.ndarray], tol: float = 1e-9) -> None:
    """检查中音谱号的结构约束，不满足时抛 ValueError。"""
    if set(strokes) != set(STROKE_NAMES):
        raise ValueError(f"笔画应为 {STROKE_NAMES}，得到 {tuple(strokes)}")
    for name, pts in strokes.items():
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            raise ValueError(f"{name} 不是 (N, 2) 折线")
        if not np.isfinite(pts).all():
            raise ValueError(f"{name} 含非有限坐标")
    thick, thin = strokes["bar_thick"], strokes["bar_thin"]
    up, lo = strokes["lobe_upper"], strokes["lobe_lower"]
    for name, bar in (("bar_thick", thick), ("bar_thin", thin)):
        if np.ptp(bar[:, 0]) > tol:
            raise ValueError(f"{name} 不是竖线")
        if abs(bar[:, 1].min() + bar[:, 1].max()) > tol:
            raise ValueError(f"{name} 没有以第三线 y=0 为中心")
    if not thick[0, 0] < thin[0, 0] < up[:, 0].min() - tol:
        raise ValueError("顺序应为：粗竖线 < 细竖线 < 弧瓣")
    if up.shape != lo.shape or not np.allclose(lo, up * [1, -1], atol=tol, rtol=0):
        raise ValueError("下弧瓣不是上弧瓣关于 y=0 的镜像")
    if abs(up[0, 1]) > tol or up[0, 0] > up[:, 0].min() + tol:
        raise ValueError("上下弧瓣必须在第三线 y=0 的最左点收束")
    if up[:, 1].min() < -tol or up[:, 1].max() <= tol:
        raise ValueError("上弧瓣必须整体位于第三线之上")


# ---------------------------------------------------------------- 轮廓工具

def _normals(pts: np.ndarray) -> np.ndarray:
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.column_stack([-t[:, 1], t[:, 0]])


def _arc(center, r, a0, a1, n=14) -> np.ndarray:
    a = np.linspace(a0, a1, n)
    return np.column_stack([center[0] + r * np.cos(a), center[1] + r * np.sin(a)])


def stroke_outline(pts: np.ndarray, left, right=None, round_caps: bool = True) -> np.ndarray:
    """折线 → 闭合多边形。left/right 为左右两侧偏移（标量或逐点数组）。"""
    pts = np.asarray(pts, dtype=float)
    right = left if right is None else right
    nrm = _normals(pts)
    lw = np.broadcast_to(np.asarray(left, dtype=float), len(pts))[:, None]
    rw = np.broadcast_to(np.asarray(right, dtype=float), len(pts))[:, None]
    lft, rgt = pts + nrm * lw, pts - nrm * rw
    parts = [lft]
    if round_caps:
        a_end = math.atan2(nrm[-1, 1], nrm[-1, 0])
        parts.append(_arc(pts[-1], float(lw[-1, 0]), a_end, a_end - np.pi)[1:-1])
    parts.append(rgt[::-1])
    if round_caps:
        a_st = math.atan2(nrm[0, 1], nrm[0, 0])
        parts.append(_arc(pts[0], float(lw[0, 0]), a_st + np.pi, a_st)[1:-1])
    return np.vstack(parts)


def rect(x0, y0, x1, y1) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


class Shape(NamedTuple):
    role: str  # "fg" 或 "accent"
    poly: np.ndarray


# ---------------------------------------------------------------- 三种方案

def variant_a(p: Params) -> list[Shape]:
    """圆弧线描：等宽圆头线，弧瓣末端加圆点收尾，细竖线作点缀色。"""
    s = base_strokes(p)
    validate_strokes(s)
    roles = {"bar_thin": "accent"}
    shapes = [Shape(roles.get(k, "fg"), stroke_outline(s[k], stroke_width(k, p) / 2)) for k in STROKE_NAMES]
    for k in ("lobe_upper", "lobe_lower"):
        shapes.append(Shape("fg", _arc(s[k][-1], BALL_RATIO * p.weight, 0, 2 * np.pi, 48)[:-1]))
    return shapes


def variant_b(p: Params) -> list[Shape]:
    """实心切割：方头实心竖线 + 「实心椭圆减去内椭圆」得到的月牙弧瓣，喙为尖角。"""
    s = base_strokes(p)
    validate_strokes(s)
    w, half = p.weight, p.height / 2
    shapes = []
    for k in ("bar_thick", "bar_thin"):
        xc, bw = s[k][0, 0], stroke_width(k, p)
        shapes.append(Shape("fg", rect(xc - bw / 2, -half, xc + bw / 2, half)))
    cx, cy, rx, ry, phi0, phi1 = lobe_ellipse(p)
    phi = np.linspace(phi0, phi1, 96)
    # 外椭圆 = 骨架外扩 w/2；内椭圆同心，横向切进 1.5w（右侧厚 2w），竖向只留 0.3w（顶端收尖）
    rxi, ryi = max(rx - 1.5 * w, 0.2 * rx), ry + 0.2 * w
    outer = np.column_stack([cx + (rx + w / 2) * np.cos(phi), cy + (ry + w / 2) * np.sin(phi)])
    inner = np.column_stack([cx + rxi * np.cos(phi), cy + ryi * np.sin(phi)])
    upper = np.vstack([[p.x0, 0.0], outer, inner[::-1]])
    shapes.append(Shape("accent", upper))
    shapes.append(Shape("accent", upper[::-1] * [1.0, -1.0]))
    return shapes


def variant_c(p: Params, n_samples: int = 1201) -> list[Shape]:
    """谱线融合：五线谱穿过并避让笔画，第三线从喙点 (x0, 0) 向右延伸。"""
    shapes = variant_a(p)
    s = base_strokes(p)
    w, gap, half = p.weight, p.gap, p.height / 2
    line_w = 0.4 * w
    ext = p.height / 8
    x_left, x_right = -ext, p.width + ext
    xs = np.linspace(x_left, x_right, n_samples)
    for k in (-2, -1, 1, 2):
        y = k * p.height / 4
        q = np.column_stack([xs, np.full_like(xs, y)])
        blocked = np.zeros(len(xs), dtype=bool)
        for name in STROKE_NAMES:
            clear = stroke_width(name, p) / 2 + gap / 2 + line_w / 2
            blocked |= _dist_to_polyline(q, s[name]) < clear
        for name in ("lobe_upper", "lobe_lower"):  # 末端圆点
            blocked |= np.linalg.norm(q - s[name][-1], axis=1) < BALL_RATIO * w + gap / 2 + line_w / 2
        for i0, i1 in _runs(~blocked):
            if xs[i1] - xs[i0] >= 1.5 * line_w:
                shapes.append(Shape("fg", rect(xs[i0], y - line_w / 2, xs[i1], y + line_w / 2)))
    shapes.append(Shape("accent", rect(p.x0, -line_w / 2, x_right, line_w / 2)))
    return shapes


VARIANTS = {
    "a": ("A arc-line", variant_a),
    "b": ("B solid-cut", variant_b),
    "c": ("C staff-fused", variant_c),
}


def _dist_to_polyline(q: np.ndarray, pts: np.ndarray) -> np.ndarray:
    a, b = pts[:-1], pts[1:]
    ab = b - a
    t = np.einsum("qsk,sk->qs", q[:, None, :] - a[None], ab) / np.maximum((ab**2).sum(1), 1e-300)
    proj = a[None] + np.clip(t, 0, 1)[..., None] * ab[None]
    return np.linalg.norm(q[:, None, :] - proj, axis=2).min(axis=1)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(s, e - 1) for s, e in zip(idx[::2], idx[1::2])]


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
}
MONO = Palette("单色", fg="#111111", accent="#111111", bg="#FFFFFF")


def get_palette(palette) -> Palette:
    return PALETTES[palette] if isinstance(palette, str) else palette


def shapes_bbox(shapes: list[Shape]) -> tuple[float, float, float, float]:
    allp = np.vstack([s.poly for s in shapes])
    return (*allp.min(axis=0), *allp.max(axis=0))


def square_bbox(bbox, pad: float = 0.12) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2


def draw_shapes(ax, shapes: list[Shape], palette, bbox=None) -> None:
    pal = get_palette(palette)
    x0, y0, x1, y1 = square_bbox(shapes_bbox(shapes)) if bbox is None else bbox
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=pal.bg, lw=0, zorder=0))
    for s in shapes:
        path = MplPath(np.vstack([s.poly, s.poly[:1]]), closed=True)
        ax.add_patch(PathPatch(path, facecolor=getattr(pal, s.role), edgecolor="none", zorder=1))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")


def render(shapes: list[Shape], palette="ink", size: float = 4.0, bbox=None) -> Figure:
    fig = Figure(figsize=(size, size), facecolor=get_palette(palette).bg)
    FigureCanvasAgg(fig)
    draw_shapes(fig.add_axes((0, 0, 1, 1)), shapes, palette, bbox)
    return fig


def rasterize(shapes: list[Shape], palette="ink", px: int = 64) -> np.ndarray:
    """按精确像素尺寸栅格化，返回 (px, px, 4) uint8 数组，用于检查小尺寸效果。"""
    fig = render(shapes, palette, size=1.0)
    fig.set_dpi(px)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def common_bbox(p: Params) -> tuple[float, float, float, float]:
    boxes = np.array([shapes_bbox(fn(p)) for _, fn in VARIANTS.values()])
    return square_bbox((*boxes[:, :2].min(0), *boxes[:, 2:].max(0)))


def compare_figure(p: Params, palettes=None) -> Figure:
    """行 = 三方案（同一尺度），列 = 三配色 + 单色。"""
    palettes = [*PALETTES.values(), MONO] if palettes is None else palettes
    bbox = common_bbox(p)
    fig = Figure(figsize=(2.4 * len(palettes), 2.6 * len(VARIANTS)), facecolor="white")
    axes = fig.subplots(len(VARIANTS), len(palettes), squeeze=False)
    keys = {pal: key for key, pal in PALETTES.items()}
    for row, (label, fn) in zip(axes, VARIANTS.values()):
        shapes = fn(p)
        for ax, pal in zip(row, palettes):
            draw_shapes(ax, shapes, pal, bbox)
            ax.set_title(f"{label} / {keys.get(pal, 'mono')}", fontsize=9)
    fig.tight_layout()
    return fig


def breakdown_figure(p: Params, palette="ink") -> Figure:
    """逐笔拆解：每条骨架线（虚线）及其轮廓，最后一格为组合并标出第三线。"""
    pal = get_palette(palette)
    s = base_strokes(p)
    a_all = variant_a(p)
    a_shapes = dict(zip(STROKE_NAMES, a_all))  # 前四个是笔画，其后是圆点收尾
    bbox = square_bbox((-p.height / 8, -p.height / 2, p.width + p.height / 8, p.height / 2), pad=0.08)
    titles = ["1 thick bar", "2 thin bar", "3 upper lobe", "4 lower lobe (mirror)", "5 assembled, C on line 3"]
    fig = Figure(figsize=(3 * len(titles), 3.4), facecolor="white")
    axes = fig.subplots(1, len(titles))
    for i, (ax, title) in enumerate(zip(axes, titles)):
        names = STROKE_NAMES if i == 4 else (STROKE_NAMES[i],)
        draw_shapes(ax, a_all if i == 4 else [a_shapes[names[0]]], pal, bbox)
        for k in range(-2, 3):
            ax.axhline(k * p.height / 4, color="#9AA0A6", lw=0.6, zorder=0.5)
        if i < 4:
            ax.plot(*s[names[0]].T, color="white", lw=1.0, ls="--", zorder=2)
        else:
            ax.axhline(0, color=pal.accent, lw=1.4, zorder=2)
            ax.plot([p.x0], [0], "o", ms=5, color=pal.accent, mec="white", zorder=3)
        ax.set_title(title, fontsize=10)
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
