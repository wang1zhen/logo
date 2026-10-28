import random
import xml.etree.ElementTree as ET

import matplotlib.image as mpimg
import numpy as np
import pytest
from matplotlib.colors import to_rgb
from matplotlib.path import Path as MplPath

import clef

DEFAULT = clef.Params()
EDGE_PARAMS = [
    DEFAULT,
    clef.Params(opening=0.0),
    clef.Params(opening=clef.OPENING_MAX),
    clef.Params(gap=0.1, width=clef.Params(gap=0.1).min_width()),  # gap = 0.025·height 下限
    clef.Params(gap=1.0, width=3.6),  # gap = height/4 上限，width = 0.9·height 上限
    clef.Params(weight=0.5, gap=0.1, width=3.6),  # weight = height/8 上限
    clef.Params(weight=0.15),  # weight = 0.0375·height 下限
    clef.Params(width=3.6),  # width = 0.9·height 上限
    clef.Params(contrast=0.0),
    clef.Params(contrast=1.0, dryness=1.0, bleed=1.0),
    clef.Params(dryness=0.0, bleed=0.0),
    clef.Params(height=1.0, width=0.69, weight=0.06, gap=0.06),
    clef.Params(seed=clef.SEED_MAX),
]
VARIANT_FNS = [fn for _, fn in clef.VARIANTS.values()]
SVG = "{http://www.w3.org/2000/svg}"


def ink(points, g):
    return clef.winding(points, [*g.outers, *g.holes])


def hausdorff(a, b):
    d = np.linalg.norm(a[:, None] - b[None], axis=2)
    return max(d.min(0).max(), d.min(1).max())


# ---------------------------------------------------------------- 轮廓：有限、闭合、三次 Bézier、有面积、方向正确

@pytest.mark.parametrize("p", EDGE_PARAMS)
@pytest.mark.parametrize("fn", VARIANT_FNS)
def test_glyph_is_valid_closed_cubic_compound(fn, p):
    g = fn(p)
    clef.validate_glyph(g)
    for c in (*g.outers, *g.holes):
        assert np.isfinite(c).all() and (len(c) - 1) % 3 == 0
        assert np.allclose(c[0], c[-1])
    assert all(clef.signed_area(c) > 0.01 * p.height**2 * 0.01 for c in g.outers)
    assert all(clef.signed_area(h) < 0 for h in g.holes)
    for s in clef.glyph_shapes(g):  # 渲染路径只含 MOVETO / CURVE4 / CLOSEPOLY
        assert set(s.path.codes) <= {MplPath.MOVETO, MplPath.CURVE4, MplPath.CLOSEPOLY}


@pytest.mark.parametrize("fn", VARIANT_FNS)
def test_holes_cut_by_orientation_not_even_odd(fn):
    """飞白是反向子路径：nonzero 下为留白，且同向化后会被填回，说明留白靠方向而非 even-odd。"""
    g = fn(DEFAULT)
    assert g.holes
    probes = np.vstack([clef.flatten(h, 4).mean(0, keepdims=True) for h in g.holes])
    probes = probes[clef.winding(probes, g.holes) == -1]  # 取确实落在飞白内部的点
    assert len(probes)
    assert (ink(probes, g) == 0).all()
    same_dir = [h[::-1] for h in g.holes]
    assert (clef.winding(probes, [*g.outers, *same_dir]) == 2).all()


# ---------------------------------------------------------------- 谱号结构：第三线、尖凹、留白、粗细

@pytest.mark.parametrize("p", EDGE_PARAMS)
@pytest.mark.parametrize("fn", VARIANT_FNS)
def test_c_marked_on_third_line(fn, p):
    g = fn(p)
    nx, ny = g.notch
    assert ny == 0
    d = 0.02 * p.half
    assert clef.winding([(nx - d, 0)], [g.body])[0] == 1  # 尖凹左侧是实心中腰
    assert clef.winding([(nx + d, 0)], [g.body])[0] == 0  # 右侧是向左切入的留白
    for sign in (1, -1):  # 尖凹上下方 0.4H 内各有卷叶回转，C 正夹在第三线
        col = np.column_stack([np.full(80, nx + 4 * d), sign * np.linspace(d, 0.4 * p.half, 80)])
        assert (clef.winding(col, [g.body]) == 1).any()
    _, y0, _, y1 = clef.glyph_bbox(g)
    assert abs(y0 + y1) < 0.05 * p.height


@pytest.mark.parametrize("fn", VARIANT_FNS)
def test_lobes_have_counters_and_terminals(fn):
    g = fn(DEFAULT)
    for (cx, cy), (bx, by) in zip(g.counters, g.terminals):
        assert ink([(cx, cy)], g)[0] == 0
        assert ink([(bx, by)], g)[0] == 1  # 圆团中心是实心
        assert bx < cx  # 圆团在留白左侧，卷回到尖叶一侧
        assert abs(by) > abs(cy) * 0.9  # 上卷圆团偏上、下卷偏下


def test_thick_belly_thin_hairline_and_bars():
    p = DEFAULT
    g = clef.variant_a(p)
    x0, _, x1, _ = clef.bbox_of([g.bar_thick])
    t0, _, t1, _ = clef.bbox_of([g.bar_thin])
    assert x1 - x0 == pytest.approx(2 * p.weight) and t1 - t0 == pytest.approx(p.thin)
    cx, cy = g.counters[0]
    row = np.column_stack([np.linspace(cx, p.width + 0.1, 400), np.full(400, cy)])
    col = np.column_stack([np.full(400, cx), np.linspace(cy, p.half + 0.1, 400)])
    belly = (ink(row, g) == 1).sum() * (row[1, 0] - row[0, 0])
    hair = (ink(col, g) == 1).sum() * (col[1, 1] - col[0, 1])
    assert belly > 3 * hair  # 外腹厚、回转细


def test_default_proportions():
    for fn in VARIANT_FNS:
        x0, y0, x1, y1 = clef.glyph_bbox(fn(DEFAULT))
        assert (x1 - x0) / (y1 - y0) == pytest.approx(0.69, abs=0.03)


def test_a_symmetric_b_free():
    a, b = clef.variant_a(DEFAULT), clef.variant_b(DEFAULT)
    pa, pb = clef.flatten(a.body, 12), clef.flatten(b.body, 12)
    assert hausdorff(pa, pa * [1, -1]) < 1e-9
    assert hausdorff(pb, pb * [1, -1]) > 0.05 * DEFAULT.half
    # B 不是 A 加纹理：主轮廓本身不同
    assert hausdorff(pa, pb) > 0.1 * DEFAULT.half


# ---------------------------------------------------------------- 破坏结构的反例

def _broken(kind):
    full = clef.variant_a(DEFAULT)
    g = clef.silhouette(full)
    H = DEFAULT.half
    if kind == "nan":
        body = g.body.copy()
        body[5, 0] = np.nan
        return g._replace(body=body)
    if kind == "open_contour":
        return g._replace(body=g.body[:-3])
    if kind == "body_clockwise":
        return g._replace(body=g.body[::-1])
    if kind == "hole_counterclockwise":
        return full._replace(holes=(full.holes[0][::-1], *full.holes[1:]))
    if kind == "hole_outside_ink":
        return g._replace(holes=(clef.rounded_rect(-2, -0.1, -1.5, 0.1, 0.02)[::-1],))
    if kind == "bars_swapped":
        return g._replace(bar_thick=g.bar_thin, bar_thin=g.bar_thick)
    if kind == "bar_off_center":
        return g._replace(bar_thick=g.bar_thick + [0, 0.3 * H])
    if kind == "notch_off_line":
        return g._replace(notch=(g.notch[0], 0.2))
    if kind == "notch_not_concave":  # 实心中腰内部的点不是尖凹
        return g._replace(notch=(g.notch[0] - 0.3, 0.0))
    if kind == "counter_filled":  # 卷叶画成实心团块
        return g._replace(body=clef.rounded_rect(DEFAULT.x_waist, -H, DEFAULT.width, H, 0.5))
    if kind == "upper_only":
        return g._replace(body=g.body * [1, 0.5] + [0, 0.5 * H])
    if kind == "body_overlaps_bar":
        return g._replace(body=g.body - [0.2, 0])
    raise AssertionError(kind)


@pytest.mark.parametrize("kind, msg", [
    ("nan", "非有限"), ("open_contour", "闭合"), ("body_clockwise", "逆时针"),
    ("hole_counterclockwise", "顺时针"), ("hole_outside_ink", "墨迹之内"), ("bars_swapped", "粗柱在细柱左侧"),
    ("bar_off_center", "第三线"), ("notch_off_line", "y=0"), ("notch_not_concave", "向左尖凹"),
    ("counter_filled", "留白"), ("upper_only", "上下两片"), ("body_overlaps_bar", "细柱右侧"),
])
def test_validate_rejects_broken_structure(kind, msg):
    with pytest.raises(ValueError, match=msg):
        clef.validate_glyph(_broken(kind))


# ---------------------------------------------------------------- 参数边界与参数效应

@pytest.mark.parametrize("kwargs", [
    dict(height=0), dict(height=-1), dict(weight=0), dict(weight=0.51),
    dict(opening=-0.01), dict(opening=clef.OPENING_MAX + 0.01), dict(gap=-0.01), dict(gap=1.01),
    dict(width=1.0), dict(width=3.61), dict(width=4.0), dict(width=float("nan")), dict(weight=float("inf")),
    dict(weight=0.01), dict(weight=0.149), dict(gap=0), dict(gap=0.099),
    dict(opening=True), dict(gap="0.3"),
    dict(contrast=-0.1), dict(contrast=1.1), dict(dryness=1.5), dict(bleed=-0.01), dict(bleed=float("nan")),
    dict(seed=-1), dict(seed=clef.SEED_MAX + 1), dict(seed=1.5), dict(seed=True), dict(seed="7"),
])
def test_params_reject_out_of_range(kwargs):
    with pytest.raises(ValueError):
        clef.Params(**kwargs)


def test_params_accept_exact_bounds():
    for kw in (dict(opening=0), dict(opening=clef.OPENING_MAX), dict(gap=0.1), dict(gap=1.0, width=3.6),
               dict(weight=0.15), dict(weight=0.5, gap=0.1, width=3.6), dict(width=3.6),
               dict(contrast=0), dict(contrast=1), dict(dryness=0), dict(dryness=1), dict(bleed=1),
               dict(seed=0), dict(seed=clef.SEED_MAX), dict(seed=np.int64(3))):
        clef.Params(**kw)
    p = clef.Params(width=clef.Params().min_width())
    clef.validate_glyph(clef.variant_a(p))


def test_weight_gap_width_bounds_scale_with_height():
    with pytest.raises(ValueError, match="weight"):
        clef.Params(weight=0.01, dryness=0, bleed=0)
    clef.Params(height=1.0, weight=0.0375, gap=0.025, width=0.69)
    clef.Params(height=10.0, weight=1.25, gap=0.25, width=9.0)
    for kw in (dict(weight=0.0374), dict(gap=0.0249), dict(width=0.91)):
        with pytest.raises(ValueError):
            clef.Params(**{**dict(height=1.0, weight=0.0625, gap=0.0625, width=0.69), **kw})


def crossings(g, n=16):
    """独立于 validate_glyph 的网格采样：把每条轮廓细分成折线，数真正相交（非相邻）的线段对。"""
    polys = [clef.flatten(c, n) for c in (*g.outers, *g.holes)]

    def cross(a, b, c):
        return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])

    total = 0
    for i, p in enumerate(polys):
        a0, a1 = p, np.roll(p, -1, 0)
        amin, amax = np.minimum(a0, a1), np.maximum(a0, a1)
        pmin, pmax = amin.min(0), amax.max(0)
        for k, q in enumerate(polys[i:], i):
            if k > i and ((pmax < q.min(0)).any() or (q.max(0) < pmin).any()):
                continue
            b0, b1 = q, np.roll(q, -1, 0)
            bmin, bmax = np.minimum(b0, b1), np.maximum(b0, b1)
            cand = ((amax[:, None, 0] >= bmin[None, :, 0]) & (bmax[None, :, 0] >= amin[:, None, 0])
                    & (amax[:, None, 1] >= bmin[None, :, 1]) & (bmax[None, :, 1] >= amin[:, None, 1]))
            if k == i:  # 同一条轮廓：去掉相邻线段，每对只数一次
                m = len(p)
                r, c = np.ogrid[:m, :m]
                cand &= (c - r > 1) & (c - r != m - 1)
            r, c = np.nonzero(cand)
            if len(r):
                pa0, pa1, qb0, qb1 = a0[r], a1[r], b0[c], b1[c]
                hit = (cross(pa0, pa1, qb0) * cross(pa0, pa1, qb1) < 0) & (cross(qb0, qb1, pa0) * cross(qb0, qb1, pa1) < 0)
                total += int(hit.sum())
    return total


def test_crossings_detects_self_intersection():
    g = clef.silhouette(clef.variant_a(DEFAULT))
    assert crossings(g) == 0
    assert crossings(g._replace(body=g.body - [0.2, 0])) > 0  # 主体压进细柱
    bow = clef.closed([clef.K(0, 0, 45), clef.K(1, 1, 270), clef.K(1, 0, 135), clef.K(0, 1, 270)])
    assert crossings(g._replace(body=bow)) > 0  # 8 字自交


@pytest.mark.parametrize("weight, gap", [(0.15, 0.1), (0.15, 1.0), (0.25, 0.25), (0.5, 0.1)])
@pytest.mark.parametrize("fn", VARIANT_FNS)
def test_no_crossings_at_supported_extremes(fn, weight, gap):
    """粗细/间距端点 × width 两端 × contrast/opening 两端 × 多个 seed，轮廓不自交、部件不互相穿插。"""
    checked = 0
    for contrast in (0, 1):
        try:
            lo = clef.Params(weight=weight, gap=gap, contrast=contrast, width=3.6).min_width()
        except ValueError as e:  # weight=height/8 且 contrast=0 时细柱最粗，0.9·height 放不下
            assert "width 太小" in str(e)
            continue
        checked += 1
        for width in (lo, 3.6):
            for opening in (0, clef.OPENING_MAX):
                for seed in (0, 6, 7):
                    p = clef.Params(weight=weight, gap=gap, width=width, contrast=contrast, opening=opening,
                                    seed=seed, dryness=1.0)
                    g = fn(p)
                    assert crossings(g) == 0, p
                    clef.validate_glyph(g)
    assert checked


def test_b_hairline_survives_jitter():
    """B 的下回转是发丝：手抖不能把外缘抖到内缘之上（contrast=1 时曾在约 8% 的 seed 上自交）。"""
    for seed in range(40):
        assert crossings(clef.silhouette(clef.variant_b(clef.Params(contrast=1, seed=seed)))) == 0, seed


def test_old_variant_c_rejected():
    assert set(clef.VARIANTS) == {"a", "b"}
    with pytest.raises(ValueError, match="已删除"):
        clef.get_variant("c")
    with pytest.raises(ValueError):
        clef.get_variant("z")
    assert not hasattr(clef, "variant_c")


def test_width_and_height_set_extent():
    for p in EDGE_PARAMS:
        a = clef.variant_a(p)
        x0, y0, x1, y1 = clef.glyph_bbox(a)
        assert (x0, y0, x1, y1) == pytest.approx((0, -p.half, p.width, p.half), abs=2e-3 * p.height)
        x0, y0, x1, y1 = clef.glyph_bbox(clef.variant_b(p))
        assert x1 == pytest.approx(p.width, abs=0.03 * p.height)
        assert -1.05 * p.half < y0 < -0.95 * p.half and 0.95 * p.half < y1 < 1.05 * p.half


def _ink_area(g, n=220):
    x0, y0, x1, y1 = clef.glyph_bbox(g)
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    return (ink(grid, g) > 0).mean() * (x1 - x0) * (y1 - y0)


def test_weight_increases_ink():
    light, heavy = clef.Params(weight=0.18, dryness=0), clef.Params(weight=0.35, dryness=0)
    for fn in VARIANT_FNS:
        assert _ink_area(fn(heavy)) > 1.15 * _ink_area(fn(light))


def test_gap_moves_thin_bar_and_waist():
    near, far = clef.Params(gap=0.1), clef.Params(gap=0.5)
    for fn in VARIANT_FNS:
        gn, gf = fn(near), fn(far)
        assert clef.bbox_of([gf.bar_thin])[0] - clef.bbox_of([gn.bar_thin])[0] == pytest.approx(0.4, abs=0.02)
        assert clef.bbox_of([gf.body])[0] > clef.bbox_of([gn.body])[0] + 0.4
    g = clef.variant_a(near)
    assert clef.bbox_of([g.bar_thin])[0] - clef.bbox_of([g.bar_thick])[2] == pytest.approx(0.1)


def test_opening_widens_mouth_between_terminal_and_leaf():
    def mouth(o, fn):
        g = fn(clef.Params(opening=o))
        bx, by = g.terminals[0]
        return by - g.notch[1] - (clef.bbox_of([g.body])[3] - by)  # 圆团离尖凹越远、离顶越近
    for fn in VARIANT_FNS:
        assert mouth(0, fn) < mouth(0.4, fn) < mouth(clef.OPENING_MAX, fn)


def test_contrast_thins_hairlines_and_thin_bar():
    lo, hi = clef.Params(contrast=0.1), clef.Params(contrast=0.9)
    assert hi.thin < lo.thin and hi.hair < lo.hair
    for fn in VARIANT_FNS:
        w = lambda p: np.ptp(clef.flatten(fn(p).bar_thin)[:, 0])  # noqa: E731
        assert w(hi) < w(lo)


def test_dryness_and_bleed_effects():
    for fn in VARIANT_FNS:
        assert fn(clef.Params(dryness=0)).holes == ()
        area = lambda d: -sum(clef.signed_area(h) for h in fn(clef.Params(dryness=d)).holes)  # noqa: E731
        assert area(0.2) < area(0.6) < area(1.0)
        assert fn(clef.Params(bleed=0)).bleed == ()
        b_lo, b_hi = fn(clef.Params(bleed=0.2)).bleed, fn(clef.Params(bleed=0.9)).bleed
        assert b_lo and sum(b.alpha for b in b_hi) > sum(b.alpha for b in b_lo)
        assert all(0 < b.alpha <= 0.2 for b in b_hi)  # 墨晕只是少量低 alpha 轮廓
        for b in b_hi:  # 有界：墨晕裁切区在字形附近
            bx = clef.bbox_of([b.clip])
            gx = clef.glyph_bbox(fn(DEFAULT))
            assert bx[0] > gx[0] - 0.2 and bx[2] < gx[2] + 0.2 and bx[1] > gx[1] - 0.3 and bx[3] < gx[3] + 0.3


def test_seed_reproducible_and_local():
    random.seed(1)
    np.random.seed(1)
    py_state, np_state = random.getstate(), np.random.get_state()
    b1, b2, b3 = clef.variant_b(DEFAULT), clef.variant_b(DEFAULT), clef.variant_b(clef.Params(seed=8))
    clef.variant_a(DEFAULT)
    assert random.getstate() == py_state
    after = np.random.get_state()
    assert after[0] == np_state[0] and (after[1] == np_state[1]).all() and after[2:] == np_state[2:]
    assert np.array_equal(b1.body, b2.body) and all(np.array_equal(x, y) for x, y in zip(b1.holes, b2.holes))
    assert not np.array_equal(b1.body, b3.body)
    clef.validate_glyph(b3)


# ---------------------------------------------------------------- 导出

def test_export_writes_two_variants_and_figures(tmp_path):
    written = clef.export_all(tmp_path)
    expected = {f"variant_{k}.{e}" for k in "ab" for e in ("png", "svg")} | {"compare.png", "strokes_breakdown.png"}
    assert set(written) == expected
    assert not list(tmp_path.glob("variant_c.*"))
    pal = clef.PALETTES["ink"]
    for key in "ab":
        img = mpimg.imread(written[f"variant_{key}.png"])[..., :3]
        assert img.shape[:2] == (512, 512)
        for color in (pal.bg, pal.fg, pal.accent):
            assert (np.abs(img - to_rgb(color)).max(axis=2) < 0.02).sum() > 50
        root = ET.parse(written[f"variant_{key}.svg"]).getroot()
        assert root.get("viewBox")
        assert not root.findall(f".//{SVG}image"), "SVG 不应嵌入栅格图"
        assert len(root.findall(f".//{SVG}path")) >= 3
        assert root.findall(f".//{SVG}clipPath"), "墨晕应以矢量裁切实现"
    for name in ("compare.png", "strokes_breakdown.png"):
        img = mpimg.imread(written[name])
        assert img.ndim == 3 and min(img.shape[:2]) > 300


@pytest.mark.parametrize("px", [32, 64])
@pytest.mark.parametrize("fn", VARIANT_FNS)
def test_small_raster_keeps_ink(fn, px):
    img = clef.rasterize(fn(DEFAULT), clef.MONO, px)
    assert img.shape == (px, px, 4)
    dark = img[..., :3].astype(int).sum(axis=2) < 3 * 128
    assert 0.1 < dark.mean() < 0.6
    rows = dark.sum(axis=1)
    mid = np.flatnonzero(rows)
    assert abs((mid[0] + mid[-1]) / 2 - (px - 1) / 2) <= 1  # 墨迹上下居中于第三线
