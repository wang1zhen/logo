import math
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
    clef.Params(gap=0.0, width=clef.Params(gap=0.0, width=4).min_width()),
    clef.Params(weight=0.5, gap=0.2, width=3.6),  # weight = height/8 上限
    clef.Params(width=4.0),  # width = height 上限
    clef.Params(height=1.0, width=0.75, weight=0.07, gap=0.06),
]
VARIANT_FNS = [fn for _, fn in clef.VARIANTS.values()]


def area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def path_of(poly):
    return MplPath(np.vstack([poly, poly[:1]]), closed=True)


def has_mirror(poly, shapes, tol=1e-9):
    m = poly * [1.0, -1.0]
    for s in shapes:
        if len(s.poly) == len(m):
            d = np.linalg.norm(m[:, None, :] - s.poly[None], axis=2).min(axis=1)
            if d.max() < tol:
                return True
    return False


# ---------------------------------------------------------------- 有限坐标与镜像

@pytest.mark.parametrize("p", EDGE_PARAMS)
@pytest.mark.parametrize("fn", VARIANT_FNS)
def test_shapes_finite_non_degenerate(fn, p):
    shapes = fn(p)
    assert shapes
    for s in shapes:
        assert s.role in ("fg", "accent")
        assert s.poly.ndim == 2 and s.poly.shape[1] == 2 and len(s.poly) >= 3
        assert np.isfinite(s.poly).all()
        assert area(s.poly) > 0


@pytest.mark.parametrize("p", EDGE_PARAMS)
@pytest.mark.parametrize("fn", VARIANT_FNS)
def test_every_shape_has_mirror_about_third_line(fn, p):
    shapes = fn(p)
    for s in shapes:
        assert has_mirror(s.poly, shapes), "存在没有关于 y=0 镜像对应的形状"
    x0, y0, x1, y1 = clef.shapes_bbox(shapes)
    assert y0 == pytest.approx(-y1, abs=1e-12)


# ---------------------------------------------------------------- C 定位

@pytest.mark.parametrize("p", EDGE_PARAMS)
def test_lobes_converge_on_third_line(p):
    s = clef.base_strokes(p)
    clef.validate_strokes(s)
    up, lo = s["lobe_upper"], s["lobe_lower"]
    assert tuple(up[0]) == tuple(lo[0]) == (p.x0, 0.0)
    assert (up[1:, 1] > 0).all() and (lo[1:, 1] < 0).all()
    assert up[:, 0].min() == pytest.approx(p.x0)  # 喙是弧瓣最左点，指向竖线
    for bar in ("bar_thick", "bar_thin"):
        assert s[bar][:, 1].sum() == pytest.approx(0, abs=1e-12)
        assert s[bar][0, 0] < p.x0


@pytest.mark.parametrize("p", EDGE_PARAMS)
def test_each_variant_marks_c_at_beak(p):
    beak = (p.x0, 0.0)
    a = clef.variant_a(p)
    assert path_of(a[2].poly).contains_point(beak) and path_of(a[3].poly).contains_point(beak)
    b_lobes = [s.poly for s in clef.variant_b(p) if s.role == "accent"]
    assert len(b_lobes) == 2
    for poly in b_lobes:  # 两片月牙共享喙点，且各自只在喙点触及 y=0
        assert np.any(np.all(np.isclose(poly, beak), axis=1))
        ys = poly[:, 1][~np.all(np.isclose(poly, beak), axis=1)]
        assert (ys > 0).all() or (ys < 0).all()
    c_lines = [s.poly for s in clef.variant_c(p) if s.role == "accent" and np.ptp(s.poly[:, 1]) < p.weight]
    assert len(c_lines) == 1
    line = c_lines[0]
    assert line[:, 1].mean() == pytest.approx(0, abs=1e-12)
    assert line[:, 0].min() == pytest.approx(p.x0)


@pytest.mark.parametrize("p", EDGE_PARAMS)
def test_variant_c_staff_lines_on_staff_and_clear_of_strokes(p):
    strokes = clef.variant_a(p)  # 笔画 + 末端圆点
    lines = [s.poly for s in clef.variant_c(p) if s.role == "fg" and area(s.poly) and len(s.poly) == 4]
    lines = [poly for poly in lines if np.ptp(poly[:, 1]) == pytest.approx(0.4 * p.weight)]
    assert lines
    ys = {round(float(poly[:, 1].mean()), 9) for poly in lines}
    assert ys == {k * p.height / 4 for k in (-2, -1, 1, 2)}
    for poly in lines:
        y = poly[:, 1].mean()
        xs = np.linspace(poly[:, 0].min(), poly[:, 0].max(), 200)
        pts = np.column_stack([xs, np.full_like(xs, y)])
        for s in strokes:
            assert not path_of(s.poly).contains_points(pts).any(), "谱线没有避让笔画"


# ---------------------------------------------------------------- 破坏结构的反例

def _broken(kind):
    s = {k: v.copy() for k, v in clef.base_strokes(DEFAULT).items()}
    if kind == "missing_bar":
        del s["bar_thin"]
    elif kind == "nan":
        s["lobe_upper"][5, 0] = np.nan
    elif kind == "not_mirrored":
        s["lobe_lower"][:, 1] -= 0.1
    elif kind == "junction_off_line":
        s["lobe_upper"][:, 1] += 0.25
        s["lobe_lower"] = s["lobe_upper"] * [1, -1]
    elif kind == "bars_swapped":
        s["bar_thick"], s["bar_thin"] = s["bar_thin"], s["bar_thick"]
    elif kind == "bar_off_center":
        s["bar_thick"][:, 1] += 0.3
    elif kind == "bar_slanted":
        s["bar_thin"][1, 0] += 0.2
    elif kind == "lobe_crosses_line":
        s["lobe_upper"][40, 1] = -0.2
        s["lobe_lower"] = s["lobe_upper"] * [1, -1]
    elif kind == "beak_not_leftmost":
        s["lobe_upper"][0, 0] += 0.3  # 仍在 y=0，但不再是指向竖线的最左点
        s["lobe_lower"] = s["lobe_upper"] * [1, -1]
    return s


@pytest.mark.parametrize(
    "kind",
    ["missing_bar", "nan", "not_mirrored", "junction_off_line", "bars_swapped",
     "bar_off_center", "bar_slanted", "lobe_crosses_line", "beak_not_leftmost"],
)
def test_validate_rejects_broken_structure(kind):
    with pytest.raises(ValueError):
        clef.validate_strokes(_broken(kind))


# ---------------------------------------------------------------- 参数边界与参数真实生效

@pytest.mark.parametrize(
    "kwargs",
    [
        dict(height=0), dict(height=-1), dict(weight=0), dict(weight=0.51),
        dict(opening=-0.01), dict(opening=clef.OPENING_MAX + 0.01), dict(gap=-0.01), dict(gap=1.01),
        dict(width=1.0), dict(width=4.01), dict(width=float("nan")), dict(weight=float("inf")),
        dict(opening=True), dict(gap="0.3"),
    ],
)
def test_params_reject_out_of_range(kwargs):
    with pytest.raises(ValueError):
        clef.Params(**kwargs)


def test_params_accept_exact_bounds():
    clef.Params(opening=0)
    clef.Params(opening=clef.OPENING_MAX)
    clef.Params(gap=0, width=2.0)
    clef.Params(gap=1.0, width=4.0)
    p = clef.Params(width=clef.Params().min_width())
    assert p.width == pytest.approx(p.x0 + 3 * p.weight)


def test_width_and_height_set_extent():
    for p in EDGE_PARAMS:
        a = clef.variant_a(p)
        strokes_box = clef.shapes_bbox(a[:4])
        assert strokes_box[2] == pytest.approx(p.width, abs=1e-3)
        bars_box = clef.shapes_bbox(clef.variant_b(p)[:2])
        assert (bars_box[1], bars_box[3]) == pytest.approx((-p.height / 2, p.height / 2))
        assert bars_box[0] == pytest.approx(0)


def test_weight_changes_bar_widths_and_ink_area():
    thin, heavy = clef.Params(weight=0.2, width=3.2), clef.Params(weight=0.4, width=3.2)
    for p in (thin, heavy):
        bars = clef.variant_b(p)[:2]
        assert np.ptp(bars[0].poly[:, 0]) == pytest.approx(2 * p.weight)
        assert np.ptp(bars[1].poly[:, 0]) == pytest.approx(p.weight)
    for fn in VARIANT_FNS:
        assert sum(area(s.poly) for s in fn(heavy)) > sum(area(s.poly) for s in fn(thin))


def test_opening_moves_lobe_end_away_from_bars():
    ends = [clef.base_strokes(clef.Params(opening=o))["lobe_upper"][-1] for o in (0, 0.4, clef.OPENING_MAX)]
    assert ends[0][0] < ends[1][0] < ends[2][0]  # 开口越大，末端离竖线越远
    assert all(e[1] > 0 for e in ends)
    a_open = clef.variant_a(clef.Params(opening=clef.OPENING_MAX))
    a_closed = clef.variant_a(clef.Params(opening=0))
    assert area(a_open[2].poly) < area(a_closed[2].poly)  # 弧段变短


def test_gap_moves_thin_bar_and_beak():
    near, far = clef.Params(gap=0.1), clef.Params(gap=0.5)
    for p in (near, far):
        s = clef.base_strokes(p)
        thick_right = s["bar_thick"][0, 0] + p.weight
        thin_left = s["bar_thin"][0, 0] - p.weight / 2
        assert thin_left - thick_right == pytest.approx(p.gap)
        assert p.x0 - (s["bar_thin"][0, 0] + p.weight / 2) == pytest.approx(p.gap)
    assert far.x0 > near.x0


# ---------------------------------------------------------------- 导出

def test_export_all_writes_three_variants_and_figures(tmp_path):
    written = clef.export_all(tmp_path)
    expected = {f"variant_{k}.{e}" for k in "abc" for e in ("png", "svg")} | {"compare.png", "strokes_breakdown.png"}
    assert set(written) == expected
    pal = clef.PALETTES["ink"]
    for key, (_, fn) in clef.VARIANTS.items():
        img = mpimg.imread(written[f"variant_{key}.png"])[..., :3]
        assert img.shape[:2] == (512, 512)
        for color in (pal.bg, pal.fg, pal.accent):  # 三种颜色都确实出现在图里
            assert (np.abs(img - to_rgb(color)).max(axis=2) < 0.02).sum() > 50
        root = ET.parse(written[f"variant_{key}.svg"]).getroot()
        paths = root.findall(".//{http://www.w3.org/2000/svg}path")
        assert len(paths) >= len(fn(DEFAULT))
        assert root.get("viewBox")
    for name in ("compare.png", "strokes_breakdown.png"):
        img = mpimg.imread(written[name])
        assert img.ndim == 3 and min(img.shape[:2]) > 300


@pytest.mark.parametrize("px", [32, 64])
@pytest.mark.parametrize("fn", VARIANT_FNS)
def test_small_raster_keeps_ink(fn, px):
    img = clef.rasterize(fn(DEFAULT), clef.MONO, px)
    assert img.shape == (px, px, 4)
    ink = (img[..., :3].astype(int).sum(axis=2) < 3 * 128).mean()
    assert 0.05 < ink < 0.6
    # 墨迹上下对称（允许 1px 的栅格误差）
    rows = (img[..., :3].astype(int).sum(axis=2) < 3 * 128).sum(axis=1)
    assert np.abs(rows - rows[::-1]).max() <= math.ceil(px / 16)
