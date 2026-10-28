# 中音谱号个人 Logo 探索

以中音谱号（C 谱号）为母题，用 matplotlib 手绘两种轮廓：左粗右细两根竖柱；右侧中腰实心，向左的尖凹顶点落在第三线 y = 0；上下尖叶接粗腹细回转的卷叶，卷端收成带钩的圆团。

- **A 精致传统**：上下对称，圆团回钩，飞白与墨晕都很克制
- **B 自由水墨**：非对称（上卷大而重、下卷收窄成斜泪滴），粗柱斜切起笔、细柱收成悬针，主体略右倾，按 `seed` 轻微手抖；这是矢量轮廓对水墨笔意的近似，不是毛笔笔触模拟

两种方案的轮廓都是独立绘制的闭合三次 Bézier 复合路径：外轮廓逆时针、飞白子路径顺时针，按 nonzero 规则填充得到留白，不依赖 even-odd。
配色：墨蓝·朱红·米白（ink）/ 深青·琥珀·暖白（teal）/ 炭黑·钴蓝·白（cobalt）/ 墨黑·纸白·朱红（sumi），另附单色。

## 使用

需要 [uv](https://docs.astral.sh/uv/)（会自动准备 Python 3.12）。

```sh
uv sync --locked          # 安装依赖
uv run pytest -q          # 几何与导出测试
uv run python clef.py     # 用默认参数、ink 配色导出到 out/
uv run jupyter lab logo.ipynb   # 交互探索；参数集中在“参数”一格
```

不打开界面、直接执行整本 notebook：

```sh
uv run jupyter nbconvert --to notebook --execute logo.ipynb \
  --output-dir out --output logo.executed.ipynb --ExecutePreprocessor.timeout=300
```

## 参数

| 参数 | 默认 | 范围 | 含义 |
|---|---|---|---|
| `height` | 4.0 | > 0 | 竖柱总高 = 五线谱上下线距离 |
| `width` | 2.76 | [`min_width()`, 0.9·height] | 粗柱左缘到外腹最右点，默认约 0.69·height |
| `weight` | 0.25 | [0.0375·height, height/8] | 基础粗细：粗柱 = 2·weight，外腹约 2.6·weight |
| `opening` | 0.25 | [0, 0.8] | 卷端开口：越大圆团越往上、尖叶越低 |
| `gap` | 0.25 | [0.025·height, height/4] | 粗细柱间距；腰部与细柱之间再留 0.3·gap |
| `contrast` | 0.6 | [0, 1] | 粗细对比：越大细柱、回转发丝越细，外腹越厚 |
| `seed` | 7 | 整数 [0, 2³²−1] | 飞白、墨晕与 B 手抖的种子；只用局部 RNG，不改全局随机状态 |
| `dryness` | 0.35 | [0, 1] | 飞白量，0 = 无 |
| `bleed` | 0.3 | [0, 1] | 墨晕量，0 = 无 |

`weight`、`gap`、`width` 的范围都相对 `height`，是按实际路径测出来的支持范围，不是任意可调：
`weight` 低于约 0.03·height 时细回转会被外缘穿过、卷叶自交；`gap` 接近 0 时两柱贴合、B 的细柱尖会压进腰部；
`width` 到 height 时圆团会撞上尖叶。超出范围直接报 `ValueError`。`min_width()` 随 `weight`、`gap`、`contrast` 变化，
所以 `weight` 接近 height/8 时还需要较小的 `gap` 才能放进 0.9·height 的宽度。
`validate_glyph` 用网格采样检查环绕数，细小的自交可能漏检；测试另用折线线段求交独立检查范围端点组合。

## 从旧版迁移

- 删除了方案 C（staff-fused 谱线融合）。`get_variant("c")` 会直接报错，不会映射到其他方案；`out/` 里不再生成 `variant_c.*`。
- 旧的“骨架折线 + 月牙”几何已全部重画。`base_strokes` / `validate_strokes` 换成 `Glyph` / `validate_glyph`；方案函数现在返回 `Glyph`，用 `render` / `draw_glyph` 绘制。
- `width` 默认值从 3.0 改为 2.76（宽高比约 0.69）；`weight` 默认值从 0.3 改为 0.25；细柱宽度不再等于 `weight`，而是 `weight·(1 − 0.6·contrast)`。
- `opening` 原来控制弧瓣末端回收多少，现在控制圆团与尖叶之间的开口；`gap` 除了柱间距，还决定腰部离细柱多远（不再要求腰部贴住细柱）。
- 新增 `contrast`、`seed`、`dryness`、`bleed`。
- `weight`、`gap`、`width` 的范围收紧成相对 `height` 的实测支持范围：`gap=0`、`width=height` 以及过细的 `weight`（如 0.01）现在会报错。

## 参考、来源与边界

- 形态参考 [SMuFL cClef（U+E05C）](https://smufl.formats.music/latest/tables/clefs.html) 及其参考字体 Bravura：中音谱号的中心落在第三线。本项目只照着看、独立手绘，没有复制任何字体的控制点或轮廓，也不打包或配置字体。Bravura 采用 SIL Open Font License；要用 Bravura 本身的字形，需另行遵守其许可。
- 墨晕只是近似：几层低透明度的外扩轮廓，再用矢量裁切限定在局部，并非真实的渗透模拟。飞白是顺笔向的反向子路径。
- SVG 全为矢量路径和裁切，没有嵌入栅格图。

## 文件

- `clef.py`：参数校验、Bézier 工具、两种方案、飞白/墨晕、渲染与导出
- `logo.ipynb`：官方参考对照、黑剪影、拆解、同尺度对比、四配色 + 单色、参数扫描（共享取景）、32/64 px、大图、导出
- `tests/test_clef.py`：闭合轮廓与方向、第三线定位、留白、结构反例、参数边界与效应、seed 复现与 RNG 隔离、SVG 无栅格、导出
- `out/`：`variant_{a,b}.{png,svg}`、`compare.png`、`strokes_breakdown.png`（已被 git 忽略）

测试只检查几何与导出是否正确，好不好看要靠人眼判断。
