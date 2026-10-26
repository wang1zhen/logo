# 中音谱号个人 Logo 探索

用 matplotlib 探索以中音谱号（C 谱号）为母题的个人 Logo：左侧粗细两条竖线，右侧上下两瓣弧线在第三线 y = 0 收束。
三种方案共用一组基础笔画：

- **A 圆弧线描**：等宽圆头线，弧瓣末端带圆点
- **B 实心切割**：实心竖线，加上“实心椭圆减内椭圆”切出的月牙弧瓣
- **C 谱线融合**：五线谱避让笔画，第三线从收束点向右伸出

配色：墨蓝·朱红·米白 / 深青·琥珀·暖白 / 炭黑·钴蓝·白，另附单色。

## 使用

需要 [uv](https://docs.astral.sh/uv/)（会自动准备 Python 3.12）。

```sh
uv sync --locked          # 安装依赖
uv run pytest -q          # 几何测试
uv run python clef.py     # 用默认参数导出到 out/
uv run jupyter lab logo.ipynb   # 交互探索；参数集中在“集中参数”一格
```

不打开界面、直接执行整本 notebook：

```sh
uv run jupyter nbconvert --to notebook --execute logo.ipynb \
  --output-dir out --output logo.executed.ipynb --ExecutePreprocessor.timeout=120
```

## 文件

- `clef.py`：参数校验、基础笔画、三种方案、渲染与导出
- `logo.ipynb`：中文说明、逐笔拆解、配色对比、参数扫描、小尺寸预览、导出
- `tests/test_clef.py`：几何测试（坐标有限、上下镜像、C 定位、结构反例、参数边界、导出）
- `out/`：导出结果，含 `variant_{a,b,c}.{png,svg}`、`compare.png`、`strokes_breakdown.png`（已被 git 忽略）

测试只检查几何与导出是否正确，“像不像谱号、好不好看”仍需人眼判断。
