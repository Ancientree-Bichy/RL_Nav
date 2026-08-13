# Truncated Policy Iteration：10×10 Grid World

这是一个不依赖 Gym 的小型、可读的有限 MDP 示例，用同一个 Grid World 对比：

- Value Iteration（价值迭代）
- Policy Iteration（策略迭代）
- Truncated Policy Iteration（截断/修正版策略迭代）
- First-Visit Monte Carlo Control（基于采样，不做 Bellman backup）

## 环境

本机已经创建 Apple Silicon 原生 Conda 环境 `rl_test`，其中包含：

- Python 3.11
- NumPy
- Matplotlib
- Pillow（保存 GIF）
- pytest

进入项目并运行：

```bash
conda activate rl_test
cd /Users/ancientree/Projects/RL_Toys/Truncated_Policy_Iteration
python run_demo.py
```

如果要在另一台机器重建相同的精简环境：

```bash
conda env create -f environment.yml
```

## Grid World 设定

- 网格为 10×10。
- 起点 `S=(9, 0)`，终点 `G=(0, 9)`。
- 三道带缺口的障碍墙迫使路径形成明显的折线。
- 动作是上、右、下、左，转移是确定性的。
- 普通移动奖励为 `-1`，进入目标奖励为 `0`；因此最大化回报等价于找最短路。
- 折扣因子默认 `γ=0.95`。
- 撞到边界或障碍时留在原格，并仍然获得 `-1`。

## 三种算法的关系

`algorithms.py` 中三种方法共享同一个 Bellman 备份与结果结构。

Value Iteration 每轮直接做最优 Bellman 备份：

```text
V(s) ← max_a [r(s,a) + γ V(s')]
```

Policy Iteration 先把当前策略评估到收敛，再做一次贪心策略改进。

Truncated Policy Iteration 每次只做 `m` 次策略评估扫描就立即改进策略。默认
`m=3`，因此它位于价值迭代和完整策略迭代之间。停止条件同时检查策略稳定和
Bellman 最优残差，避免因为价值函数尚未评估充分而过早退出。

可调整截断长度：

```bash
python run_demo.py --truncated-sweeps 1
python run_demo.py --truncated-sweeps 10
```

## 输出

默认运行会在 `outputs/` 生成：

- `algorithm_comparison.png`：三个算法的最终 value、policy 箭头和路径。
- `convergence.png`：以累计 state backup 为横轴的收敛曲线。
- `value_evolution.gif`：value 传播和当前贪心路径的逐步动画。
- `results.json`：路径、迭代次数和计算量等机器可读结果。

若只想快速生成静态图：

```bash
python run_demo.py --no-animation
```

运行测试：

```bash
pytest -q
```

## Monte Carlo Basic demo

Monte Carlo demo 不读取完整转移模型，也不调用 `action_values()` 做 Bellman
backup。它只重复执行以下流程：

1. 用 epsilon-greedy policy 与环境交互，采样完整 episode。
2. 从 episode 末尾反向计算折扣回报 `G_t`。
3. 对每个首次出现的 `(state, action)` 用 sample mean 更新 `Q(s,a)`。
4. 下一轮继续使用相对于当前 `Q` 的 epsilon-greedy policy。

运行：

```bash
python run_monte_carlo_demo.py
```

默认训练 30,000 个 episode。为了让整张地图都获得采样，50% episode 从指定
起点开始，其余 episode 使用随机 exploring start。epsilon 从 1.0 线性下降到
0.02。输出包括：

- `outputs/monte_carlo_summary.png`
- `outputs/monte_carlo_results.json`

可以调整训练规模和随机种子：

```bash
python run_monte_carlo_demo.py --episodes 50000 --seed 42
```

## 文件结构

```text
grid_world.py       Grid World/MDP、转移、策略和路径提取
algorithms.py       三种动态规划算法与共享 Bellman 操作
visualization.py    value heatmap、policy、路径、收敛图和动画
run_demo.py         命令行入口
monte_carlo.py      First-visit MC control、episode sampling 和 Q 更新
run_monte_carlo_demo.py  Monte Carlo 命令行入口
tests/              环境与算法正确性测试
environment.yml     Conda 环境定义
```
