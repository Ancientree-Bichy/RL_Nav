# RL Navigation (Modular)

把 2D Grid Navigation 从单文件重构为可替换算法的模块化工程，便于训练与算法对比。

## 目录结构

```text
rl_nav.py                  # 统一训练/评估入口
rlnav/
  envs/                    # 导航平台（GridWorld）
  algorithms/              # RL算法实现（DQN, PG）
  infra/                   # 训练基础设施（logger, replay, checkpoint）
  training.py              # 通用训练循环
  evaluation.py            # 评估 rollout
  networks.py              # 通用网络模块
```

## 安装依赖

```bash
python3 -m pip install -r requirements.txt
```

## 训练

```bash
python3 rl_nav.py --algo dqn
python3 rl_nav.py --algo pg
```

常用参数：

- `--episodes`: 训练回合数
- `--run-dir`: 输出目录
- `--save-every`: checkpoint 保存间隔
- `--moving-avg-window`: 训练曲线平滑窗口
- `--render-eval`: 评估阶段打印地图轨迹

实时仿真（训练中观察 agent 真实移动和碰撞）：

```bash
python3 rl_nav.py --algo dqn --live-view --live-every 10 --live-fps 12
```

- `--live-view`: 打开训练实时可视化窗口
- `--live-every`: 每隔多少个 episode 渲染一次
- `--live-fps`: 可视化刷新帧率
- `--live-trail-len`: 轨迹保留长度

## 评估已训练策略

先加载 checkpoint 做固定地图评估：

```bash
python3 rl_nav.py --eval-only --checkpoint outputs/dqn_xxx/checkpoints/best.pt --eval-episodes 20
```

在随机地图分布上做泛化评估：

```bash
python3 rl_nav.py \
  --eval-only \
  --checkpoint outputs/dqn_xxx/checkpoints/best.pt \
  --eval-random-maps \
  --eval-num-maps 100 \
  --map-obstacle-prob 0.20 \
  --eval-steps 200
```

- `--eval-random-maps`: 启用随机地图批量评估
- `--eval-num-maps`: 随机地图数量
- `--map-obstacle-prob`: 障碍物采样概率
- `--map-no-require-path`: 不强制地图存在可达路径（默认强制有路）
- `--random-map-max-attempts`: 单张地图最多重采样次数

## 输出产物

每次训练在 `run_dir` 下保存：

- `config.json`: 训练与环境参数
- `metrics.csv`: 每回合指标（return/success/loss/epsilon等）
- `training_curve.png`: 训练可视化曲线（若可用 matplotlib）
- `checkpoints/best.pt`: 最优模型参数
- `checkpoints/last.pt`: 最后一次模型参数

## 新增算法

1. 在 `rlnav/algorithms/` 新建 agent，继承 `RLAgent` 接口。
2. 在 `rlnav/algorithms/__init__.py` 的 `AGENT_REGISTRY` 注册名称。
3. 通过 `python3 rl_nav.py --algo <name>` 直接复用同一训练基础设施进行对比。
