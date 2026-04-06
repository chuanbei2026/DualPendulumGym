[English](README.md)

# DualPendulumGym

基于 [Gymnasium](https://gymnasium.farama.org/) 的小车双摆摆起与平衡环境，配有 3D OpenGL 渲染。训练 RL 智能体仅通过水平推力将两根铰接杆摆起并保持直立平衡。

<p align="center">
  <img src="docs/demo.gif" width="720" alt="训练好的智能体平衡双摆">
</p>

## 问题描述

一辆小车沿水平轨道滑动。两根杆通过铰链连接：杆 1 悬挂在小车上，杆 2 悬挂在杆 1 的末端（类似双节棍）。唯一的控制输入是向左或向右推动小车。目标是将两根杆从自然下垂位置摆到完全直立，并**尽可能长时间**保持平衡。

这是一个经典的欠驱动控制问题。系统有 3 个自由度（小车位置、杆 1 角度、杆 2 角度）但只有 1 个控制输入，这使得它比单倒立摆难得多。

## 特性

- **拉格朗日力学**，RK4 积分（每步 4 个子步，200 秒内能量漂移仅 0.006%）
- **3D OpenGL 渲染**，实时 HUD 显示动作、力度条、平衡连续时长
- **手动游玩模式**，支持键盘控制和演示录制
- **三阶段训练流水线**：SFT（行为克隆）-> GRPO（组相对策略优化）
- **自适应力度增长**：持续向同一方向推动会增加力的大小
- **状态指示灯**：红色（杆在下方）-> 绿色（均在水平线上方）-> 蓝色（杆 2 在杆 1 上方）

## 架构

```
dual_pendulum_gym/
  physics/dynamics.py      # 拉格朗日运动方程、RK4 积分器、物理参数
  envs/dual_pendulum.py    # Gymnasium 环境、奖励函数、观测空间
  rendering/renderer.py    # PyOpenGL 3D 渲染器 + HUD
  training/
    actor_critic.py        # 共享 MLP 策略网络 (14 -> 128 -> 128 -> 3)
    sft.py                 # 第一阶段：从人类演示进行行为克隆
    train.py               # 第二阶段：并行环境下的 GRPO 训练
  play.py                  # 手动游玩模式 + 演示录制
  eval.py                  # 加载模型评估 + 渲染
```

## 物理模型

系统使用拉格朗日力学推导小车-双摆耦合系统的精确运动方程：

| 参数 | 值 | 描述 |
|------|-----|------|
| 小车质量 | 20.0 kg | 较重的小车作为稳定底座 |
| 杆 1 质量 | 0.3 kg | 相对小车较轻的杆 |
| 杆 2 质量 | 0.5 kg | 略重的外杆 |
| 杆 1 长度 | 1.0 m | |
| 杆 2 长度 | 0.8 m | |
| 力（最小/最大） | 80 / 250 N | 持续推动时逐渐增大 |
| 轨道长度 | 10.0 m | 到达边界时碰撞 |
| 小车摩擦力 | 8.0 N*s/m | 轨道粘性摩擦 |
| 关节阻尼 | 0.1 N*m*s/rad | 轻微铰链阻尼 |

**自适应力度**：当小车连续向同一方向推动时，力在 30 帧内从 80N 增长到 250N。反向时重置为 80N。这鼓励了摆起所需的泵送运动。

## 观测空间（14 维）

| # | 特征 | 范围 | 描述 |
|---|------|------|------|
| 1 | `x_norm` | [-1, 1] | 小车位置（+-1 = 墙壁） |
| 2 | `x_dot` | 无界 | 小车速度 |
| 3 | `x_accel` | 无界 | 小车加速度 |
| 4 | `sin(th1)` | [-1, 1] | 杆 1 角度（无不连续性） |
| 5 | `cos(th1)` | [-1, 1] | 杆 1 角度 |
| 6 | `sin(th2)` | [-1, 1] | 杆 2 角度 |
| 7 | `cos(th2)` | [-1, 1] | 杆 2 角度 |
| 8 | `th1_dot` | 无界 | 杆 1 角速度 |
| 9 | `th2_dot` | 无界 | 杆 2 角速度 |
| 10 | `y_rod1` | [-1, 1] | 杆 1 末端高度（归一化） |
| 11 | `y_rod2` | [-1, 1] | 杆 2 末端高度（归一化） |
| 12 | `idle_time` | [0, 1] | 杆 1 上次高于水平线至今的时间 |
| 13 | `force_ramp` | [0, 1] | 当前力度增长级别 |
| 14 | `balance_streak` | [0, 1] | 连续平衡步数 |

使用 `sin/cos` 替代原始角度可避免 +-pi 处的不连续性。杆末端高度和平衡连续时长让模型直接获取与奖励相关的特征。

## 奖励设计

奖励函数分两个阶段：

**摆起阶段**（任一杆低于水平线）：
- 质心高度：`1.0 * h1_norm + 3.0 * h2_norm`（杆 2 权重为 3 倍）
- 进步奖励：质心高度的变化量，放大 3 倍
- 空闲惩罚：杆 1 在水平线以下超过 200 步后逐步加重惩罚

**平衡阶段**（两根杆均在水平线上方）：
- 质心高度乘以**连续时长乘数**（200 步内从 1x 增长到 3x）
- 稳定性奖励：低角速度时给予奖励（鼓励静止而非旋转）
- 旋转惩罚：角速度 > 4 rad/s 时施加惩罚（防止"大环"作弊）

**始终生效：**
- 撞墙：-10 惩罚 + 回合终止
- 靠近墙壁：到达轨道 80% 处开始惩罚

此设计经过多次迭代，以封堵奖励作弊漏洞（旋转、贴墙、空闲振荡）。

## 训练流水线

### 第一阶段：监督微调 (SFT)

录制人类演示，然后通过行为克隆训练策略：

```bash
# 录制演示（方向键 = 移动，ESC = 退出）
python -m dual_pendulum_gym.play --record demos/human_demo.npz

# 训练 SFT 模型
python -m dual_pendulum_gym.training.sft --demos "demos/human_demo.npz" --save-path checkpoints/model_sft.pt
```

SFT 给智能体提供了一个合理的初始策略（~95% 动作准确率），避免 RL 从随机行为开始。

### 第二阶段：GRPO（组相对策略优化）

GRPO 是一种无 Critic 的 RL 算法，灵感来自 [DeepSeek-Math](https://arxiv.org/abs/2402.03300)。它不训练价值函数来估计优势，而是：

1. 从当前策略运行 **N 条并行轨迹**（默认 4 条）
2. 按组内总奖励排序
3. 计算**组相对优势**：`(reward - group_mean) / group_std`
4. 使用 PPO 风格的截断代理目标进行更新

```bash
# 从 SFT 检查点开始 GRPO 训练（带实时渲染）
python -m dual_pendulum_gym.training.train \
    --pretrained checkpoints/model_sft.pt \
    --render --render-fps 120 \
    --group-size 4 --max-episodes 5000
```

**为什么选 GRPO 而不是 PPO？** 标准 PPO 需要一个 Critic 网络来估计状态价值。在我们的实验中，Critic 经常校准不准确，导致策略崩溃（智能体突然开始撞墙）。GRPO 完全去掉了 Critic——"基线"就是组内平均奖励，天然准确。

### 训练结果

| 方法 | 平均奖励（100 回合） | 稳定性 |
|------|---------------------|--------|
| A2C | -80（崩溃） | 约 12 回合后策略崩溃 |
| PPO | 1,000 | 逐渐退化 |
| **GRPO** | **930+**（持续上升） | 稳定，无崩溃 |

GRPO 配合质心奖励在 250+ 组训练中实现了持续提升，无任何策略崩溃。

## 快速开始

```bash
# 安装
pip install -e ".[train]"

# 手动游玩
python -m dual_pendulum_gym.play

# 录制演示用于 SFT
python -m dual_pendulum_gym.play --record demos/my_demo.npz

# SFT 训练
python -m dual_pendulum_gym.training.sft --demos "demos/my_demo.npz"

# GRPO 训练
python -m dual_pendulum_gym.training.train --pretrained checkpoints/model_sft.pt --render

# 评估训练好的模型
python -m dual_pendulum_gym.eval --model checkpoints/model_best.pt
```

## 作为 Gymnasium 环境使用

```python
import gymnasium as gym
import dual_pendulum_gym

env = gym.make("DualPendulum-v0", render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # 0=左, 1=不动, 2=右
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## 依赖

- Python >= 3.9
- gymnasium, numpy, pygame, PyOpenGL
- PyTorch >= 2.0（仅训练时需要）

## 经验总结

1. **奖励作弊真实存在**：智能体找到了每一种捷径——大圈旋转、底部振荡、贴墙。每种漏洞都需要针对性的奖励修复。
2. **RL 前先做 SFT 至关重要**：纯 RL 从零开始在 12 回合内就会撞墙。从人类演示出发提供了可行的基础策略。
3. **无 Critic 的 RL (GRPO) 更鲁棒**：去掉价值网络消除了此任务中训练不稳定的主要来源。
4. **观测设计很重要**：从原始角度切换到 `sin/cos` 编码，加入杆末端高度，以及包含任务相关特征（平衡连续时长、力度增长级别）显著提升了学习效果。
5. **自适应力度增长**鼓励了人类在摆起时直觉使用的泵送运动。

## 许可证

MIT
