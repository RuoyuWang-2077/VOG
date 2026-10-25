# DVOC 算法与代码对应

1. **采样**：当前 LoRA 策略执行 N=9 步确定性 Euler，缓存每个状态与速度，采样图全部 detach。Z-Image 的 Euler 速度为 transformer 原始输出的负值，模型时间使用 `(1000 - scheduler_timestep) / 1000`。
2. **目标**：从实际终点 y 开始做 J 次归一化奖励梯度上升。总半径为 `rho * ||y||₂`，每步长度为 `eta * radius / J`，每步投影回球内。J=2 为默认设置。proxy 变体从 `x_q - sigma_q * v_q` 开始，半径仍由实际终点确定，再将获得的位移加到 y。
3. **稀疏拟合**：每条轨迹使用计划里自己的 K 个活跃状态。只在这些 detach 的状态重算可微速度，其余速度使用缓存；按完整 Euler 和式重建终点 `y_hat`。
4. **连接与损失**：`connected = y_hat + stopgrad(y - y_hat)`，`target = stopgrad(y + delta)`。损失为逐坐标 MSE 除以每个样本的平均绝对位移，分母 detach，最低 1e-5，计算分母时使用 float64。没有 N/K 倍率。
5. **参数更新**：逐样本累积梯度后跨 rank 求平均，clip norm=1，然后更新 AdamW 与 EMA。EMA 衰减为 `min((1+step)/(10+step), 0.9)`。

目标优化只对输入 latent 求导。拟合时梯度只穿过活跃状态处的速度预测；缓存轨迹和目标不参与反向。训练数据计划记录 prompt、noise seed、K 个状态、每次目标梯度的随机种子，J1/J2 共享第一次目标随机流。

主要对应关系：目标构造见 `algorithm/targets.py`；终点合成与损失见 `algorithm/objective.py`；模型时间和速度符号见 `models/zimage.py`；完整阶段调用见 `training/phases.py`。
