# GbI: two-hot / symlog 工具（DreamerV3 式）。
#
# 动机（GbI.md §3.1，v5 P1 修复）：
#   - metaworld 各任务奖励尺度差 3 个数量级且存在饱和平台/重尾（drawer-close 类），
#     MSE 回归会学出状态依赖的大幅偏置（"坏尺子"）。
#   - 将奖励经 symlog 压缩后，在分箱上做 two-hot soft-label 交叉熵回归：
#     * symlog 压缩跨任务尺度差，免去逐任务 EMA 归一化；
#     * two-hot 交叉熵不惩罚"落在高概率邻箱"的预测，压制重尾样本绑架梯度。
#
# 默认口径（对应 config/agent/components/world_model.yaml）：
#   reward_bins=101, reward_min=-400, reward_max=+400（原始奖励空间上下界），
#   分箱均匀铺在 [symlog(reward_min), symlog(reward_max)] 上（≈[-5.99, 5.99]）。

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from mtrl.utils.types import TensorType


def symlog(x: TensorType) -> TensorType:
    """y = sign(x) * ln(|x| + 1)，对任意实值定义域无下溢问题。"""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(y: TensorType) -> TensorType:
    """symlog 的逆映射。"""
    return torch.sign(y) * (torch.expm1(torch.abs(y)))


class TwoHotSymlog:
    """奖励 → two-hot 标签（训练目标）；分箱 → 奖励均值（打分路径的通路）。

    分箱铺在 symlog 空间：low = symlog(reward_min), high = symlog(reward_max)。
    """

    def __init__(
        self,
        reward_bins: int = 101,
        reward_min: float = -400.0,
        reward_max: float = 400.0,
    ):
        assert reward_bins >= 3, "two-hot 需要至少 3 个分箱"
        assert reward_min < reward_max
        self.reward_bins = reward_bins
        self.reward_min = float(reward_min)
        self.reward_max = float(reward_max)
        self.low = symlog(torch.tensor(self.reward_min, dtype=torch.float32)).item()
        self.high = symlog(torch.tensor(self.reward_max, dtype=torch.float32)).item()
        # (bins,) 分箱中心，symlog 空间
        self.register_bins()

    def register_bins(self) -> None:
        centers = torch.linspace(self.low, self.high, self.reward_bins)
        self.register_buffer("bin_centers", centers)  # symlog 空间
        # symexp 回原始奖励空间（打分/偏置表反解用）
        self.register_buffer("bin_centers_raw", symexp(centers))

    # ---- 与 nn.Module.splitter 一致的注册接口（组件内部持有，非独立 module） ----
    def register_buffer(self, name: str, tensor: TensorType) -> None:
        self.__dict__[name] = tensor


def twohot_label(
    reward: TensorType,
    twohot: TwoHotSymlog,
    device: torch.device,
) -> TensorType:
    """把 (B, 1) 原始奖励转为 (B, bins) two-hot soft 标签。

    位于 symlog(reward) 两侧的最近两个分箱按线性距离分配概率权重，
    其余分箱概率为 0，行和为 1。
    """
    y = symlog(reward.float())  # (B, 1)
    centers = twohot.bin_centers.to(device)  # (bins,)
    # 用上界式编码: 把 y 映射到 [0, bins-1] 标尺上
    span = centers[-1] - centers[0]
    offset = (y - centers[0]).clamp(min=0.0, max=span)  # (B, 1) 非负
    scale = (twohot.reward_bins - 1) / span
    pos = offset * scale  # 连续分箱位置 (B, 1)
    lo = pos.floor().long()  # (B, 1)
    hi = lo + 1
    frac = (pos - lo.float()).clamp(min=0.0, max=1.0)  # 高箱权重
    # 边界: hi 越界时把权重全给 lo
    hi_ok = (hi < twohot.reward_bins).float()
    lo_w = (1.0 - frac) + (1.0 - hi_ok) * frac
    hi_w = frac * hi_ok
    hi_clamped = hi.clamp(max=twohot.reward_bins - 1)

    B = reward.shape[0]
    label = torch.zeros(B, twohot.reward_bins, device=device, dtype=torch.float32)
    label.scatter_add_(1, lo, lo_w)
    label.scatter_add_(1, hi_clamped, hi_w)
    return label


def bins_to_reward(logits: TensorType, twohot: TwoHotSymlog) -> TensorType:
    """two-hot logits → 期望奖励（原始空间）。

    期望在 symlog 分箱空间计算后整体 symexp 反解（DreamerV3 口径）：
    E[r] = symexp(Σ probs·center_symlog)。
    早期版本直接在 raw 空间加权（Σ probs·symexp(center)）：bin 中心被
    symexp 拉到 ±400，softmax 尾部微小概率泄漏到极端 bin 即绑架期望
    （±2 级偏差，远大于真实 scaled reward ±1），想象打分噪声过大
    （中危 #8，2026-09-02 修复）。
    """
    probs = torch.softmax(logits, dim=-1)  # (..., bins)
    y = (probs * twohot.bin_centers.to(logits.device)).sum(dim=-1)
    return symexp(y)


def twohot_loss(
    logits: TensorType,
    reward: TensorType,
    twohot: TwoHotSymlog,
    reduction: str = "none",
    weights: TensorType = None,
) -> TensorType:
    """two-hot 交叉熵损失。

    Args:
        logits: (B, bins) 奖励头输出。
        reward: (B, 1) 原始奖励。
        weights: (B, 1) 可选样本权重（残差加权流用）。
    """
    target = twohot_label(reward, twohot, logits.device)  # (B, bins)
    loss = -torch.sum(target * F.log_softmax(logits, dim=-1), dim=-1)  # (B,)
    if weights is not None:
        loss = loss * weights.squeeze(-1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "none":
        return loss
    raise ValueError(f"unknown reduction: {reduction}")