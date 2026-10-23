import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(norm_type: str, ch: int):
    norm_type = norm_type.lower()
    if norm_type == "bn":
        return nn.BatchNorm2d(ch)
    if norm_type == "gn":
        return nn.GroupNorm(8, ch)
    raise ValueError("norm_type must be 'bn' or 'gn'")


@torch.no_grad()
def cal_A_tensor(fs_list, ft_list, out_hw):
    """
    fs_list/ft_list: list of 3 tensors, each [B,C,h,w]
    out_hw: (H,W)
    return:
      A: [B,3,H,W]   (3 layers cosine diff evidence)
    """
    H, W = out_hw
    a_list = []
    for fs, ft in zip(fs_list, ft_list):
        a = 1.0 - F.cosine_similarity(fs, ft, dim=1)   # [B,h,w]
        a = a.unsqueeze(1)                             # [B,1,h,w]
        a = F.interpolate(a, size=(H, W), mode="bilinear", align_corners=False)
        a_list.append(a)
    return torch.cat(a_list, dim=1)                    # [B,3,H,W]


class PixelFusionGate(nn.Module):
    """
    兼容原脚本的类名，但内部已改成 residual additive gating

    输入:
        A: [B,3,H,W]
           三个通道分别是三层的 anomaly evidence: 1 - cosine_similarity

    输出:
        logits: [B,1,H,W]
        g:      [B,3,H,W]   每像素、每尺度的 gate，范围(0,1)

    结构:
        base       = A1 + A2 + A3
        correction = g1*A1 + g2*A2 + g3*A3
        fused      = base + alpha * correction
        logits     = scale * fused + bias
    """
    def __init__(self, hidden=16, norm_type="bn",
                 use_zscore=True, clamp_val=5.0,
                 clamp_A=True):
        super().__init__()
        self.use_zscore = use_zscore
        self.clamp_val = clamp_val
        self.clamp_A = clamp_A

        # gate feature extractor
        self.feat = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1, bias=False),
            make_norm(norm_type, hidden),
            nn.SiLU(inplace=True),

            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            make_norm(norm_type, hidden),
            nn.SiLU(inplace=True),
        )

        # 每像素输出 3 个 additive gates
        self.g_head = nn.Conv2d(hidden, 3, kernel_size=1, bias=True)

        # 初始化成“先接近原始 RD++ baseline”
        nn.init.zeros_(self.g_head.weight)
        nn.init.zeros_(self.g_head.bias)

        # alpha 控制 residual correction 的强度
        # sigmoid(-4) ≈ 0.018，初始几乎等于 base，训练后再慢慢打开 correction
        self.logit_alpha = nn.Parameter(torch.tensor([-4.0]))

        self.out_conv = nn.Conv2d(1, 1, kernel_size=1, bias=True)

        # 初始化成接近恒等映射
        nn.init.constant_(self.out_conv.weight, 1.0)
        nn.init.constant_(self.out_conv.bias, 0.0)



    @staticmethod
    def _zscore_per_channel(x, eps=1e-6):
        """
        x: [B,3,H,W]
        每通道按(H,W)做 z-score
        """
        mu = x.mean(dim=(2, 3), keepdim=True)
        sd = x.std(dim=(2, 3), keepdim=True, unbiased=False)
        return (x - mu) / (sd + eps)

    def forward(self, A):
        """
        A: [B,3,H,W]
        """
        A_raw = A
        if self.clamp_A:
            A_raw = A_raw.clamp(0.0, 2.0)

        # gate 分支只用来“看模式”，不是直接替代 anomaly map
        x = A_raw
        if self.use_zscore:
            x = self._zscore_per_channel(x).clamp(-self.clamp_val, self.clamp_val)

        h = self.feat(x)

        # additive gates: 不竞争，不要求和为1
        g = torch.sigmoid(self.g_head(h))               # [B,3,H,W], 每个 gate ∈ (0,1)

        # baseline: 原始 RD++ 多尺度累加
        base = A_raw.sum(dim=1, keepdim=True)          # [B,1,H,W]

        # learned correction: 只做增强/修正
        correction = (g * A_raw).sum(dim=1, keepdim=True)   # [B,1,H,W]

        # residual 强度，初始非常小，避免一开始破坏 baseline
        alpha = torch.sigmoid(self.logit_alpha)        # 标量，初始约 0.018

        fused = base + alpha * correction              # [B,1,H,W]

        score = torch.sigmoid(self.out_conv(fused))

        return score, g