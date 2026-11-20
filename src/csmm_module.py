import torch
import torch.nn as nn


class PatchBranch(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        pad = patch_size // 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=patch_size,
                stride=1,
                padding=pad,
                groups=mid_channels,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w


class SEAMHeadCSMM(nn.Module):
    def __init__(self, c1: int | None = None, patch_sizes=(1, 3, 5), reduction: int = 8):
        super().__init__()
        self.c1 = c1
        self.patch_sizes = tuple(int(p) for p in patch_sizes)
        self.reduction = reduction
        self._built = False
        if c1 is not None:
            self._build(int(c1))

    def _build(self, channels: int):
        self.c1 = channels
        mid_c = max(16, channels // 2)
        self.branches = nn.ModuleList(
            [PatchBranch(channels, mid_c, p) for p in self.patch_sizes]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * len(self.patch_sizes), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.se = SEBlock(channels, reduction=self.reduction)
        self.act = nn.SiLU(inplace=True)
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self._build(x.shape[1])
            self.to(device=x.device, dtype=x.dtype)
        feats = [branch(x) for branch in self.branches]
        fused = self.fuse(torch.cat(feats, dim=1))
        out = self.se(fused)
        return self.act(x + out)


def register_csmm():
    import ultralytics.nn.modules as ul_modules
    import ultralytics.nn.tasks as ul_tasks

    setattr(ul_modules, "SEAMHeadCSMM", SEAMHeadCSMM)
    setattr(ul_tasks, "SEAMHeadCSMM", SEAMHeadCSMM)
