import torch
import torch.nn as nn
import torch.nn.functional as F

import gdown
import config
from attribution.base import AttributionMethod

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------

DEVICE = torch.device(
    config.DEVICE if hasattr(config, "DEVICE")
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

# Canonical U2Net input size
_U2NET_SIZE = (320, 320)
_MODEL_CACHE: dict[str, nn.Module] = {}

# Specific path requested by the user
MODELS_DIR = getattr(config, "MODELS_DIR", None)
WEIGHTS_PATH = MODELS_DIR / "u2net.pth"

# Direct download link for the provided Google Drive ID
DRIVE_ID = "1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ"
REMOTE_URL = f"https://drive.google.com/uc?id={DRIVE_ID}&export=download"


# ---------------------------------------------------------------------------
# U-2-Net Architecture
# ---------------------------------------------------------------------------

class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super().__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=dirate, dilation=dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn_s1(self.conv_s1(x)))


def _upsample_like(src, tgt):
    return F.interpolate(src, size=tgt.shape[2:], mode="bilinear", align_corners=False)


class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool1(hx1))
        hx3 = self.rebnconv3(self.pool2(hx2))
        hx4 = self.rebnconv4(self.pool3(hx3))
        hx5 = self.rebnconv5(self.pool4(hx4))
        hx6 = self.rebnconv6(self.pool5(hx5))
        hx7 = self.rebnconv7(hx6)
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx5d = self.rebnconv5d(torch.cat((_upsample_like(hx6d, hx5), hx5), 1))
        hx4d = self.rebnconv4d(torch.cat((_upsample_like(hx5d, hx4), hx4), 1))
        hx3d = self.rebnconv3d(torch.cat((_upsample_like(hx4d, hx3), hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((_upsample_like(hx3d, hx2), hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((_upsample_like(hx2d, hx1), hx1), 1))
        return hx1d + hxin


class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool1(hx1))
        hx3 = self.rebnconv3(self.pool2(hx2))
        hx4 = self.rebnconv4(self.pool3(hx3))
        hx5 = self.rebnconv5(self.pool4(hx4))
        hx6 = self.rebnconv6(hx5)
        hx5d = self.rebnconv5d(torch.cat((_upsample_like(hx6, hx5), hx5), 1))
        hx4d = self.rebnconv4d(torch.cat((_upsample_like(hx5d, hx4), hx4), 1))
        hx3d = self.rebnconv3d(torch.cat((_upsample_like(hx4d, hx3), hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((_upsample_like(hx3d, hx2), hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((_upsample_like(hx2d, hx1), hx1), 1))
        return hx1d + hxin


class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool1(hx1))
        hx3 = self.rebnconv3(self.pool2(hx2))
        hx4 = self.rebnconv4(self.pool3(hx3))
        hx5 = self.rebnconv5(hx4)
        hx4d = self.rebnconv4d(torch.cat((_upsample_like(hx5, hx4), hx4), 1))
        hx3d = self.rebnconv3d(torch.cat((_upsample_like(hx4d, hx3), hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((_upsample_like(hx3d, hx2), hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((_upsample_like(hx2d, hx1), hx1), 1))
        return hx1d + hxin


class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool1(hx1))
        hx3 = self.rebnconv3(self.pool2(hx2))
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((_upsample_like(hx4, hx3), hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((_upsample_like(hx3d, hx2), hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((_upsample_like(hx2d, hx1), hx1), 1))
        return hx1d + hxin


class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        return hx1d + hxin


class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F(512, 256, 512)

        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx1 = self.stage1(x)
        hx2 = self.stage2(self.pool12(hx1))
        hx3 = self.stage3(self.pool23(hx2))
        hx4 = self.stage4(self.pool34(hx3))
        hx5 = self.stage5(self.pool45(hx4))
        hx6 = self.stage6(self.pool56(hx5))

        hx5d = self.stage5d(torch.cat((_upsample_like(hx6, hx5), hx5), 1))
        hx4d = self.stage4d(torch.cat((_upsample_like(hx5d, hx4), hx4), 1))
        hx3d = self.stage3d(torch.cat((_upsample_like(hx4d, hx3), hx3), 1))
        hx2d = self.stage2d(torch.cat((_upsample_like(hx3d, hx2), hx2), 1))
        hx1d = self.stage1d(torch.cat((_upsample_like(hx2d, hx1), hx1), 1))

        d1 = self.side1(hx1d)
        d2 = _upsample_like(self.side2(hx2d), d1)
        d3 = _upsample_like(self.side3(hx3d), d1)
        d4 = _upsample_like(self.side4(hx4d), d1)
        d5 = _upsample_like(self.side5(hx5d), d1)
        d6 = _upsample_like(self.side6(hx6), d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        return torch.sigmoid(d0), d1, d2, d3, d4, d5, d6


# ---------------------------------------------------------------------------
# Loader & Public Method
# ---------------------------------------------------------------------------

def _ensure_u2net():
    if "u2net" not in _MODEL_CACHE:
        if not WEIGHTS_PATH.exists():
            print(f"[U2Net] Weights missing at {WEIGHTS_PATH}. Downloading from Drive...")
            gdown.download(REMOTE_URL, str(WEIGHTS_PATH), quiet=False)

        print(f"[U2Net] Initializing model on {DEVICE}...")
        model = U2NET(in_ch=3, out_ch=1)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        model.to(DEVICE).eval()
        _MODEL_CACHE["u2net"] = model
    return _MODEL_CACHE["u2net"]


class U2NetSaliencyMethod(AttributionMethod):
    """
    Independent Saliency Method using U-2-Net.
    Operates at C:\\Users\\manoy\\Desktop\\לימודים\\פרויקט גמר\\P-metric\\models
    """

    def __init__(self, method_name: str = "u2net_saliency") -> None:
        super().__init__(method_name)
        self._u2net = None

    def _get_u2net(self):
        if self._u2net is None:
            self._u2net = _ensure_u2net()
        return self._u2net

    def compute(
        self,
        model: nn.Module,  # Ignored
        images: torch.Tensor,  # (B, 3, H, W)
        targets: torch.Tensor,  # Ignored
    ) -> torch.Tensor:
        B, C, H, W = images.shape
        u2net = self._get_u2net()

        # Resize batch to 320x320 for U2Net
        img_input = F.interpolate(images, size=_U2NET_SIZE, mode="bilinear", align_corners=False)

        with torch.no_grad():
            d0, *_ = u2net(img_input)
            # Upsample back to original input resolution
            heatmap = F.interpolate(d0, size=(H, W), mode="bilinear", align_corners=False)

        return self._normalize_attribution(heatmap)