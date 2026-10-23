# pv_cutpaste.py
# 光伏伪异常生成：大面积黑心（blob） + 微裂纹（随机方向 + 分叉）
# ✅ 返回统一：(aug, mask)  其中 mask 为 PIL.Image("L")，带羽化边缘
# ✅ Mix 时严格“像素级不重叠”（不是 bbox），放不下就宁可不叠加第二种缺陷
# ✅ 黑心改为：随机灰度“覆盖”（cover），而不是乘法变暗（不透底纹）
# ✅ 兼容你之前 CutPasteMix 的 **kwargs 乱传（会被忽略），不会 TypeError

import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# -----------------------------
# utils
# -----------------------------
def _to_int_range(x, default=(1, 1)):
    """把 (a,b)/[a,b]/a 转成 (int(a), int(b))"""
    if x is None:
        return (int(default[0]), int(default[1]))
    if isinstance(x, (int, np.integer)):
        v = int(x)
        return (v, v)
    a, b = x
    return (int(a), int(b))


def _to_float_range(x, default=(0.0, 1.0)):
    """把 (a,b)/[a,b]/a 转成 (float(a), float(b))"""
    if x is None:
        return (float(default[0]), float(default[1]))
    if isinstance(x, (int, float, np.number)):
        v = float(x)
        return (v, v)
    a, b = x
    return (float(a), float(b))


def _bin_mask(mask_l: Image.Image, thr=80):
    """PIL L -> binary numpy mask；thr 越大，占用区域越小（适合羽化黑心）"""
    m = np.array(mask_l)
    return (m >= int(thr)).astype(np.uint8)


def _has_overlap(bin1, bin2):
    """二值mask是否有相交像素"""
    return np.any((bin1 > 0) & (bin2 > 0))


def _apply_alpha_overlay_gray(img_rgb: Image.Image, alpha_l: Image.Image, gray: int):
    """
    ✅ 随机灰度覆盖（cover，不透底纹）
    img_rgb: PIL RGB
    alpha_l: PIL L (0..255), feathered
    gray: 0..255，覆盖区域的灰度值（0黑，255白）
    """
    img = np.asarray(img_rgb).astype(np.float32) / 255.0  # [H,W,3]
    a = np.asarray(alpha_l).astype(np.float32) / 255.0    # [H,W]
    a = a[..., None]                                      # [H,W,1]

    g = float(int(gray)) / 255.0
    col = np.array([g, g, g], dtype=np.float32)[None, None, :]  # [1,1,3]
    out = img * (1.0 - a) + col * a

    out = (out * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _apply_alpha_darkening(img_rgb: Image.Image, alpha_l: Image.Image, dark_factor: float):
    """
    （保留给裂纹用）乘法变暗：会透出底纹，更像“微裂纹/浅划痕”
    dark_factor: 0..1，越小越黑（例如 0.4）
    """
    img = np.asarray(img_rgb).astype(np.float32) / 255.0
    a = np.asarray(alpha_l).astype(np.float32) / 255.0
    a = a[..., None]

    img_dark = img * float(dark_factor)
    out = img * (1.0 - a) + img_dark * a
    out = (out * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


# -----------------------------
# 1) 大面积黑心：不规则 blob + 羽化边缘 + 随机灰度覆盖（不透底纹）
# -----------------------------
class PVBlackSpot(object):
    def __init__(
        self,
        area_ratio=(0.03, 0.25),     # 黑心占整图面积比例
        n_blobs=(2, 4),              # 叠加几个椭圆形成不规则形
        blur_sigma=(6, 18),          # 羽化强度（越大边缘越软）
        cover_gray=(0, 70),          # ✅ 覆盖灰度范围：0~70 = 黑到深灰（不透底纹）
        **kwargs
    ):
        self.area_ratio = _to_float_range(area_ratio, (0.03, 0.25))
        self.n_blobs = _to_int_range(n_blobs, (2, 6))
        self.blur_sigma = _to_float_range(blur_sigma, (6, 18))
        self.cover_gray = _to_int_range(cover_gray, (0, 70))

    def __call__(self, img: Image.Image):
        w, h = img.size
        target_area = random.uniform(*self.area_ratio) * w * h

        # 估算一个主半径（按面积反推圆半径）
        base_r = math.sqrt(target_area / math.pi)
        base_r = max(8, int(base_r))

        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        # 随机中心（避免越界）
        cx = random.randint(base_r, max(base_r, w - base_r))
        cy = random.randint(base_r, max(base_r, h - base_r))

        k = random.randint(self.n_blobs[0], self.n_blobs[1])
        for _ in range(k):
            rx = int(base_r * random.uniform(0.4, 1.2))
            ry = int(base_r * random.uniform(0.4, 1.2))
            ox = int(random.uniform(-0.6, 0.6) * base_r)
            oy = int(random.uniform(-0.6, 0.6) * base_r)

            x0 = max(0, cx + ox - rx)
            y0 = max(0, cy + oy - ry)
            x1 = min(w - 1, cx + ox + rx)
            y1 = min(h - 1, cy + oy + ry)
            draw.ellipse([x0, y0, x1, y1], fill=255)

        sigma = random.uniform(*self.blur_sigma)
        mask_feather = mask.filter(ImageFilter.GaussianBlur(radius=sigma))

        # ✅ 随机灰度覆盖：每个黑心亮度不同，但不透底纹
        g = random.randint(self.cover_gray[0], self.cover_gray[1])
        aug = _apply_alpha_overlay_gray(img.convert("RGB"), mask_feather, gray=g)
        return aug, mask_feather


# -----------------------------
# 2) 微裂纹：随机方向 + 缓慢漂移 + 分叉（连续线，不断续）
# -----------------------------
class PVMicroCrack(object):
    def __init__(
        self,
        n_cracks=(1, 2),
        thickness=(1, 2),            # 线宽（像素），必须是 int
        length_ratio=(0.25, 0.55),    # 裂纹长度相对对角线比例
        jitter=(2, 8),               # 折线抖动幅度
        segments=(10, 26),           # 折线段数（曲折次数核心）
        blur_sigma=(0.8, 1.8),       # 轻微羽化
        dark_factor=(0.3, 0.6),      # 裂纹暗化（建议偏大：更淡更像微裂）
        drift_deg=(2, 10),           # 每段方向漂移（度）
        branch_prob=0.30,            # 每条主裂纹产生分叉概率
        branch_count=(1, 2),         # 分叉条数
        branch_len_ratio=(0.15, 0.45),
        branch_angle_deg=(20, 70),
        **kwargs
    ):
        self.n_cracks = _to_int_range(n_cracks, (1, 2))
        self.thickness = _to_int_range(thickness, (1, 2))
        self.length_ratio = _to_float_range(length_ratio, (0.25, 0.85))
        self.jitter = _to_float_range(jitter, (2, 8))
        self.segments = _to_int_range(segments, (10, 26))
        self.blur_sigma = _to_float_range(blur_sigma, (0.8, 1.8))
        self.dark_factor = _to_float_range(dark_factor, (0.3, 0.6))

        self.drift_deg = _to_float_range(drift_deg, (2, 10))
        self.branch_prob = float(branch_prob)
        self.branch_count = _to_int_range(branch_count, (1, 2))
        self.branch_len_ratio = _to_float_range(branch_len_ratio, (0.15, 0.45))
        self.branch_angle_deg = _to_float_range(branch_angle_deg, (20, 70))

    def _walk_polyline(self, w, h, start_xy, theta, total_len, n_seg):
        x, y = start_xy
        pts = [(x, y)]
        step = total_len / max(1, n_seg)

        drift_rad = math.radians(random.uniform(*self.drift_deg))
        for _ in range(n_seg):
            theta = theta + random.uniform(-drift_rad, drift_rad)

            dx = step * math.cos(theta) + random.uniform(-1, 1) * random.uniform(*self.jitter)
            dy = step * math.sin(theta) + random.uniform(-1, 1) * random.uniform(*self.jitter)
            x = int(round(x + dx))
            y = int(round(y + dy))
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            pts.append((x, y))
        return pts

    def _gen_main_and_branches(self, w, h):
        x0 = random.randint(0, w - 1)
        y0 = random.randint(0, h - 1)

        theta = random.uniform(-math.pi, math.pi)  # 主方向随机

        diag = math.sqrt(w * w + h * h)
        total_len = random.uniform(*self.length_ratio) * diag
        n_seg = random.randint(self.segments[0], self.segments[1])

        pts_main = self._walk_polyline(w, h, (x0, y0), theta, total_len, n_seg)

        branches = []
        if random.random() < self.branch_prob and len(pts_main) > 8:
            bc = random.randint(self.branch_count[0], self.branch_count[1])
            for _ in range(bc):
                idx = random.randint(len(pts_main) // 3, 2 * len(pts_main) // 3)
                bx, by = pts_main[idx]

                delta = math.radians(random.uniform(*self.branch_angle_deg)) * (1 if random.random() < 0.5 else -1)
                theta_b = theta + delta

                branch_len = random.uniform(*self.branch_len_ratio) * total_len
                branch_seg = max(6, int(n_seg * random.uniform(0.25, 0.55)))

                pts_branch = self._walk_polyline(w, h, (bx, by), theta_b, branch_len, branch_seg)
                branches.append(pts_branch)

        return pts_main, branches

    def gen_raw_mask(self, w, h):
        """只生成硬mask（不羽化不叠加），用于不重叠采样"""
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        n = random.randint(self.n_cracks[0], self.n_cracks[1])
        for _ in range(n):
            pts_main, branches = self._gen_main_and_branches(w, h)
            width = random.randint(self.thickness[0], self.thickness[1])

            draw.line(pts_main, fill=255, width=width)
            for pts_b in branches:
                draw.line(pts_b, fill=255, width=width)

        return mask

    def apply_from_raw_mask(self, img: Image.Image, raw_mask: Image.Image):
        """把 raw_mask 羽化 + 乘法暗化叠加到图像上（裂纹保留“透底纹”）"""
        sigma = random.uniform(*self.blur_sigma)
        mask_feather = raw_mask.filter(ImageFilter.GaussianBlur(radius=sigma))

        df = random.uniform(*self.dark_factor)
        aug = _apply_alpha_darkening(img.convert("RGB"), mask_feather, dark_factor=df)
        return aug, mask_feather

    def __call__(self, img: Image.Image):
        w, h = img.size
        raw = self.gen_raw_mask(w, h)
        return self.apply_from_raw_mask(img, raw)


# -----------------------------
# 3) Mix：黑心 / 裂纹 / 二者叠加（严格不重叠）
# -----------------------------
class PVCutPasteMix(object):
    """
    返回：(aug, mask)

    - spot / crack / mix 概率默认各 1/3
    - mix 时：先黑心，再采样裂纹 raw_mask，像素级检查不重叠，成功才叠加；
      尝试失败 -> 直接返回“只有黑心”(严格不重叠优先)
    """

    def __init__(
        self,
        # --- probs ---
        p_spot=1/3,
        p_crack=1/3,
        p_mix=1/3,

        # --- strict non-overlap ---
        spot_occ_thr=80,        # 黑心羽化mask二值化阈值：越大占用越小，越容易放裂纹
        max_attempts=30,        # mix 时裂纹重采样次数

        # --- spot params ---
        spot_area_ratio=(0.15, 0.25),
        spot_n_blobs=(2, 3),
        spot_blur_sigma=(6, 18),
        spot_cover_gray=(0, 70),   # ✅ 新增：黑心覆盖灰度范围（不透底纹）

        # --- crack params ---
        crack_n_cracks=(1, 3),
        crack_thickness=(6, 8),
        crack_length_ratio=(0.25, 0.55),
        crack_jitter=(4, 8),
        crack_segments=(10, 15),
        crack_blur_sigma=(0.8, 1.8),
        crack_dark_factor=(0.45, 0.6),
        crack_drift_deg=(3, 8),
        crack_branch_prob=0.20,
        crack_branch_count=(1, 2),
        crack_branch_len_ratio=(0.15, 0.45),
        crack_branch_angle_deg=(20, 70),

        # ignore extra kwargs
        **kwargs
    ):

        self.p_spot = float(p_spot)
        self.p_crack = float(p_crack)
        self.p_mix = float(p_mix)
        s = self.p_spot + self.p_crack + self.p_mix
        self.p_spot, self.p_crack, self.p_mix = self.p_spot / s, self.p_crack / s, self.p_mix / s

        self.spot_occ_thr = int(spot_occ_thr)
        self.max_attempts = int(max_attempts)

        self.black = PVBlackSpot(
            area_ratio=spot_area_ratio,
            n_blobs=spot_n_blobs,
            blur_sigma=spot_blur_sigma,
            cover_gray=spot_cover_gray,   # ✅ 传入随机灰度覆盖
        )

        self.crack = PVMicroCrack(
            n_cracks=crack_n_cracks,
            thickness=crack_thickness,
            length_ratio=crack_length_ratio,
            jitter=crack_jitter,
            segments=crack_segments,
            blur_sigma=crack_blur_sigma,
            dark_factor=crack_dark_factor,
            drift_deg=crack_drift_deg,
            branch_prob=crack_branch_prob,
            branch_count=crack_branch_count,
            branch_len_ratio=crack_branch_len_ratio,
            branch_angle_deg=crack_branch_angle_deg,
        )

    def _apply_crack_avoid_overlap(self, img_aug, spot_mask_feather):
        """
        在已有黑心的基础上加裂纹，严格不重叠：
        - 黑心用阈值 spot_occ_thr 得到“占用区域”
        - 裂纹用 raw_mask（二值）检查 overlap
        """
        w, h = img_aug.size
        spot_bin = _bin_mask(spot_mask_feather, thr=self.spot_occ_thr)

        for _ in range(self.max_attempts):
            raw_crack = self.crack.gen_raw_mask(w, h)
            crack_bin = _bin_mask(raw_crack, thr=1)  # 裂纹本来就是硬线

            if _has_overlap(spot_bin, crack_bin):
                continue

            # ✅ 不重叠才真正叠加
            aug2, crack_mask = self.crack.apply_from_raw_mask(img_aug, raw_crack)
            return aug2, crack_mask

        # 放不下就不叠加裂纹（严格不重叠）
        empty = Image.new("L", (w, h), 0)
        return img_aug, empty

    def __call__(self, img: Image.Image):
        r = random.random()
        if r < self.p_spot:
            return self.black(img)
        elif r < self.p_spot + self.p_crack:
            return self.crack(img)
        else:
            # Mix：先黑心，再裂纹（严格不重叠）
            aug1, mask1 = self.black(img)
            aug2, mask2 = self._apply_crack_avoid_overlap(aug1, mask1)

            # 合并 mask（不重叠时 max 即可）
            m = np.maximum(np.array(mask1), np.array(mask2)).astype(np.uint8)
            final_mask = Image.fromarray(m, mode="L")
            return aug2, final_mask


__all__ = ["PVBlackSpot", "PVMicroCrack", "PVCutPasteMix"]
