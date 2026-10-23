import random
import math
import numpy as np
from PIL import Image
from torchvision import transforms

class CutPaste(object):
    """Base class for cutpaste variants with common operations"""
    def __init__(self, **kwargs):
        # ✅ 接受并忽略多余 kwargs，避免上层乱传参直接 TypeError
        self.colorJitter = transforms.ColorJitter(
            brightness=(0.1, 0.3),
            contrast=(0.5, 0.7),
            saturation=(0.5, 0.7),
            hue=(-0.1, 0.1)
        )

class CutPasteNormal(CutPaste):
    def __init__(self, area_ratio=[0.02, 0.15], aspect_ratio=0.3, **kwargs):
        super().__init__(**kwargs)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):
        w, h = img.size
        ratio_area = random.uniform(*self.area_ratio) * w * h
        aspect = math.exp(random.uniform(math.log(self.aspect_ratio), math.log(1.0 / self.aspect_ratio)))
        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))

        from_x = random.randint(0, w - cut_w)
        from_y = random.randint(0, h - cut_h)
        patch = img.crop((from_x, from_y, from_x + cut_w, from_y + cut_h))
        if self.colorJitter:
            patch = self.colorJitter(patch)

        to_x = random.randint(0, w - cut_w)
        to_y = random.randint(0, h - cut_h)
        aug = img.copy()
        aug.paste(patch, (to_x, to_y))

        mask = Image.new("L", (w, h), 0)
        patch_mask = Image.new("L", (cut_w, cut_h), 255)
        mask.paste(patch_mask, (to_x, to_y))
        return aug, mask

class CutPasteScar(CutPaste):
    def __init__(self, width=[2, 10], height=[25, 45], rotation=[-45, 45], **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.rotation = rotation

    def __call__(self, img):
        w, h = img.size
        cut_w = int(random.uniform(*self.width))
        cut_h = int(random.uniform(*self.height))

        from_x = random.randint(0, w - cut_w)
        from_y = random.randint(0, h - cut_h)
        patch = img.crop((from_x, from_y, from_x + cut_w, from_y + cut_h))
        if self.colorJitter:
            patch = self.colorJitter(patch)

        rot_deg = random.uniform(*self.rotation)
        patch = patch.convert("RGBA").rotate(rot_deg, expand=True)
        alpha = patch.split()[-1]          # 旋转后 alpha 当作 mask
        patch = patch.convert("RGB")

        aug = img.copy()
        to_x = random.randint(0, w - patch.size[0])
        to_y = random.randint(0, h - patch.size[1])
        aug.paste(patch, (to_x, to_y), mask=alpha)

        mask_img = Image.new("L", (w, h), 0)
        mask_img.paste(alpha, (to_x, to_y))
        return aug, mask_img

class CutPasteMix(object):
    """
    ✅ 返回统一：只返回 (aug, mask)
    ✅ 参数拆分：normal 用 area_ratio/aspect_ratio；scar 用 width/height/rotation
    """
    def __init__(self,
                 area_ratio=[0.02, 0.15], aspect_ratio=0.3,
                 width=[2, 16], height=[10, 25], rotation=[-45, 45],
                 **kwargs):
        self.normal = CutPasteNormal(area_ratio=area_ratio, aspect_ratio=aspect_ratio, **kwargs)
        self.scar   = CutPasteScar(width=width, height=height, rotation=rotation, **kwargs)

    def __call__(self, img):
        r = random.random()
        if r < 1 / 3:
            return self.normal(img)   # (aug, mask)
        elif r < 2 / 3:
            return self.scar(img)     # (aug, mask)
        else:
            # Mix: Normal + Scar (non-overlapping)
            aug1, mask1 = self.normal(img)

            # mask1 bbox
            mask_np = np.array(mask1)
            coords = np.argwhere(mask_np > 0)
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                avoid_box = (x_min, y_min, x_max, y_max)
            else:
                avoid_box = None

            aug2, mask2 = self.apply_scar_avoid(aug1, avoid_box)

            final_mask = Image.fromarray(np.maximum(np.array(mask1), np.array(mask2)).astype(np.uint8))
            return aug2, final_mask

    def apply_scar_avoid(self, aug1, avoid_box, max_attempts=10):
        for _ in range(max_attempts):
            aug2, mask2 = self.scar(aug1)

            mask_np = np.array(mask2)
            coords = np.argwhere(mask_np > 0)
            if len(coords) == 0:
                continue
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            new_box = (x_min, y_min, x_max, y_max)

            if avoid_box is None or not self.overlapping(avoid_box, new_box):
                return aug2, mask2

        # 尝试失败就返回最后一次（至少有输出）
        return aug2, mask2

    @staticmethod
    def overlapping(box1, box2):
        l1, t1, r1, b1 = box1
        l2, t2, r2, b2 = box2
        return not (r1 <= l2 or r2 <= l1 or b1 <= t2 or b2 <= t1)
