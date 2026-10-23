from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np
from torchvision.transforms import InterpolationMode

from dataset.cutpaste import CutPasteMix
from dataset.noise import Simplex_CLASS
import cv2

from dataset.pv_cutpaste import PVCutPasteMix


class ToTensor(object):
    def __call__(self, image):
        try:
            image = torch.from_numpy(image.transpose(2, 0,1))
        except:
            print('Invalid_transpose, please make sure images have shape (H, W, C) before transposing')
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image
    
    
class Normalize(object):
    """
    Only normalize images
    """
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)
    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image

def get_data_transforms(size, isize):
    data_transforms = transforms.Compose([Normalize(),\
                    ToTensor()])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size), interpolation= InterpolationMode.NEAREST),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

# class MVTecDataset_train(torch.utils.data.Dataset):
#     def __init__(self, root, transform):
#         self.img_path = root
#         self.simplexNoise = Simplex_CLASS()
#         self.transform = transform
#         # load dataset
#         self.img_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1
#
#     def load_dataset(self):
#         img_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.jpg")
#         return img_paths
#
#     def __len__(self):
#         return len(self.img_paths)
#
#     def __getitem__(self, idx):
#         img_path = self.img_paths[idx]
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img= cv2.resize(img/255., (512, 512))
#         ## Normal
#         img_normal = self.transform(img)
#         ## simplex_noise
#         size = 512
#         h_noise = np.random.randint(10, int(size//8))
#         w_noise = np.random.randint(10, int(size//8))
#         start_h_noise = np.random.randint(1, size - h_noise)
#         start_w_noise = np.random.randint(1, size - w_noise)
#         noise_size = (h_noise, w_noise)
#         simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 6, 0.6)
#         init_zero = np.zeros((512,512,3))
#         init_zero[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise, :] = 0.2 * simplex_noise.transpose(1,2,0)
#         img_noise = img + init_zero
#         img_noise = self.transform(img_noise)
#         return img_normal, img_noise, img_path.split('/')[-1]


class MVTecDataset_train(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.img_path = root
        self.transform = transform
        self.cutpaste = PVCutPasteMix()
        # self.cutpaste = CutPasteMix()
        # load dataset
        self.img_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.jpg")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= cv2.resize(img, (512, 512))


        img_pil = Image.fromarray(img)
        aug_pil, _ = self.cutpaste(img_pil)

        img_np = np.array(img_pil).astype(np.float32) / 255.0
        aug_np = np.array(aug_pil).astype(np.float32) / 255.0

        img_normal = self.transform(img_np)
        img_noise = self.transform(aug_np)

        return img_normal, img_noise, img_path.split('/')[-1]


# class MVTecDataset_train_seg(torch.utils.data.Dataset):
#     """
#     训练 seg/fusion 用：
#       输入：train/good 图
#       输出：img_aug(加了 simplex noise), mask(噪声区域 GT), name
#     """
#     def __init__(
#         self,
#         root,
#         transform,
#         size=256,
#         amp=0.2,                 # 噪声强度（你原来就是 0.2 * simplex）
#         octaves=6,
#         persistence=0.6,
#         frequency=32,
#         min_hw=10,
#         max_hw_div=8,            # max_hw = size//max_hw_div（你原来是 size//8）
#         mask_mode="rect",        # "rect"：整个ROI为1；"thr"：按噪声强度阈值生成更贴合的mask
#         mask_thr=0.02,           # mask_mode="thr" 时生效（建议和 amp 同量纲，0.01~0.05 之间试）
#     ):
#         self.img_path = root
#         self.transform = transform
#         self.size = int(size)
#
#         self.amp = float(amp)
#         self.octaves = int(octaves)
#         self.persistence = float(persistence)
#         self.frequency = float(frequency)
#
#         self.min_hw = int(min_hw)
#         self.max_hw = max( self.min_hw + 1, int(self.size // max_hw_div) )
#
#         self.mask_mode = str(mask_mode)
#         self.mask_thr = float(mask_thr)
#
#         self.simplexNoise = Simplex_CLASS()
#         self.img_paths = glob.glob(os.path.join(self.img_path, "good") + "/*.jpg")
#
#     def __len__(self):
#         return len(self.img_paths)
#
#     def __getitem__(self, idx):
#         img_path = self.img_paths[idx]
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (self.size, self.size))
#         img = img.astype(np.float32) / 255.0  # [H,W,3] in [0,1]
#
#         # --- sample rectangle ROI ---
#         h_noise = np.random.randint(self.min_hw, self.max_hw)
#         w_noise = np.random.randint(self.min_hw, self.max_hw)
#         start_h = np.random.randint(0, self.size - h_noise)
#         start_w = np.random.randint(0, self.size - w_noise)
#
#         # --- generate simplex patch ---
#         noise = self.simplexNoise.rand_3d_octaves(
#             (3, h_noise, w_noise),
#             self.octaves,
#             self.persistence,
#             self.frequency
#         )  # shape [3,h,w]
#         patch = self.amp * noise.transpose(1, 2, 0)  # [h,w,3]
#
#         aug = img.copy()
#         roi = aug[start_h:start_h + h_noise, start_w:start_w + w_noise, :]
#         roi = np.clip(roi + patch, 0.0, 1.0)
#         aug[start_h:start_h + h_noise, start_w:start_w + w_noise, :] = roi
#
#         # --- build GT mask (0/1) ---
#         mask = np.zeros((self.size, self.size), dtype=np.float32)
#
#         if self.mask_mode == "rect":
#             mask[start_h:start_h + h_noise, start_w:start_w + w_noise] = 1.0
#         elif self.mask_mode == "thr":
#             # 更贴合噪声强度：只把“噪声幅度较大”的像素标成1（避免整块矩形都算异常）
#             mag = np.mean(np.abs(patch), axis=2)  # [h,w]
#             m_roi = (mag > self.mask_thr).astype(np.float32)
#             mask[start_h:start_h + h_noise, start_w:start_w + w_noise] = m_roi
#         else:
#             raise ValueError(f"Unknown mask_mode={self.mask_mode}, use 'rect' or 'thr'.")
#
#         img_aug = self.transform(aug)                         # [3,H,W]
#         mask_t  = torch.from_numpy(mask).unsqueeze(0)         # [1,H,W]
#
#         return img_aug, mask_t, img_path.split("/")[-1]

class MVTecDataset_train_seg(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.img_path = root
        self.transform = transform
        # self.cutpaste = CutPasteMix(
        #     area_ratio=[0.05, 0.30],
        #     aspect_ratio=0.3,
        #     width=[2, 16],
        #     height=[10, 25],
        #     rotation=[-45, 45],
        # )
        self.cutpaste = PVCutPasteMix()
        self.img_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.jpg")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        img_pil = Image.fromarray(img)

        # CutPaste：拿到 aug + mask
        aug_pil, mask_pil = self.cutpaste(img_pil)

        aug_np = np.array(aug_pil).astype(np.float32) / 255.0
        img_aug = self.transform(aug_np)

        # ✅ mask 作为 GT：二值 0/1（>0 就是 1）
        mask_arr = np.array(mask_pil)                     # 0~255（scar 可能连续）
        mask_bin = (mask_arr > 0).astype(np.float32)      # 0/1
        mask = torch.from_numpy(mask_bin).unsqueeze(0)    # [1,256,256]

        return img_aug, mask, img_path.split('/')[-1]

class MVTecDataset_test(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform):
        self.img_path = os.path.join(root, 'test')
        self.gt_path = os.path.join(root, 'ground_truth')
        self.simplexNoise = Simplex_CLASS()
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.jpg")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))  # ✅ uint8 resize（和 train 一致）
        img = img.astype(np.float32) / 255.0  # ✅ 再转 float32 + 归一化

        ## Normal
        img = self.transform(img)
        
        if gt == 0:
            gt = torch.zeros([1, img.shape[-1], img.shape[-1]])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = (gt > 0.5).float()
        assert img.shape[1:] == gt.shape[1:], "image.size != gt.size !!!"

        return (img, gt, label, img_type, img_path.split('/')[-1])



