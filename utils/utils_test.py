import torch
from torch.nn import functional as F
import cv2
import numpy as np
from numpy import ndarray
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, auc
from skimage import measure
from statistics import mean
from scipy.ndimage import gaussian_filter
import warnings

from model.pixel_fusion_gate import cal_A_tensor

warnings.filterwarnings('ignore')
def cal_anomaly_map(fs_list, ft_list, out_size=256, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def minmax01(x, eps=1e-12):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + eps)

def compute_pro_curve_single(masks, amaps, num_th=200, fpr_limit=0.3, eps=1e-12, norm_mode="max"):

    d = {'pro': [], 'fpr': [], 'threshold': []}
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / num_th if max_th > min_th else 1.0

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for bin_map, mask in zip(binary_amaps, masks):
            labeled = measure.label(mask)
            for region in measure.regionprops(labeled):
                coords = region.coords
                tp = bin_map[coords[:, 0], coords[:, 1]].sum()
                pros.append(tp / (region.area + eps))
        pro = mean(pros) if pros else 0.0

        inv = 1 - masks
        fp = np.logical_and(inv, binary_amaps).sum()
        fpr = fp / (inv.sum() + eps)

        d['pro'].append(pro)
        d['fpr'].append(fpr)
        d['threshold'].append(th)

    df = pd.DataFrame(d)
    df = df[df["fpr"] < fpr_limit].copy()
    if len(df) == 0:
        df["fpr_norm"] = []
        return df, 0.0

    if norm_mode == "max":
        denom = df["fpr"].max() + eps
    else:
        denom = fpr_limit + eps

    df["fpr_norm"] = df["fpr"] / denom
    pro_auc = auc(df["fpr_norm"], df["pro"])
    return df, float(pro_auc)


def evaluation_multi_proj(encoder, proj, bn, decoder, dataloader, device, save_pro_csv=None, sp_csv=None):
    encoder.eval()
    proj.eval()
    bn.eval()
    decoder.eval()

    # ✅ 改：用数组块列表，而不是 extend 标量 list
    gt_list_px = []
    pr_list_px = []

    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []

    masks_list = []
    amaps_list = []

    sp_rows = []

    with torch.no_grad():
        for (img, gt, label, _, _) in dataloader:
            img = img.to(device)

            inputs = encoder(img)
            features = proj(inputs)
            outputs = decoder(bn(features))

            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=0)  # numpy [H,W]

            # binarize gt（gt 是 torch tensor）
            gt = (gt > 0.5).int()

            gt_sp = int(label.item() != 0)

            # ===== Pixel AUROC：正常+异常都统计 =====
            if label.item() == 0:
                # 正常图：没有mask就视为全0
                gt_np = np.zeros_like(anomaly_map, dtype=np.uint8)
            else:
                # 异常图：用真实mask
                gt_np = gt.squeeze(0).cpu().numpy().astype(np.uint8)

            gt_list_px.append(gt_np.ravel())
            pr_list_px.append(anomaly_map.astype(np.float32).ravel())

            # ===== AUPRO：仍只对异常图算 =====
            if label.item() != 0:
                aupro_list.append(compute_pro(gt_np.astype(int), anomaly_map[np.newaxis, :, :]))

                if save_pro_csv is not None:
                    masks_list.append(gt_np)
                    amaps_list.append(minmax01(anomaly_map)[np.newaxis, :, :])

            gt_list_sp.append(int(gt.max().item()))
            # pr_list_sp.append(float(np.max(anomaly_map)))
            flat = anomaly_map.reshape(-1).astype(np.float32)
            k = max(1, int(0.002 * flat.size))
            score_sp = float(np.partition(flat, -k)[-k:].mean())
            pr_list_sp.append(score_sp)
            sp_rows.append([gt_sp, score_sp])

    # ---- 计算指标 ----
    if len(gt_list_px) > 0:
        gt_all = np.concatenate(gt_list_px, axis=0)
        pr_all = np.concatenate(pr_list_px, axis=0)
        auroc_px = round(roc_auc_score(gt_all, pr_all), 4)
        ap_px = round(average_precision_score(gt_all, pr_all), 4)
    else:
        auroc_px = 0.0
        ap_px = 0.0

    auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
    aupro_px = round(float(np.mean(aupro_list)) if len(aupro_list) > 0 else 0.0, 4)

    if save_pro_csv is not None and len(masks_list) > 0:
        # 统一到同一个 x 轴网格：0~1
        fpr_grid = np.linspace(0.0, 1.0, 200)
        pro_sum = np.zeros_like(fpr_grid, dtype=np.float64)
        auc_list = []

        for mask, amap in zip(masks_list, amaps_list):  # 这里 amaps_list 你存的是 minmax01 后的 (H,W)
            df_i, auc_i = compute_pro_curve_single(mask, amap, num_th=200, fpr_limit=0.3, norm_mode="max")
            auc_list.append(auc_i)
            if len(df_i) == 0:
                continue
            # 插值到统一 fpr_grid（df_i["fpr_norm"] 一定要升序）
            xi = df_i["fpr_norm"].to_numpy()
            yi = df_i["pro"].to_numpy()
            order = np.argsort(xi)
            pro_i = np.interp(fpr_grid, xi[order], yi[order], left=yi[order][0], right=yi[order][-1])
            pro_sum += pro_i

        pro_mean = pro_sum / (len(auc_list) + 1e-12)
        pro_auc_mean = float(auc(fpr_grid, pro_mean))  # 这会更接近 eva 打印的均值

        df_out = pd.DataFrame({
            "fpr_norm": fpr_grid,
            "fpr": fpr_grid * 0.3,  # 画图用 0~0.3
            "pro": pro_mean
        })
        df_out.to_csv(save_pro_csv, index=False)
        print("Saved mean PRO curve:", save_pro_csv, "AUPRO(mean-curve)=", pro_auc_mean, "AUPRO(mean-per-image)=",
              float(np.mean(auc_list)))

    if sp_csv is not None and len(sp_rows) > 0:
        df_sp = pd.DataFrame(sp_rows, columns=["GroundTruth", "AnomalyScore"])
        df_sp.to_csv(sp_csv, index=False)
        print("Saved sp csv:", sp_csv)

    return auroc_px, ap_px, auroc_sp, aupro_px

def evaluation_multi_proj_fusion(
    encoder, proj, bn, decoder, fusion, dataloader, device,
    save_pro_csv=None, sp_csv=None, smooth_sigma=0
):
    encoder.eval()
    proj.eval()
    bn.eval()
    decoder.eval()
    fusion.eval()

    # ✅ Pixel AUROC：正常+异常都统计（用 ravel block）
    gt_list_px = []
    pr_list_px = []

    # ✅ Sample-level AUROC：全量统计
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []

    masks_list = []
    amaps_list = []

    sp_rows = []

    with torch.no_grad():
        for (img, gt, label, _, _) in dataloader:
            img = img.to(device)

            inputs = encoder(img)
            features = proj(inputs)
            outputs = decoder(bn(features))

            # ---- fusion anomaly map ----
            H, W = img.shape[-2], img.shape[-1]
            A = cal_A_tensor(inputs, outputs, out_hw=(H, W))      # torch [B,3,H,W] or [1,3,H,W]
            score, _ = fusion(A)                                 # torch [B,1,H,W]
            anomaly_map = score[0, 0].detach().cpu().numpy().astype(np.float32)

            if smooth_sigma is not None and smooth_sigma > 0:
                anomaly_map = gaussian_filter(anomaly_map, sigma=smooth_sigma)

            # binarize gt（gt 是 torch tensor）
            gt = (gt > 0.5).int()

            # sample-level GT：用 label 判正常/异常（与上面一致）
            gt_sp = int(label.item() != 0)

            # ===== Pixel AUROC：正常+异常都统计 =====
            if label.item() == 0:
                gt_np = np.zeros_like(anomaly_map, dtype=np.uint8)
            else:
                gt_np = gt.squeeze(0).cpu().numpy().astype(np.uint8)

            gt_list_px.append(gt_np.ravel())
            pr_list_px.append(anomaly_map.ravel().astype(np.float32))

            # ===== AUPRO：只对异常图算 =====
            if label.item() != 0:
                aupro_list.append(
                    compute_pro(gt_np.astype(int), anomaly_map[np.newaxis, :, :])
                )
                if save_pro_csv is not None:
                    masks_list.append(gt_np)
                    amaps_list.append(minmax01(anomaly_map)[np.newaxis, :, :])

            # ===== Sample-level score：top-0.2% 均值（与上面一致） =====
            flat = anomaly_map.reshape(-1).astype(np.float32)
            k = max(1, int(0.002 * flat.size))
            score_sp = float(np.partition(flat, -k)[-k:].mean())

            gt_list_sp.append(gt_sp)
            pr_list_sp.append(score_sp)
            sp_rows.append([gt_sp, score_sp])

            # gt_list_sp.append(int(gt.max().item()))
            # # 图像级分数：取 anomaly_map 的最大值
            # score_sp = float(np.max(anomaly_map))
            # pr_list_sp.append(score_sp)
            # sp_rows.append([gt_sp, score_sp])

    # ---- 计算指标 ----
    if len(gt_list_px) > 0:
        gt_all = np.concatenate(gt_list_px, axis=0)
        pr_all = np.concatenate(pr_list_px, axis=0)
        auroc_px = round(roc_auc_score(gt_all, pr_all), 4)
        ap_px = round(average_precision_score(gt_all, pr_all), 4)
    else:
        auroc_px = 0.0
        ap_px = 0.0

    auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
    aupro_px = round(float(np.mean(aupro_list)) if len(aupro_list) > 0 else 0.0, 4)

    # ---- 保存 mean PRO curve（可选，逻辑同上）----
    if save_pro_csv is not None and len(masks_list) > 0:
        fpr_grid = np.linspace(0.0, 1.0, 200)
        pro_sum = np.zeros_like(fpr_grid, dtype=np.float64)
        auc_list = []

        for mask, amap in zip(masks_list, amaps_list):
            df_i, auc_i = compute_pro_curve_single(mask, amap, num_th=200, fpr_limit=0.3, norm_mode="max")
            auc_list.append(auc_i)
            if len(df_i) == 0:
                continue

            xi = df_i["fpr_norm"].to_numpy()
            yi = df_i["pro"].to_numpy()
            order = np.argsort(xi)
            pro_i = np.interp(
                fpr_grid, xi[order], yi[order],
                left=yi[order][0], right=yi[order][-1]
            )
            pro_sum += pro_i

        pro_mean = pro_sum / (len(auc_list) + 1e-12)
        pro_auc_mean = float(auc(fpr_grid, pro_mean))

        df_out = pd.DataFrame({
            "fpr_norm": fpr_grid,
            "fpr": fpr_grid * 0.3,
            "pro": pro_mean
        })
        df_out.to_csv(save_pro_csv, index=False)
        print("Saved mean PRO curve:", save_pro_csv,
              "AUPRO(mean-curve)=", pro_auc_mean,
              "AUPRO(mean-per-image)=", float(np.mean(auc_list)))

    # ---- 保存 sample-level csv（可选）----
    if sp_csv is not None and len(sp_rows) > 0:
        df_sp = pd.DataFrame(sp_rows, columns=["GroundTruth", "AnomalyScore"])
        df_sp.to_csv(sp_csv, index=False)
        print("Saved sp csv:", sp_csv)

    return auroc_px, ap_px, auroc_sp, aupro_px


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

#     df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    d = {'pro':[], 'fpr':[],'threshold': []}
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

#         df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        d['pro'].append(mean(pros))
        d['fpr'].append(fpr)
        d['threshold'].append(th)
    df = pd.DataFrame(d)
    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc
