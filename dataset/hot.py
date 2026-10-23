import os
import cv2
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
from utils.utils_train import MultiProjectionLayer
from dataset.dataset import get_data_transforms
from model.pixel_fusion_gate import PixelFusionGate, cal_A_tensor


def minmax01(x, eps=1e-12):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + eps)


@torch.no_grad()
def generate_heatmap_and_mask(
    image_path,
    save_dir="./vis_result",
    checkpoint_folder="./RD++_checkpoint_result_pvcut",
    fusion_ckpt="./fusion_ckpts_sigmoid/pvel_best_fusion.pth",
    class_name="pvel",
    image_size=512,
    smooth_sigma=0,
    threshold=0.5,   # 预测mask阈值，可自己调
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)

    # ========== 1. 读取原图 ==========
    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    orig_h, orig_w = orig_bgr.shape[:2]

    data_transform, _ = get_data_transforms(image_size, image_size)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32) / 255.0
    img_tensor = data_transform(img).unsqueeze(0).to(device)

    # ========== 2. 加载模型 ==========
    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False)
    proj = MultiProjectionLayer(base=64)
    fusion = PixelFusionGate(
        hidden=16,
        norm_type="bn",
        use_zscore=True,
        clamp_val=5.0,
        clamp_A=True
    )

    ckpt_path = os.path.join(checkpoint_folder, class_name, f"wres50_{class_name}.pth")
    ckp = torch.load(ckpt_path, map_location="cpu")
    proj.load_state_dict(ckp["proj"])
    bn.load_state_dict(ckp["bn"])
    decoder.load_state_dict(ckp["decoder"])

    fckpt = torch.load(fusion_ckpt, map_location="cpu")
    fusion.load_state_dict(fckpt["fusion"], strict=True)

    encoder = encoder.to(device).eval()
    proj = proj.to(device).eval()
    bn = bn.to(device).eval()
    decoder = decoder.to(device).eval()
    fusion = fusion.to(device).eval()

    # ========== 3. 前向推理 ==========
    inputs = encoder(img_tensor)
    features = proj(inputs)
    outputs = decoder(bn(features))

    H, W = img_tensor.shape[-2], img_tensor.shape[-1]
    A = cal_A_tensor(inputs, outputs, out_hw=(H, W))   # [1,3,H,W]
    logits, _ = fusion(A)                              # [1,1,H,W]

    anomaly_map = logits[0, 0].detach().cpu().numpy().astype(np.float32)

    if smooth_sigma > 0:
        anomaly_map = gaussian_filter(anomaly_map, sigma=smooth_sigma)

    # ========== 4. resize回原图大小 ==========
    anomaly_map = minmax01(anomaly_map)
    anomaly_map = cv2.resize(anomaly_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # ========== 5. 生成蓝底红热点热图 ==========
    # 5. 热图（蓝底红热点）
    heat_u8 = np.uint8(np.power(anomaly_map, 1.8) * 255.0)
    heatmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    # ========== 6. 叠加图 ==========
    overlay = cv2.addWeighted(orig_bgr, 0.55, heatmap, 0.45, 0)

    # ========== 7. 预测掩膜 ==========
    pred_mask = (anomaly_map >= threshold).astype(np.uint8) * 255

    # 如果你想让mask更干净，可以加这两句形态学后处理
    kernel = np.ones((3, 3), np.uint8)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)

    # ========== 8. 保存 ==========
    name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(save_dir, f"{name}_heatmap.png"), heatmap)
    cv2.imwrite(os.path.join(save_dir, f"{name}_overlay.png"), overlay)
    cv2.imwrite(os.path.join(save_dir, f"{name}_mask.png"), pred_mask)

    print(f"[OK] saved heatmap: {os.path.join(save_dir, f'{name}_heatmap.png')}")
    print(f"[OK] saved overlay: {os.path.join(save_dir, f'{name}_overlay.png')}")
    print(f"[OK] saved mask:    {os.path.join(save_dir, f'{name}_mask.png')}")


if __name__ == "__main__":
    generate_heatmap_and_mask(
        image_path="./0160.jpg",
        save_dir="./vis_result",
        checkpoint_folder="./RD++_checkpoint_result_pvcut",
        fusion_ckpt="./fusion_ckpts_sigmoid/pvel_best_fusion.pth",
        class_name="pvel",
        image_size=512,
        smooth_sigma=0,
        threshold=0.55
    )

# import os
# import cv2
# import torch
# import numpy as np
# from PIL import Image
# from scipy.ndimage import gaussian_filter
#
# from model.resnet import wide_resnet50_2
# from model.de_resnet import de_wide_resnet50_2
# from utils.utils_train import MultiProjectionLayer
# from dataset.dataset import get_data_transforms
# from utils.utils_test import cal_anomaly_map
#
#
# def minmax01(x, eps=1e-12):
#     x = x.astype(np.float32)
#     mn, mx = float(x.min()), float(x.max())
#     return (x - mn) / (mx - mn + eps)
#
#
# @torch.no_grad()
# def generate_heatmap_and_mask_no_fusion(
#     image_path,
#     save_dir="./vis_result",
#     checkpoint_folder="./RD++_checkpoint_result_pvcut",
#     class_name="pvel",
#     image_size=512,
#     smooth_sigma=0,
#     threshold=0.5,
#     invert=False,   # 如果热图方向反了就改成 True
# ):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     os.makedirs(save_dir, exist_ok=True)
#
#     # 1. 读原图
#     orig_bgr = cv2.imread(image_path)
#     if orig_bgr is None:
#         raise FileNotFoundError(f"Cannot read image: {image_path}")
#     orig_h, orig_w = orig_bgr.shape[:2]
#
#     data_transform, _ = get_data_transforms(image_size, image_size)
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (512, 512))
#     img = img.astype(np.float32) / 255.0
#     img_tensor = data_transform(img).unsqueeze(0).to(device)
#
#     # 2. 加载模型
#     encoder, bn = wide_resnet50_2(pretrained=True)
#     decoder = de_wide_resnet50_2(pretrained=False)
#     proj = MultiProjectionLayer(base=64)
#
#     ckpt_path = os.path.join(checkpoint_folder, class_name, f"wres50_{class_name}.pth")
#     ckp = torch.load(ckpt_path, map_location="cpu")
#     proj.load_state_dict(ckp["proj"])
#     bn.load_state_dict(ckp["bn"])
#     decoder.load_state_dict(ckp["decoder"])
#
#     encoder = encoder.to(device).eval()
#     proj = proj.to(device).eval()
#     bn = bn.to(device).eval()
#     decoder = decoder.to(device).eval()
#
#     # 3. 前向推理
#     inputs = encoder(img_tensor)
#     features = proj(inputs)
#     outputs = decoder(bn(features))
#
#     anomaly_map, _ = cal_anomaly_map(inputs, outputs, out_size=img_tensor.shape[-1], amap_mode='a')
#     anomaly_map = anomaly_map.astype(np.float32)
#     # np.save("hot_raw.npy", anomaly_map )
#     # np.save("hot_input.npy", img_tensor[0].detach().cpu().numpy())
#     if smooth_sigma > 0:
#         anomaly_map = gaussian_filter(anomaly_map, sigma=smooth_sigma)
#
#     # 4. 归一化 + 可选反转
#     anomaly_map = minmax01(anomaly_map)
#     if invert:
#         anomaly_map = 1.0 - anomaly_map
#
#     # resize 回原图大小
#     anomaly_map = cv2.resize(anomaly_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
#
#     # 5. 热图（蓝底红热点）
#     heat_u8 = np.uint8(np.power(anomaly_map, 1.5) * 255.0)
#     heatmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
#
#     # 6. 叠加图
#     overlay = cv2.addWeighted(orig_bgr, 0.55, heatmap, 0.45, 0)
#
#     # 7. 预测 mask
#     pred_mask = (anomaly_map >= threshold).astype(np.uint8) * 255
#
#     # 可选形态学后处理
#     kernel = np.ones((3, 3), np.uint8)
#     pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
#     pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
#
#     # 8. 保存
#     name = os.path.splitext(os.path.basename(image_path))[0]
#     cv2.imwrite(os.path.join(save_dir, f"{name}_heatmap.png"), heatmap)
#     cv2.imwrite(os.path.join(save_dir, f"{name}_overlay.png"), overlay)
#     cv2.imwrite(os.path.join(save_dir, f"{name}_mask.png"), pred_mask)
#
#     print(f"[OK] saved heatmap: {os.path.join(save_dir, f'{name}_heatmap.png')}")
#     print(f"[OK] saved overlay: {os.path.join(save_dir, f'{name}_overlay.png')}")
#     print(f"[OK] saved mask:    {os.path.join(save_dir, f'{name}_mask.png')}")
#
#
# if __name__ == "__main__":
#     generate_heatmap_and_mask_no_fusion(
#         image_path="./0059.jpg",
#         save_dir="./vis_result",
#         checkpoint_folder="./RD++_checkpoint_result_pvcut",
#         class_name="pvel",
#         image_size=512,
#         smooth_sigma=0,
#         threshold=0.25,
#         invert=False
#     )