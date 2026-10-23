import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["KMP_WARNINGS"] = "0"
import argparse


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ====== RD++ backbone（和 main.py / train_seg.py 保持一致）======
from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
from utils.utils_train import MultiProjectionLayer

# ====== 伪异常+mask 训练集；真实 test 集 ======
from dataset.dataset import MVTecDataset_test, get_data_transforms, MVTecDataset_train_seg

# ====== Pixel-wise fusion head ======
from model.pixel_fusion_gate import PixelFusionGate, cal_A_tensor

# ====== test 指标（沿用你现有 compute_pro）======
from utils.utils_test import compute_pro, evaluation_multi_proj_fusion
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter


# ------------------------------
# Dice loss (binary)
# ------------------------------
def dice_loss_from_logits(score, target, eps=1e-6):
    """
    logits: [B,1,H,W]
    target: [B,1,H,W] float {0,1}
    """
    num = 2.0 * (score * target).sum(dim=(2, 3)) + eps
    den = (score + target).sum(dim=(2, 3)) + eps
    dice = 1.0 - (num / den)
    return dice.mean()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./mvtec")
    parser.add_argument("--category", type=str, default="pvel")

    # Stage-1 RD++ checkpoint (contains proj/bn/decoder)
    parser.add_argument("--rd_ckpt", type=str, default="./RD++_checkpoint_result_pvcut/pvel/wres50_pvel.pth")

    # Stage-2 train
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)

    # fusion head config
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--norm", type=str, default="bn")   # bn / gn
    parser.add_argument("--use_zscore", type=str, default="true")  # true/false

    # loss weights
    parser.add_argument("--w_bce", type=float, default=1)
    parser.add_argument("--w_dice", type=float, default=1)

    # eval smooth
    parser.add_argument("--smooth_sigma", type=float, default=0)

    # save
    parser.add_argument("--save_dir", type=str, default="./fusion_ckpts_sigmoid")
    args = parser.parse_args()

    use_zscore = str(args.use_zscore).lower() in ["1", "true", "yes", "y"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------
    # datasets
    # ------------------------------
    train_root = os.path.join(args.data_path, args.category, "train")
    test_root  = os.path.join(args.data_path, args.category)

    data_transform, gt_transform = get_data_transforms(512, 512)

    train_set = MVTecDataset_train_seg(root=train_root, transform=data_transform)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )

    test_set = MVTecDataset_test(root=test_root, transform=data_transform, gt_transform=gt_transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # ------------------------------
    # build RD++ modules
    # ------------------------------
    encoder, bn = wide_resnet50_2(pretrained=True)
    proj = MultiProjectionLayer(base=64)
    decoder = de_wide_resnet50_2(pretrained=False)

    encoder = encoder.to(device)
    bn = bn.to(device)
    proj = proj.to(device)
    decoder = decoder.to(device)

    # load stage-1 ckpt (proj/bn/decoder)
    ckpt = torch.load(args.rd_ckpt, map_location="cpu")
    proj.load_state_dict(ckpt["proj"])
    bn.load_state_dict(ckpt["bn"])
    decoder.load_state_dict(ckpt["decoder"])

    # freeze all RD++ parts
    for m in [encoder, proj, bn, decoder]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    # ------------------------------
    # fusion head (trainable)
    # ------------------------------
    fusion = PixelFusionGate(
        hidden=args.hidden,
        norm_type=args.norm,
        use_zscore=use_zscore
    ).to(device)

    optimizer = torch.optim.Adam(fusion.parameters(), lr=args.lr)

    best_score = -1.0

    # ------------------------------
    # train loop
    # ------------------------------
    for epoch in range(1, args.epochs + 1):
        fusion.train()
        total_loss = 0.0

        for (img_aug, mask, _) in train_loader:
            img_aug = img_aug.to(device)   # [B,3,H,W]
            mask = mask.to(device)         # [B,1,H,W]

            # 1) frozen RD++ forward -> A
            with torch.no_grad():
                fs = encoder(img_aug)
                feat = proj(fs)
                ft = decoder(bn(feat))
                A = cal_A_tensor(fs, ft, out_hw=(img_aug.shape[-2], img_aug.shape[-1]))  # [B,3,H,W]

            # 2) fusion forward -> logits
            score, _ = fusion(A)  # [B,1,H,W]

            # 3) losses (BCE + Dice)
            loss_bce = F.binary_cross_entropy(score, mask)
            loss_dice = dice_loss_from_logits(score, mask)
            loss = args.w_bce * loss_bce + args.w_dice * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_loader))

        # ------------------------------
        # eval each epoch on test
        # ------------------------------
        auroc_px, _, auroc_sp, pro = evaluation_multi_proj_fusion(
            encoder, proj, bn, decoder, fusion, test_loader, device,
            save_pro_csv=None, smooth_sigma=args.smooth_sigma
        )

        score = (auroc_px + auroc_sp + pro) / 3.0
        print(
            f"[Epoch {epoch:03d}] loss={avg_loss:.4f} | "
            f"AUROC_px={auroc_px:.4f} | AUROC_sp={auroc_sp:.4f} | PRO={pro:.4f} | "
            f"score(px+pro)/2={score:.4f}"
        )

        if score > best_score:
            best_score = score
            save_path = os.path.join(args.save_dir, f"{args.category}_best_fusion.pth")
            torch.save({
                "fusion": fusion.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "auroc_px": auroc_px,
                "auroc_sp": auroc_sp,
                "pro": pro,
                "args": vars(args),
            }, save_path)
            print(f"  ✔ saved best fusion to: {save_path}")


if __name__ == "__main__":
    main()
