#!/usr/bin/env python3
"""
Pix2Pix-style training with ResNet-based generator (ResNet-9blocks).
Adds inference/evaluation using fixed input images at checkpoint frequency.

Data layout expected (you provided):
 - real images (targets): /path/to/real_samples/{char}.png
 - skeletons (inputs): /path/to/skeleton/{char}_skeleton_dilated_k3.png  (preferred)
   fallback patterns: {char}_skeleton_dilated.png, {char}_skeleton.png

Usage example:
python train_pix2pix_resnet.py \
  --real_dir /hpc2hdd/home/mli861/cmaaproject/data/real_samples \
  --sk_dir  /hpc2hdd/home/mli861/cmaaproject/data/skeleton \
  --out_dir /hpc2hdd/home/mli861/cmaaproject/pix2pix_resnet_run1 \
  --inference_dir /hpc2hdd/home/mli861/cmaaproject/inference_skeleton \
  --img_size 512 --batch 8 --max_steps 100000 --amp

"""
import os
import argparse
from pathlib import Path
import random
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from generate_data import make_skeleton_from_bitmap, dilate_mask_to_black_strokes
# lpips
try:
    import lpips
except Exception:
    lpips = None

# ---------------------------
# Network: ResNet Generator (Johnson style) + PatchGAN Discriminator (保持不变)
# ---------------------------
def conv_block(in_ch, out_ch, kernel_size=7, stride=1, padding=0, norm=True, activation='relu'):
    layers = []
    layers.append(nn.ReflectionPad2d(padding))
    layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride))
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'lrelu':
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            norm_layer(dim, affine=True),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
            norm_layer(dim, affine=True)
        )
    def forward(self, x):
        return x + self.block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_blocks=9):
        super().__init__()
        # c7s1-64
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                 nn.InstanceNorm2d(ngf, affine=True),
                 nn.ReLU(inplace=True)]
        # d128, d256
        curr_dim = ngf
        for i in range(2):
            model += [nn.Conv2d(curr_dim, curr_dim*2, kernel_size=3, stride=2, padding=1, bias=False),
                      nn.InstanceNorm2d(curr_dim*2, affine=True),
                      nn.ReLU(inplace=True)]
            curr_dim *= 2
        # res blocks
        for i in range(n_blocks):
            model += [ResnetBlock(curr_dim)]
        # u128, u64
        for i in range(2):
            model += [nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                      nn.InstanceNorm2d(curr_dim//2, affine=True),
                      nn.ReLU(inplace=True)]
            curr_dim //= 2
        # final
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(curr_dim, out_channels, kernel_size=7, stride=1, padding=0),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6, ndf=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ndf*nf_mult, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers += [
            nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf*nf_mult, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        layers += [nn.Conv2d(ndf*nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ---------------------------
# Dataset (保持不变)
# ---------------------------
class PairedFontDataset(Dataset):
    def __init__(self, real_dir, sk_dir, img_size=512, transform=None, prefer_dilated=True):
        self.real_dir = Path(real_dir)
        self.sk_dir = Path(sk_dir)
        self.real_paths = sorted([p for p in self.real_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
        self.img_size = img_size
        self.prefer_dilated = prefer_dilated

        if transform is None:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),  # [0,1]
                T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))  # -> [-1,1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.real_paths)

    def _locate_skeleton(self, base_filename):
        dilated_name_k3 = f"{base_filename}_dilated_k3.png"
        p_k3 = self.sk_dir / dilated_name_k3
        if p_k3.exists():
            return p_k3
        return None

    def __getitem__(self, idx):
        real_p = self.real_paths[idx]
        base_filename = real_p.stem 
        sk_p = self._locate_skeleton(base_filename)
        
        if sk_p is None:
            raise FileNotFoundError(f"Cannot find skeleton for base file '{base_filename}' in {self.sk_dir}. Expected: {base_filename}_dilated_k3.png")
            
        real = Image.open(real_p).convert('L').convert('RGB')
        sk = Image.open(sk_p).convert('L').convert('RGB')
        
        real = self.transform(real)
        sk = self.transform(sk)
        
        return {'A': sk, 'B': real, 'name': base_filename}

# ---------------------------
# NEW: Inference Data Loader
# ---------------------------
def load_inference_data(inference_dir, img_size):
    """
    加载固定的推理输入图片，对其进行 skeletonize 和 dilate 处理，
    并返回 (tensor, filenames)。
    """
    inference_dir = Path(inference_dir)
    # 查找所有 .png, .jpg, .jpeg 文件
    inference_paths = sorted([p for p in inference_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    
    if not inference_paths:
        raise FileNotFoundError(f"Inference directory {inference_dir} is empty or contains no valid images.")
        
    input_tensors = []
    filenames = []
    
    for p in inference_paths:
        # 1. Load grayscale and Resize
        img = Image.open(p).convert("L").resize((img_size, img_size))
        img_arr = np.array(img) # H, W, 8-bit grayscale
        
        # 2. Skeletonize and Dilate (核心修正)
        sk = make_skeleton_from_bitmap(img_arr)
        sk = dilate_mask_to_black_strokes(sk, kernel_size=3) # 假设 k=5
        
        # 3. Convert skeleton (H,W) -> (H,W,3) RGB
        sk_rgb = np.stack([sk, sk, sk], axis=-1)

        # 4. Normalize to [-1,1]
        arr = sk_rgb.astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0  # [0,1] → [-1,1]

        # 5. HWC → CHW and Add batch dim
        arr = arr.transpose(2, 0, 1)
        tensor = torch.tensor(arr).unsqueeze(0)
        
        input_tensors.append(tensor)
        filenames.append(p.stem)
        
    # 将所有 tensor 堆叠成一个 batch
    return torch.cat(input_tensors, dim=0), filenames


# ---------------------------
# NEW: Inference Runner
# ---------------------------
def run_inference_and_save(G, inference_A, filenames, out_dir, global_step, device, amp_enabled, pbar):
    """
    运行推理并在 samples 文件夹中保存生成的图片。
    """
    G.eval()
    step_samples_dir = out_dir / f"step_{global_step:06d}"
    step_samples_dir.mkdir(parents=True, exist_ok=True) 
    with torch.no_grad():
        inference_A = inference_A.to(device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            fake_B = G(inference_A)
        
        # 将 tensor 转换回 [0, 1] 范围的 CPU 图像
        images = (fake_B.cpu() + 1.0) / 2.0 

        # 遍历生成的图片并单独保存
        for i in range(images.size(0)):
            filename = filenames[i]
            # output_dir/samples/step_005000_imageName.png
            out_path = step_samples_dir / f"{filename}.png"
            # 使用 vutils.save_image 保存单张图片
            vutils.save_image(images[i], str(out_path), normalize=False)

        pbar.write(f"Saved inference results for step {global_step} to {step_samples_dir}")
    G.train()
    
# ---------------------------
# Training loop
# ---------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # 检查 lpips 是否可用
    if not args.no_lpips and lpips is None:
        raise RuntimeError("lpips not installed. pip install lpips or run with --no-lpips")

    # dataset & loader
    dataset = PairedFontDataset(args.real_dir, args.sk_dir, img_size=args.img_size, prefer_dilated=True)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # models
    G = ResnetGenerator(in_channels=3, out_channels=3, ngf=args.ngf, n_blocks=args.n_blocks).to(device)
    D = PatchDiscriminator(in_channels=6, ndf=args.ndf).to(device)

    # init
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    G.apply(weights_init); D.apply(weights_init)

    # losses & optimizers
    criterion_GAN = nn.MSELoss().to(device)  # LSGAN
    criterion_L1 = nn.L1Loss().to(device)
    lpips_model = lpips.LPIPS(net='vgg').to(device) if not args.no_lpips else None

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    scaler_G = torch.cuda.amp.GradScaler(enabled=args.amp)
    scaler_D = torch.cuda.amp.GradScaler(enabled=args.amp)

    # checkpoint dirs
    out_dir = Path(args.out_dir)
    samples_dir = out_dir / "samples"
    ckpt_dir = out_dir / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # NEW: 加载固定的推理输入数据
    if args.inference_dir:
        try:
            inference_A, inference_filenames = load_inference_data(args.inference_dir, args.img_size)
            print(f"Loaded {len(inference_filenames)} fixed inference images from {args.inference_dir}")
        except FileNotFoundError as e:
            print(f"Error loading inference data: {e}")
            inference_A, inference_filenames = None, None
    else:
        inference_A, inference_filenames = None, None

    global_step = 0
    pbar = tqdm(total=args.max_steps)
    
    # 首次运行推理 (Step 0)
    if inference_A is not None:
        run_inference_and_save(G, inference_A, inference_filenames, samples_dir, global_step, device, args.amp, pbar)
        
    while global_step < args.max_steps:
        for batch in loader:
            if global_step >= args.max_steps:
                break
            A = batch['A'].to(device)  # skeleton (input)
            B = batch['B'].to(device)  # real target

            # Train D
            D.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                fake_B = G(A)
                real_pair = torch.cat([A, B], dim=1)
                fake_pair = torch.cat([A, fake_B.detach()], dim=1)
                pred_real = D(real_pair)
                pred_fake = D(fake_pair)
                real_label = torch.ones_like(pred_real, device=device)
                fake_label = torch.zeros_like(pred_fake, device=device)
                loss_D = 0.5 * (criterion_GAN(pred_real, real_label) + criterion_GAN(pred_fake, fake_label))
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

            # Train G
            G.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                fake_B = G(A)
                pred_fake_for_G = D(torch.cat([A, fake_B], dim=1))
                loss_G_GAN = criterion_GAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G, device=device))
                loss_L1 = criterion_L1(fake_B, B) * args.lambda_l1
                if lpips_model is not None:
                    # lpips expects inputs in [-1,1] or [0,1]; ensure [0,1]
                    lpips_val = lpips_model((fake_B + 1.0) / 2.0, (B + 1.0) / 2.0).mean()
                    loss_lpips = lpips_val * args.lambda_lpips
                else:
                    loss_lpips = torch.tensor(0.0, device=device)
                loss_G = loss_G_GAN + loss_L1 + loss_lpips
            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            # logging
            if global_step % args.log_freq == 0:
                tqdm.write(f"step {global_step} | loss_D {loss_D.item():.4f} | loss_G {loss_G.item():.4f} | L1 {loss_L1.item():.4f} | LPIPS {loss_lpips.item() if isinstance(loss_lpips, torch.Tensor) else loss_lpips:.4f}")

            # save checkpoint and run inference
            if global_step % args.ckpt_freq == 0 and global_step > 0:
                # 1. Save Checkpoint
                ckpt = {
                    'G': G.state_dict(),
                    'D': D.state_dict(),
                    'opt_G': opt_G.state_dict(),
                    'opt_D': opt_D.state_dict(),
                    'step': global_step
                }
                torch.save(ckpt, ckpt_dir / f"ckpt_{global_step:06d}.pt")
                tqdm.write(f"Saved checkpoint step {global_step}")
                
                # 2. Run Inference (Evaluation)
                if inference_A is not None:
                    run_inference_and_save(G, inference_A, inference_filenames, samples_dir, global_step, device, args.amp, pbar)
                
            # NOTE: 移除原有的 args.sample_freq 逻辑
            # if global_step % args.sample_freq == 0: 
            #   ... original save_sample logic removed

            global_step += 1
            pbar.update(1)

    pbar.close()
    # final save
    torch.save({'G': G.state_dict(), 'D': D.state_dict()}, ckpt_dir / "final.pt")
    print("Training finished. Final model saved to", ckpt_dir / "final.pt")

# ---------------------------
# CLI (添加 --inference_dir 参数)
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dir', type=str, required=True)
    parser.add_argument('--sk_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='./pix2pix_resnet_out')
    parser.add_argument('--inference_dir', type=str, default='./inference_skeleton/', help='Directory containing fixed input images for inference (e.g., inference_skeleton/)') # 新增参数
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--lambda_l1', type=float, default=100.0)
    parser.add_argument('--lambda_lpips', type=float, default=0.8)
    parser.add_argument('--sample_freq', type=int, default=1000) # 保持不变，但功能被替代
    parser.add_argument('--ckpt_freq', type=int, default=5000)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--amp', action='store_true', help='enable mixed precision')
    parser.add_argument('--no-lpips', action='store_true', help='disable lpips loss')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ngf', type=int, default=64, help='generator base filters')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator base filters')
    parser.add_argument('--n_blocks', type=int, default=9, help='number of resnet blocks')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print("Args:", args)
    train(args)