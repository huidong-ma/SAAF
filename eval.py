import torch
import torch.nn.functional as F
from torchvision import transforms
from models.saaf import SAAF
import warnings
import os
import sys
import math
import argparse
import time
import shutil
from PIL import Image
warnings.filterwarnings("ignore")
torch.set_num_threads(10)

def save_image(tensor, filename):
    img = transforms.ToPILImage()(tensor.squeeze(0).cpu())
    img.save(filename)

def save_metrics(filename, file, psnr, bitrate):
    with open(file, 'a') as f:
        f.write(f"{filename}\t{psnr:.5f}\t{bitrate:.5f}\n")

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--data", nargs='+', type=str, help="Path to dataset")
    parser.add_argument("--save_path", default=None, type=str, help="Path to save")
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args(argv)
    return args

def evaluate(net, device, data_dirs, cuda=False, real=False, p=128, save_path=None):
    print(' Evaluate '.center(40, '='))
    print(
        "Dataset".center(10) +
        "Count".center(10) +
        "PSNR".center(10) +
        "Bpp".center(10)
    )

    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    for data_dir in data_dirs:
        img_list = []
        if os.path.isfile(data_dir):
            img_list.append(os.path.basename(data_dir))
            data_dir = os.path.dirname(data_dir)
            dataset = os.path.basename(data_dir) if data_dir else "single_image"
        else:
            for file in os.listdir(data_dir):
                if file[-3:] in ["jpg", "png", "peg"]:
                    img_list.append(file)
            dataset = os.path.basename(data_dir)

        count = len(img_list)
        PSNR = 0
        Bit_rate = 0

        if real:
            net.update()
            for img_name in img_list:
                img_path = os.path.join(data_dir, img_name)
                img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
                x = img.unsqueeze(0)
                x_padded, padding = pad(x, p)

                with torch.no_grad():
                    out_enc = net.compress(x_padded)
                    out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
                    out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
                    num_pixels = x.size(0) * x.size(2) * x.size(3)
                    Bit_rate += sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                    PSNR += compute_psnr(x, out_dec["x_hat"])
        else:
            if save_path is not None:
                if os.path.exists(os.path.join(save_path, dataset)):
                    shutil.rmtree(os.path.join(save_path, dataset))
                os.makedirs(os.path.join(save_path, dataset))
                with open(os.path.join(save_path, dataset, "metrics.txt"), 'w') as f:
                    f.write(f"Image\tPSNR(dB)\tBpp\n")

            for img_name in img_list:
                img_path = os.path.join(data_dir, img_name)
                img = Image.open(img_path).convert('RGB')
                x = transforms.ToTensor()(img).unsqueeze(0).to(device)
                x_padded, padding = pad(x, p)

                with torch.no_grad():
                    out_net = net.forward(x_padded)
                    out_net['x_hat'].clamp_(0, 1)
                    out_net["x_hat"] = crop(out_net["x_hat"], padding)
                    PSNR += compute_psnr(x, out_net["x_hat"])
                    Bit_rate += compute_bpp(out_net)
                    if save_path is not None:
                        save_metrics(
                            img_name.split('.')[0],
                            os.path.join(save_path, dataset, "metrics.txt"),
                            compute_psnr(x, out_net["x_hat"]),
                            compute_bpp(out_net)
                        )
                        save_image(out_net["x_hat"], os.path.join(save_path, dataset, f"decoded_{img_name}"))

        PSNR = PSNR / count
        Bit_rate = Bit_rate / count

        print(
            f"{dataset}".center(10) +
            f"{count}".center(10) +
            f"{PSNR:.5f}".center(10) +
            f"{Bit_rate:.5f}".center(10)
        )
    print(''.center(40, '='))
    print()


def main(argv):
    torch.backends.cudnn.enabled = False
    args = parse_args(argv)
    p = 128

    if args.cuda:
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'

    net = SAAF()
    net = net.to(device)
    net.eval()

    if args.checkpoint:
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        else:
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        net.load_state_dict(state_dict)

    evaluate(net, device, args.data, args.cuda, args.real, p, args.save_path)


if __name__ == "__main__":
    main(sys.argv[1:])
