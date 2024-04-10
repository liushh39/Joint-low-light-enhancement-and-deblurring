import os
import cv2
import argparse
import glob
import torch
from basicsr.utils import imwrite, img2tensor, tensor2img, scandir
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
import numpy as np


def check_image_size(x, down_factor):
    _, _, h, w = x.size()
    mod_pad_h = (down_factor - h % down_factor) % down_factor
    mod_pad_w = (down_factor - w % down_factor) % down_factor
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_path', type=str, default='.\\realblur_dataset_test')
    parser.add_argument('--result_path', type=str, default='.\\result')

    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    if args.test_path.endswith('/'):  # solve when path ends with /
        args.test_path = args.test_path[:-1]
    if args.result_path.endswith('/'):  # solve when path ends with /
        args.result_path = args.result_path[:-1]
    result_root = f'{args.result_path}/{os.path.basename(args.test_path)}'

    # ------------------ set up network -------------------
    down_factor = 8 # check_image_size
    net = ARCH_REGISTRY.get('Net')(channels=[32, 64, 64, 64], connection=False).to(device)

    checkpoint = torch.load('.\weights1\\net.pth')['params']
    net.load_state_dict(checkpoint)
    net.eval()

    # -------------------- start to processing ---------------------
    # scan all the jpg and png images
    img_paths = sorted(list(scandir(args.test_path, suffix=('jpg', 'png', 'bmp'), recursive=True, full_path=True)))


    # --------------------  measure predicting time ---------------------
    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    timings = np.zeros((len(img_paths), 1))
    seq = 0

    print('testing ...\n')
    # ------------------------------------------------------------------

    for img_path in img_paths:
        img_name = img_path.replace(args.test_path+'/', '')
        print(f'Processing: {img_name}')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # prepare data
        img_t = img2tensor(img / 255., bgr2rgb=True, float32=True)

        # SNR map
        img_nf = img_t.permute(1, 2, 0).numpy() * 255.0
        img_nf = cv2.blur(img_nf, (5, 5))
        img_nf = img_nf * 1.0 / 255.0
        img_nf = torch.Tensor(img_nf).float().permute(2, 0, 1)

        dark = img_t
        dark = dark[0:1, :, :] * 0.299 + dark[1:2, :, :] * 0.587 + dark[2:3, :, :] * 0.114  # gray-scale
        light = img_nf
        light = light[0:1, :, :] * 0.299 + light[1:2, :, :] * 0.587 + light[2:3, :, :] * 0.114
        noise = torch.abs(dark - light)  # noise map

        mask = torch.div(light, noise + 0.0001)  # SNR map = clear map / noise map

        height = mask.shape[1]
        width = mask.shape[2]
        mask_max = torch.max(mask.view(1, -1), dim=1)[0]
        mask_max = mask_max.view(1, 1, 1)
        mask_max = mask_max.repeat(1, height, width)
        mask = mask * 1.0 / (mask_max + 0.0001)  # normalize its values to range [0, 1]

        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()
        mask = mask.unsqueeze(0).to(device)

        img_t = img_t.unsqueeze(0).to(device)

        # inference
        with torch.no_grad():
            # check_image_size
            H, W = img_t.shape[2:]
            img_t = check_image_size(img_t, down_factor)

            # --------------------  measure predicting time ---------------------
            starter.record()

            output_t = net(img_t, mask)

            ender.record()
            torch.cuda.synchronize()  # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
            timings[seq] = curr_time
            seq += 1
            # ------------------------------------------------------------------

            output_t = output_t[:,:,:H,:W]
            output = tensor2img(output_t, rgb2bgr=True, min_max=(0, 1))

        del output_t
        torch.cuda.empty_cache()

        output = output.astype('uint8')

        # save restored img
        save_restore_path = img_path.replace(args.test_path, result_root)
        imwrite(output, save_restore_path)

    print(f'\nAll results are saved in {result_root}')

    avg = timings.sum() / len(img_paths)
    print('\navg={}\n'.format(avg))
