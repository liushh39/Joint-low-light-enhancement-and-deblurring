import random
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder
from basicsr.data.data_util import paired_paths_from_folder_prior
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.data.transforms import augment, paired_random_crop_prior
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import cv2
import torch
import numpy as np


def Normalize(data):
    data_normalize = data.copy()
    max = data_normalize.max()
    min = data_normalize.min()
    data_normalize = (data_normalize - min) / (max - min)
    return data_normalize

@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """


    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])
        self.use_flip = opt.get('use_flip', True)
        self.use_rot = opt.get('use_rot', True)
        self.crop_size = opt.get('crop_size', 256)
        self.scale = opt.get('scale', 1)


        if self.opt['phase'] == 'train':
            self.gt_folder, self.lq_folder, self.gt_fre_folder, self.gt_edge_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt[
                'dataroot_gt_fre'], opt['dataroot_gt_edge']
            if 'filename_tmpl' in opt:
                self.filename_tmpl = opt['filename_tmpl']
            else:
                self.filename_tmpl = '{}'

            self.paths = paired_paths_from_folder_prior([self.lq_folder, self.gt_folder, self.gt_fre_folder, self.gt_edge_folder],
                                                  ['lq', 'gt', 'gt_fre', 'gt_edge'], self.filename_tmpl)
            random.shuffle(self.paths)
        else:
            self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
            if 'filename_tmpl' in opt:
                self.filename_tmpl = opt['filename_tmpl']
            else:
                self.filename_tmpl = '{}'

            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        if self.opt['phase'] == 'train':
            gt_fre_path = self.paths[index]['gt_fre_path']
            img_bytes = self.file_client.get(gt_fre_path, 'gt_fre')
            img_gt_fre = imfrombytes(img_bytes, float32=True)

            gt_edge_path = self.paths[index]['gt_edge_path']
            img_bytes = self.file_client.get(gt_edge_path, 'gt_edge')
            img_gt_edge = imfrombytes(img_bytes, float32=True)


        # augmentation for training
        if self.opt['phase'] == 'train':
            # random crop
            img_gt, img_lq, img_gt_fre, img_gt_edge = paired_random_crop_prior(img_gt, img_lq, img_gt_fre, img_gt_edge, self.crop_size, self.scale, gt_path)
            # flip, rotation
            img_gt, img_lq, img_gt_fre, img_gt_edge = augment([img_gt, img_lq, img_gt_fre, img_gt_edge], self.use_flip, self.use_rot)

            # cv2.imshow('img_gt', img_gt)
            # cv2.imshow('high_fre', img_gt_fre)
            # cv2.imshow('img_gt_edge', img_gt_edge)
            # cv2.waitKey(0)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # SNR map
        img_nf = img_lq.permute(1, 2, 0).numpy() * 255.0
        img_nf = cv2.blur(img_nf, (5, 5))
        img_nf = img_nf * 1.0 / 255.0
        img_nf = torch.Tensor(img_nf).float().permute(2, 0, 1)


        if self.opt['phase'] == 'train':
            # edge map
            img_edge = img2tensor(img_gt_edge, bgr2rgb=True, float32=True)

            # img_edge = img_gt.permute(1, 2, 0).numpy() * 255.0
            #
            # # Sober算子
            # # 核函数的取值范围：1，3，5，7，9，核函数过大效果不好
            # Ksize = 3
            # sobelx = cv2.Sobel(img_edge, cv2.CV_64F, 1, 0, ksize=Ksize)
            # sobely = cv2.Sobel(img_edge, cv2.CV_64F, 0, 1, ksize=Ksize)
            # # sobel-x方向
            # sobel_X = cv2.convertScaleAbs(sobelx)
            # # sobel-y方向
            # sobel_Y = cv2.convertScaleAbs(sobely)
            # # sobel-xy方向
            # img_edge = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)
            # img_edge = img_edge * 1.0 / 255.0
            # img_edge = torch.Tensor(img_edge).float().permute(2, 0, 1)

            # high-frequency map
            high_fre = img2tensor(img_gt_fre, bgr2rgb=True, float32=True)
            # # cv2.imshow('img_idct', HIGH[:, :, 0])
            # # cv2.imshow('img_idct1', HIGH[:, :, 1])
            # # cv2.imshow('img_idct2', HIGH[:, :, 2])
            # # cv2.imshow('high_fre', high_fre)
            # # cv2.waitKey(0)

        if self.opt['phase'] == 'val':
            img_edge = []
            high_fre = []
            img_edge = torch.Tensor(img_edge).float()
            high_fre = torch.Tensor(high_fre).float()


        # normalize(img_nf, self.mean, self.std, inplace=True)
        #
        # # normalize
        # normalize(img_lq, self.mean, self.std, inplace=True)
        # normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'nf': img_nf, 'edge': img_edge, 'gt_fre': high_fre}

    def __len__(self):
        return len(self.paths)

