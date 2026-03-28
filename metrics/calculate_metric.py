'''
Compute image generation quality metrics between generated and ground truth images

This script evaluates the quality of generated images by comparing them with ground truth images using multiple metrics:
1. Peak Signal-to-Noise Ratio (PSNR) - measures image quality based on pixel errors
2. Structural Similarity Index (SSIM) - measures perceived image similarity
3. Mean Squared Error (MSE) - measures pixel-wise differences
4. Fréchet Inception Distance (FID) - measures similarity of image distributions using deep features
5. Learned Perceptual Image Patch Similarity (LPIPS) - measures perceptual differences using a neural network

Note: It computes the metrics upon result folder with the following structure:
root_dir/
├── image/
│   ├── patient_id_gen.png  # Generated images
│   ├── patient_id_gt.png   # Ground truth images
├── image_metrics.csv       # Created by the script to store results

The script outputs aggregate statistics (mean and standard deviation) for each metric across all test images.

Usage:
    python calculate_metric.py --eval_path /path/to/results --suffix_gt _gt.png --suffix_gene _gen.png
'''
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import argparse
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

import torch.nn as nn
import torchvision.models as models

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
""" generation related result """
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.color import rgb2gray
from scipy import linalg
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOSS_FN_VGG = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(DEVICE)


""" get SIGF info """
def load_SIGF_info(file_path='./data/SIGF_info.xlsx'):
    """
    load and processing the Excel file of SIGF (the metainfo of SIGF_make)
    Args:
        file_path (str): path to SIGF_info.xlsx (I write the relative path here, but acturally I oftenly use absolute path since it is more simple)
    Returns:
        dict: the dic with patient_id as its key and the last element of label seq as its value
    """
    # read excel file
    df = pd.read_excel(file_path)

    # collect lines with type 'test'
    test_df = df[df['type'] == 'test']

    # init result dic
    result_dict = {}

    # get all labels
    for _, row in test_df.iterrows():
        patient_id = row['patient_id']
        labels = eval(str(row['labels']))
        last_label = labels[-1]

        result_dict[patient_id] = last_label

    return result_dict


""" image generation-related metrics """
class InceptionV3Feature(nn.Module):
    """get the feature of Inception V3"""

    def __init__(self):
        super().__init__()
        # load pre-trained inception v3
        inception = models.inception_v3(pretrained=True)
        # unly use the layers before Mixed_7c
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, inception.maxpool1,
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            inception.maxpool2, inception.Mixed_5b,
            inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b,
            inception.Mixed_6c, inception.Mixed_6d,
            inception.Mixed_6e, inception.Mixed_7a,
            inception.Mixed_7b, inception.Mixed_7c
        )

    def forward(self, x):
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)

FEATRUE_EXTRACTOR = InceptionV3Feature().to(device=DEVICE)

def calculate_statistics(features):
    """
    Ref: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=True)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    Ref: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_fid(tensor1, tensor2):
    """
    calculate Fréchet Inception Distance (FID)

    Args:
        tensor1: a cpu tensor with shape (C,H,W), the rang of this tensor is 0-1
        tensor2: a cpu tensor with shape (C,H,W), the rang of this tensor is 0-1

    Returns:
        float: FID
    """
    # make sure inputs are 4D tensors (batch, channel, height, width)
    if tensor1.dim() == 3:
        tensor1 = tensor1.unsqueeze(0)
    if tensor2.dim() == 3:
        tensor2 = tensor2.unsqueeze(0)


    FEATRUE_EXTRACTOR.eval()

    with torch.no_grad():
        # get feature using Inception v3
        features1 = FEATRUE_EXTRACTOR(tensor1.to(DEVICE)).cpu().numpy()
        features2 = FEATRUE_EXTRACTOR(tensor2.to(DEVICE)).cpu().numpy()

        # get statistics
        mu1, sigma1 = calculate_statistics(features1)
        mu2, sigma2 = calculate_statistics(features2)

        fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    return float(fid)


def normalize(img, pattern="-1~1"):
    if pattern == "-1~1":
        img = img.float() / 255.0
        img = (img - 0.5) * 2.0
    elif pattern == "0~1":
        img = img.float() / 255.0
    else:
        raise NotImplementedError(f"the input pattern:{pattern} is not supported")
    return img

def _to_numpy(input_data):
    if isinstance(input_data, torch.Tensor):
        return input_data.detach().cpu().numpy()
    return input_data

def calculate_mse(img1, img2):
    to255 = lambda x: np.clip(255*x, 0, 255).astype(np.int32)
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if img1.ndim == 4:
        img1 = img1[0, :, :, :]
    if img2.ndim == 4:
        img2 = img2[0, :, :, :]

    img1 = to255(np.array(img1.cpu()))
    img2 = to255(np.array(img2.cpu()))

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return float(mse)

def calculate_psnr(img1, img2, data_range=1):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two color images.

    Args:
    image1_path (str): Path to the first image
    image2_path (str): Path to the second image

    Returns:
    float: PSNR value between the two images
    """
    # Check if images have same dimensions

    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    if img1.ndim == 4:
        img1 = img1[0, :, :, :]
    if img2.ndim == 4:
        img2 = img2[0, :, :, :]
    img1 = _to_numpy(img1)
    img2 = _to_numpy(img2)
    psnr_value = psnr(img1, img2, data_range=data_range)

    return psnr_value


def calculate_ssim(img1, img2, data_range=1):
    """
    Calculate Structural Similarity Index (SSIM) between two color images.

    Args:
    image1_path (str): Path to the first image
    image2_path (str): Path to the second image

    Returns:
    float: SSIM value between the two images
    """
    # Check if images have same dimensions
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")

    if img1.ndim == 4:
        img1 = img1[0, :, :, :]
    if img2.ndim == 4:
        img2 = img2[0, :, :, :]
    img1 = _to_numpy(img1)
    img2 = _to_numpy(img2)

    # Calculate SSIM for multi-channel image
    ssim_value = ssim(rgb2gray(img1,channel_axis=0), rgb2gray(img2, channel_axis=0), data_range=data_range, use_sample_covariance=False, win_size=11, gaussian_weights=True, sigma=1.5, K1=0.01, K2=0.03)

    return ssim_value

def calculate_lpips(img1, img2):
    # LPIPS needs the images to be in the [-1, 1] range.
    img1 = (2 * img1 - 1).unsqueeze(dim=0)
    img2 = (2 * img2 - 1).unsqueeze(dim=0)
    lpsis = LOSS_FN_VGG(img1.to(DEVICE), img2.to(DEVICE))
    return lpsis.item()


def img_metrics(args, dataloader):
    # storage all metrics using one dict
    metrics_values = {
        'PSNR': [],
        'SSIM': [],
        'MSE': [],
        'FID': [],
        'LPIPS': []
    }

    # collect all metrics
    for batch in tqdm(dataloader):
        bs = batch['gt'].size(0)
        for i in range(bs):
            id, gt_img, gene_img = batch['id'][i], batch['gt'][i], batch['gene'][i]

            each_psnr = calculate_psnr(gt_img, gene_img, data_range=1)
            each_ssim = calculate_ssim(gt_img, gene_img, data_range=1)
            each_mse = calculate_mse(gt_img, gene_img)
            each_lpips = calculate_lpips(gt_img, gene_img)
            each_fid = calculate_fid(gt_img, gene_img)

            # add each metric
            metrics_values['PSNR'].append(each_psnr)
            metrics_values['SSIM'].append(each_ssim)
            metrics_values['MSE'].append(each_mse)
            metrics_values['LPIPS'].append(each_lpips)
            metrics_values['FID'].append(each_fid)

    # storage mean and std of all metrics of all test img-pairs (then we use mean for multiple runs)
    metrics_stats = {
        'Metric': [],
        'Mean': [],
        'Std': []
    }

    for metric_name, values in metrics_values.items():
        values = np.array(values)
        metrics_stats['Metric'].append(metric_name)
        metrics_stats['Mean'].append(np.mean(values))
        metrics_stats['Std'].append(np.std(values))

    # init DataFrame
    metrics_df = pd.DataFrame(metrics_stats)

    # save into 5 digit
    metrics_df['Mean'] = metrics_df['Mean'].round(5)
    metrics_df['Std'] = metrics_df['Std'].round(5)


    # create output dir
    os.makedirs(args.eval_path, exist_ok=True)

    # save to csv
    output_path = os.path.join(args.eval_path, 'image_metrics.csv')
    metrics_df.to_csv(output_path, index=False)
    print(f"\nImage Metrics statistics saved to: {output_path}")
    print(metrics_df)

    return metrics_df

""" classification metrics """

class ResultsDataset(Dataset):
    def __init__(self, root_dir, suffix_gt="_gt.png", suffix_gene="_gen.png"):
        """
        args:
            root_dir (str): the test dir, which contains a sub-folder called "image" with "{id}_gene.png/{id}_gt.png"
            suffix_gene (str): suffix of images (_gen.png/_gt.png)
        """
        self.root_dir = os.path.join(root_dir, 'image')
        self.suffix_gt = suffix_gt
        self.suffix_gene = suffix_gene

        # load clip id and its label
        self.all_info = load_SIGF_info()

        # sort according to clip_id
        self.patient_ids = list(self.all_info.keys())
        self.patient_ids.sort()

        # image processing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        gt_path = os.path.join(self.root_dir, patient_id + self.suffix_gt)
        gene_path = os.path.join(self.root_dir, patient_id + self.suffix_gene)

        gt_img = Image.open(gt_path)
        gene_img = Image.open(gene_path)

        label = self.all_info[patient_id]

        if self.transform:
            gt_img = self.transform(gt_img)
            gene_img = self.transform(gene_img)

        return {
            'gt': gt_img,
            'gene': gene_img,
            'label': label,
            'id': patient_id
        }


def create_parser():
    parser = argparse.ArgumentParser(description='calculate image generation metrics')

    parser.add_argument('--batch_size', type=int, default=27, help='batch size for training')
    parser.add_argument('--num_worker', type=int, default=6, help='number of workers for dataset loading')
    parser.add_argument('--eval_path', type=str, default="/your_path_for_testing", help='path containing the generated images (include gt and gene image)')
    parser.add_argument('--suffix_gt', type=str, default="_gt.png", help='path to pretrained classifier checkpoint')
    parser.add_argument('--suffix_gene', type=str, default="_gen.png", help='path to pretrained classifier checkpoint')

    return parser


if __name__ == '__main__':
    os.environ['TORCH_HOME'] = "./pre-trained"  # path to save torch pre-train ckpt
    parser = create_parser()
    args = parser.parse_args()
    print(f"metrics working dir: {args.eval_path}")

    dataset = ResultsDataset(root_dir=args.eval_path, suffix_gt=args.suffix_gt, suffix_gene=args.suffix_gene)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        drop_last=False,
    )
    img_metrics(args, dataloader)

