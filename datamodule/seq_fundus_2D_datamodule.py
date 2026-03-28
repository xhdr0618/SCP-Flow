# -*- coding:utf-8 -*-
"""
dataset and dataloader for longitudial sequenece
"""
import os

import cv2
import numpy as np
import torch
from natsort import natsorted
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


def _center_crop_with_pad(image, center_x, center_y, crop_size):
    half = crop_size // 2
    h, w = image.shape[:2]
    x1 = max(center_x - half, 0)
    y1 = max(center_y - half, 0)
    x2 = min(center_x + half, w)
    y2 = min(center_y + half, h)
    crop = image[y1:y2, x1:x2]
    pad_y = crop_size - crop.shape[0]
    pad_x = crop_size - crop.shape[1]
    if pad_y > 0 or pad_x > 0:
        crop = np.pad(
            crop,
            ((0, pad_y), (0, pad_x), (0, 0)),
            mode="edge",
        )
    return crop


def _estimate_disc_center(image_bgr):
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    y1, y2 = h // 4, (3 * h) // 4
    x1, x2 = w // 4, (3 * w) // 4
    central = gray[y1:y2, x1:x2]
    threshold = np.percentile(central, 99.0)
    mask = central >= threshold
    if not np.any(mask):
        return w // 2, h // 2
    yy, xx = np.nonzero(mask)
    cx = int(np.round(xx.mean())) + x1
    cy = int(np.round(yy.mean())) + y1
    return cx, cy


def _estimate_structure_scalars(disc_roi_bgr):
    gray = cv2.cvtColor(disc_roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    radius = min(h, w) * 0.42
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2

    disc_thresh = np.percentile(gray[mask], 60.0)
    cup_thresh = np.percentile(gray[mask], 85.0)
    disc_mask = mask & (gray >= disc_thresh)
    cup_mask = mask & (gray >= cup_thresh)

    def _vertical_extent(binary_mask):
        rows = np.where(binary_mask.any(axis=1))[0]
        if rows.size == 0:
            return 1.0
        return float(rows[-1] - rows[0] + 1)

    disc_extent = _vertical_extent(disc_mask)
    cup_extent = _vertical_extent(cup_mask)

    vcdr = np.clip(cup_extent / max(disc_extent, 1.0), 0.0, 1.0)
    ocod = np.clip(cup_mask.sum() / max(float(disc_mask.sum()), 1.0), 0.0, 1.0)
    return np.float32(vcdr), np.float32(ocod)


def _build_structure_condition(image_bgr, roi_size):
    center_x, center_y = _estimate_disc_center(image_bgr)
    disc_roi = _center_crop_with_pad(image_bgr, center_x, center_y, roi_size)
    polar_roi = cv2.linearPolar(
        disc_roi,
        (roi_size / 2.0, roi_size / 2.0),
        roi_size / 2.0,
        cv2.WARP_FILL_OUTLIERS,
    )
    polar_roi = cv2.resize(polar_roi, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)
    vcdr, ocod = _estimate_structure_scalars(disc_roi)
    return disc_roi, polar_roi, vcdr, ocod


def _apply_missing_visits(images, times, labels, count, strategy, sample_seed):
    if count <= 0 or strategy == "none":
        return images, times, labels, np.zeros(images.shape[0], dtype=np.float32)

    history_len = max(images.shape[0] - 1, 0)
    if history_len == 0:
        return images, times, labels, np.zeros(images.shape[0], dtype=np.float32)

    candidates = list(range(history_len))
    count = min(count, len(candidates))
    if count <= 0:
        return images, times, labels, np.zeros(images.shape[0], dtype=np.float32)

    if strategy == "tail":
        missing_indices = candidates[-count:]
    elif strategy == "uniform":
        missing_indices = np.linspace(0, len(candidates) - 1, num=count, dtype=int).tolist()
    elif strategy == "random":
        rng = np.random.default_rng(sample_seed)
        missing_indices = sorted(rng.choice(candidates, size=count, replace=False).tolist())
    else:
        raise ValueError(f"Unsupported missing_visit_strategy: {strategy}")

    missing_mask = np.zeros(images.shape[0], dtype=np.float32)
    for miss_idx in missing_indices:
        missing_mask[miss_idx] = 1.0
        if miss_idx == 0:
            images[miss_idx] = 0.0
            labels[miss_idx] = 0
            continue
        images[miss_idx] = images[miss_idx - 1]
        times[miss_idx] = times[miss_idx - 1]
        labels[miss_idx] = labels[miss_idx - 1]

    return images, times, labels, missing_mask


def SeqFundusDatamodule(
        data_root,
        image_size=(256, 256),
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        mode='train',
        structure_roi_size=96,
        missing_visit_count=0,
        missing_visit_strategy='none',
        missing_seed=0,
):

    dataset = SIGF_Dataset(
        data_root=data_root,
        image_size=image_size,
        mode=mode,
        structure_roi_size=structure_roi_size,
        missing_visit_count=missing_visit_count,
        missing_visit_strategy=missing_visit_strategy,
        missing_seed=missing_seed,
    )

    output_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False
    )

    return output_dataloader


class SIGF_Dataset(Dataset):
    def __init__(
        self,
        data_root,
        image_size,
        mode,
        structure_roi_size=96,
        missing_visit_count=0,
        missing_visit_strategy='none',
        missing_seed=0,
    ):

        self.data_root = data_root
        self.mode = mode
        self.image_size = image_size
        self.structure_roi_size = structure_roi_size
        self.missing_visit_count = missing_visit_count
        self.missing_visit_strategy = missing_visit_strategy
        self.missing_seed = missing_seed
        
        self.data_list = []

        
        if self.mode == 'train':
            path1 = os.path.join(self.data_root, 'train')
            datalist1 = natsorted(os.listdir(path1))
            for dataname in datalist1:
                self.data_list.append(os.path.join(path1, dataname))
        elif self.mode == 'validation':
            path2 = os.path.join(self.data_root, 'validation')
            datalist2 = natsorted(os.listdir(path2))
            for dataname in datalist2:
                self.data_list.append(os.path.join(path2, dataname))
        else:
            path3 = os.path.join(self.data_root, 'test')
            datalist3 = natsorted(os.listdir(path3))
            for dataname in datalist3:
                self.data_list.append(os.path.join(path3, dataname))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        data_path = self.data_list[idx]
        data_id = os.path.basename(data_path)
        data = np.load(data_path)

        fundus_images = np.array(data['seq_imgs']).copy()
        real_times = np.array(data['times']).copy()
        image_label = np.array(data['labels']).copy()

        fundus_images, real_times, image_label, missing_mask = _apply_missing_visits(
            fundus_images,
            real_times,
            image_label,
            self.missing_visit_count,
            self.missing_visit_strategy,
            self.missing_seed + idx,
        )

        observed_idx = max(fundus_images.shape[0] - 2, 0)
        observed_image = fundus_images[observed_idx].astype(np.uint8)
        disc_roi, polar_roi, vcdr, ocod = _build_structure_condition(observed_image, self.structure_roi_size)

        fundus_images = torch.Tensor(fundus_images)
        fundus_images = fundus_images / 255
        fundus_images = torch.permute(fundus_images, (0, 3, 1, 2))
        fundus_images = torch.cat((fundus_images[:, 2:3, :, :], fundus_images[:, 1:2, :, :], fundus_images[:, 0:1, :, :]), dim=1)

        disc_roi = torch.from_numpy(disc_roi).float() / 255.0
        disc_roi = torch.permute(disc_roi, (2, 0, 1))
        disc_roi = torch.cat((disc_roi[2:3, :, :], disc_roi[1:2, :, :], disc_roi[0:1, :, :]), dim=0)

        polar_roi = torch.from_numpy(polar_roi).float() / 255.0
        polar_roi = torch.permute(polar_roi, (2, 0, 1))
        polar_roi = torch.cat((polar_roi[2:3, :, :], polar_roi[1:2, :, :], polar_roi[0:1, :, :]), dim=0)

        real_times = torch.LongTensor(real_times)
        image_label = torch.LongTensor(image_label)
        return {
            'image': fundus_images,
            'time': real_times,
            'image_id': data_id,
            'label': image_label,
            'vcdr': torch.tensor(vcdr, dtype=torch.float32),
            'ocod': torch.tensor(ocod, dtype=torch.float32),
            'disc_roi': disc_roi,
            'polar_roi': polar_roi,
            'missing_mask': torch.tensor(missing_mask, dtype=torch.float32),
        }

if __name__ == '__main__':
    # example of usage
    data_type = "train"
    data_loader = SeqFundusDatamodule(
            data_root="data/SIGF_make", # your data path
            batch_size=20,
            image_size=(256, 256),
            num_workers=0,
            shuffle=True,
            drop_last=False,
            mode=data_type,
        )
    for batch in tqdm(data_loader):
            bs = batch['image'].size(0)
            last_img = batch["image"][:,-1, ...]
            last_label = batch["label"][:, -1]
            last_name = batch["image_id"]
            break
