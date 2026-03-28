"""
image classifier:
per-train & interface(using load_classifier in classifier.net)

dataset requirement:

Example of the sturcture of the pretrained image dataset (base_dir)
Image naming pattern: {dataset_name}_{image_name}.extension

your/fundus_img/dataset_path/           # Base directory
├── train/                              # Training set
│   ├── 0/                              # Class 0 (negative)
│   │   ├── dataset1_image001.jpg       # Format: {dataset}_{name}.ext
│   │   ├── dataset1_image002.png
│   │   └── ...
│   └── 1/                              # Class 1 (positive)
│       ├── dataset1_image101.jpg
│       ├── dataset2_image102.png
│       └── ...
├── validation/                         # Validation set
│   ├── 0/                              # Class 0 (negative)
│   │   ├── SIGF_val001.jpg
│   │   ├── SIGF_val002.png
│   │   └── ...
│   └── 1/                              # Class 1 (positive)
│       ├── SIGF_val101.jpg
│       ├── SIGF_val102.png
│       └── ...
└── test/                               # Test set
    ├── 0/                              # Class 0 (negative)
    │   ├── SIGF_test001.jpg
    │   ├── SIGF_test002.png
    │   └── ...
    └── 1/                              # Class 1 (positive)
        ├── SIGF_test101.jpg
        ├── SIGF_test102.png
        └── ...
        
Note: 
we evaluate the performace of the classifier via `metrics.calculate_metric_class.py`, only keep a basic strucure of `on_test_start`, `test_step` and `on_test_epoch_end` as usage reference.
"""
# -*- coding:utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
import os
import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from classifier.net import Classifier
from classifier.dataloader import ClsImgDataSet
from classifier.metric import compute_all_metrics, plot_confusion_matrix, plot_roc_curve
from classifier.util import draw_loss_fig
from base.init_experiment import initExperiment
from classifier.loss import BalancedSoftmaxCE

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='image_classifier')
    parser.add_argument('--data_root', type=str, default='your/fundus_img/dataset_path')
    parser.add_argument('--result_root', type=str, default='./result_classifier/')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--image_size', default=(256, 256))
    parser.add_argument('--loss_weight', type=str, default="(0.64, 0.36)")
    parser.add_argument("--command", default="fit")  # command (fit/test) -> train/tests

    # Model args
    parser.add_argument('--hidden_dim', type=str, default='512,512')  # the hidden dim (MLP) for classifier
    parser.add_argument('--freeze_backbone', type=bool, default=True)

    # Training args
    parser.add_argument("--max_epochs",type=int, default=500)
    parser.add_argument("--base_learning_rate", type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--scale_lr', type=bool, default=False, help="scale base-lr by ngpu * batch_size * n_accumulate")
    parser.add_argument("--limit_train_batches", type=int, default=5000)
    parser.add_argument("--limit_val_batches", type=int, default=2000)
    # Lightning args
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', default="auto")
    parser.add_argument('--reproduce', type=int, default=True)
    return parser


def from_argparse_args(config, callbacks=None):
    args = {
        "callbacks": callbacks,
        "max_epochs": config.max_epochs,
        "accelerator": config.accelerator,
        "limit_train_batches": config.limit_train_batches,
        "limit_val_batches": config.limit_val_batches,
        "accumulate_grad_batches": config.accumulate_grad_batches,
        "devices": eval(config.devices) if config.devices != "auto" else "auto",
        "default_root_dir": config.default_root_dir,
        "val_check_interval": 50,
        "check_val_every_n_epoch": 1,
    }
    return args


class ClassifierTrainer(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        hidden_dim = tuple(map(int, opts.hidden_dim.split(',')))
        self.model = Classifier(hidden_dim=hidden_dim, n_classes=2,dropout_rate=0.5)
        tuple_str = opts.loss_weight.strip('()')
        loss_weight = [float(x) for x in tuple_str.split(',')]
        print(f"loss_weight: {loss_weight}")
        self.criterion = BalancedSoftmaxCE(class_counts=torch.tensor(loss_weight, dtype=torch.float32), temperature=2)
        self.train_loss = []
        self.val_loss = []
        self.best_f1 = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch['image'].to("cuda")
        labels = batch['label'].to("cuda")
        logits = self(images)
        loss = self.criterion(logits, labels)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "train/loss" in metrics:
            current_loss = metrics["train/loss"]
            self.train_loss.append(
                [self.current_epoch, current_loss.item() if torch.is_tensor(current_loss) else current_loss])
        draw_loss_fig(self.train_loss, saving_path=os.path.join(self.opts.default_root_dir, 'train_loss.png'))

    def on_validation_start(self):
        # Initialize dictionaries to store predictions by case
        self.val_case_predictions = {}
        self.val_loss_info = []


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """ using the SIGF_make to select samples"""
        images = batch['image'].to("cuda")
        labels = batch['label'].to("cuda")
        names = batch['name']

        logits = self(images)
        loss = self.criterion(logits, labels)

        # Get predictions
        probs = torch.softmax(logits,dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        # Log loss
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        # Store predictions by case
        for name, gt, pred, prob in zip(names, labels, pred_labels, probs):
            case_key = name
            if case_key not in self.val_case_predictions:
                self.val_case_predictions[case_key] = {
                    'gt_labels': [],
                    'pred_labels': [],
                    'pred_probs': []
                }
            self.val_case_predictions[case_key]['gt_labels'] = gt.item()
            self.val_case_predictions[case_key]['pred_labels'] = pred.item()
            self.val_case_predictions[case_key]['pred_probs'] = prob[1].item() # pred probs for class 1

        self.val_loss_info.append(loss.item())

    def on_validation_epoch_end(self):
        case_metrics = [
            {
                'case': case_key,
                'gt_label': predictions['gt_labels'],
                'pred_label': predictions['pred_labels'],
                'pred_prob': predictions['pred_probs']
            }
            for case_key, predictions in self.val_case_predictions.items()
        ]

        all_gt_labels, all_pred_labels, all_pred_probs = map(
            np.array,
            zip(*[(m['gt_label'], m['pred_label'], m['pred_prob']) for m in case_metrics])
        )

        # Compute case-level metrics
        metrics = compute_all_metrics(gt_labels=all_gt_labels, pred_labels=all_pred_labels, pred_probs=all_pred_probs)

        # Log scalar metrics
        for key, value in metrics.items():
            if isinstance(value, (float, int)):
                self.log(f"val/{key}", value, prog_bar=True)
        self.log(f"val_auc", metrics["auc"], prog_bar=True)
        self.log(f"val_f1", metrics["f1"], prog_bar=True)
        self.log(f"val_sen", metrics["recall"], prog_bar=True)

        self.best_f1 = metrics["f1"] if metrics["f1"] > self.best_f1 else self.best_f1
        self.log(f"val_best_f1", self.best_f1, prog_bar=True)

        # save validation loss
        avg_loss = torch.tensor(self.val_loss_info).mean()
        self.val_loss.append([self.current_epoch, avg_loss.item()])
        draw_loss_fig(self.val_loss, saving_path=os.path.join(self.opts.default_root_dir, 'val_loss.png'))
        os.makedirs(os.path.join(self.opts.default_root_dir, "training_process"), exist_ok=True)
        
        # save validation cm and roc
        cm = plot_confusion_matrix(metrics['cm'], os.path.join(self.opts.default_root_dir, "training_process", f'confusion_matrix_{self.global_step}.png'))
        roc_c = plot_roc_curve(*metrics['roc_curve'], metrics['auc'], os.path.join(self.opts.default_root_dir, "training_process", f'roc_curve_{self.global_step}.png'))
        self.logger.experiment.add_image("Confusion Matrix", torch.from_numpy(cm).permute(2, 0, 1), global_step=self.global_step, dataformats='CHW')
        self.logger.experiment.add_image("ROC Curve", torch.from_numpy(roc_c).permute(2, 0, 1), global_step=self.global_step, dataformats='CHW')

    def on_test_start(self):
        # Initialize case predictions and results DataFrames
        self.test_case_predictions = {}
        self.test_cases_result = pd.DataFrame(columns=["case", "gt_label", "pred_label", "pred_prob"])
        self.test_whole_result = pd.DataFrame(columns=["metric", "value"])


    def test_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']
        names = batch['name']

        logits = self(images)
        probs = torch.softmax(logits,dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        # Store predictions by case
        for name, gt, pred, prob in zip(names, labels, pred_labels, probs):
            case_key = name
            if case_key not in self.test_case_predictions:
                self.test_case_predictions[case_key] = {
                    'gt_labels': [],
                    'pred_labels': [],
                    'pred_probs': []
                }
            self.test_case_predictions[case_key]['gt_labels'].append(gt.item())
            self.test_case_predictions[case_key]['pred_labels'].append(pred.item())
            self.test_case_predictions[case_key]['pred_probs'].append(prob[1].item())  # pred probs for class 1

    def on_test_epoch_end(self, outputs):
        case_metrics = [
            {
                'case': case_key,
                'gt_label': predictions['gt_labels'],
                'pred_label': predictions['pred_labels'],
                'pred_prob': predictions['pred_probs']
            }
            for case_key, predictions in self.test_case_predictions.items()
        ]

        all_gt_labels, all_pred_labels, all_pred_probs = map(
            np.array,
            zip(*[(m['gt_label'], m['pred_label'], m['pred_prob']) for m in case_metrics])
        )

        # Compute case-level metrics
        metrics = compute_all_metrics(gt_labels=all_gt_labels, pred_labels=all_pred_labels, pred_probs=all_pred_probs)

        # Store overall results
        for metric_name, value in metrics.items():
            if isinstance(value, (float, int)):
                self.test_whole_result.loc[len(self.test_whole_result)] = {
                    "metric": metric_name,
                    "value": value
                }

        # Save results
        os.makedirs(self.opts.result_root, exist_ok=True)
        plot_confusion_matrix(metrics['cm'], os.path.join(self.opts.result_root, f'confusion_matrix.png'))
        plot_roc_curve(*metrics['roc_curve'], metrics['auc'], os.path.join(self.opts.result_root, f'roc_curve.png'))
        self.test_cases_result.to_csv(os.path.join(self.opts.result_root, 'test_cases_result.csv'), index=False)
        self.test_whole_result.to_csv(os.path.join(self.opts.result_root, 'test_whole_result.csv'), index=False)

        # Print results
        print("\nTest Results:")
        print(self.test_whole_result.to_string(index=False))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.opts.base_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            amsgrad=False
        )
        
        class CustomScheduler(torch.optim.lr_scheduler._LRScheduler):
            def __init__(self, optimizer, warmup_steps, start_factor=1.0, peak_factor=10.0, end_factor=0.5):
                self.warmup_steps = warmup_steps
                self.start_factor = start_factor
                self.peak_factor = peak_factor
                self.end_factor = end_factor
                super().__init__(optimizer)

            def get_lr(self):
                if self.last_epoch < self.warmup_steps:
                    # linear increase
                    factor = self.start_factor + (self.peak_factor - self.start_factor) * (
                                self.last_epoch / self.warmup_steps)
                else:
                    # linear decay
                    factor = max(self.end_factor,
                                 self.peak_factor - (self.peak_factor - self.end_factor) * (
                                             (self.last_epoch - self.warmup_steps) / self.warmup_steps))

                return [base_lr * factor for base_lr in self.base_lrs]

        scheduler = CustomScheduler(
            optimizer,
            warmup_steps=3000,
            start_factor=1.0,
            peak_factor=5.0,
            end_factor=0.5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 100
            }
        }


def main(opts):
    print(opts)
    model = ClassifierTrainer(opts)

    if opts.command == "fit":
        train_loader = DataLoader(
            ClsImgDataSet(opts.data_root, 'train', opts.image_size),
            batch_size=opts.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=opts.num_workers,
            pin_memory=False
        )

        val_loader = DataLoader(
            ClsImgDataSet(opts.data_root, 'validation', opts.image_size),
            batch_size=opts.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=opts.num_workers,
            pin_memory=False
        )
        print('Train dataset size:', len(train_loader.dataset))
        print('Validation dataset size:', len(val_loader.dataset))

        callbacks = [
            ModelCheckpoint(
                monitor='val_f1',
                mode='max',
                save_top_k=5,
                filename='val_f1-model-{val_f1:.4f}',
                save_weights_only=True
            ),
        ]

        trainer = pl.Trainer(**from_argparse_args(opts, callbacks=callbacks))
        trainer.num_sanity_val_steps = 0
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    else:  # Test mode
        test_loader = DataLoader(
            ClsImgDataSet(opts.data_root, 'test', opts.image_size),
            batch_size=opts.batch_size,
            shuffle=False,
            pin_memory=False
        )
        print('Test dataset size:', len(test_loader.dataset))

        if opts.resume:
            checkpoint = torch.load(opts.resume)
            model.load_state_dict(checkpoint['state_dict'])

        trainer = pl.Trainer(**from_argparse_args(opts))
        trainer.num_sanity_val_steps = 0
        trainer.test(model, test_loader)


if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    os.environ['TORCH_HOME'] = "./pre-trained"  # path to save torch pre-train ckpt
    initExperiment(opts)
    main(opts)