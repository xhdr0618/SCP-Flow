'''
Compute all category forecasting related metrics

This script evaluates the performance of image classification models by:
1. Loading generated images and ground truth images from a results folder
2. Running inference using a pre-trained classifier on both sets of images
3. Computing and comparing various classification metrics (accuracy, precision, recall, f1, etc.)
4. Calculating agreement metrics (Cohen's Kappa) between ground truth and generated predictions
5. Saving all results as Excel files and visualizations (confusion matrix, ROC curve)

Note: It computes the metrics upon result folder with the following structure:
root_dir/
│
├── image/
│   ├── patient_id_gen.png  # Generated images
│   └── patient_id_gt.png   # Ground truth images
│
└── cls_results/            # Created by the script to store results
    ├── gene_test_result.xlsx
    ├── gt_test_result.xlsx
    ├── gene_confusion_matrix.png
    ├── gt_confusion_matrix.png
    ├── gene_roc_curve.png
    └── gt_roc_curve.png

Usage:
    python calculate_metric_class.py --eval_path '/path/to/results' --info_path '/path/to/SIGF_info.xlsx' --classifier_ckpt_path 'xxx/tHPM-LDM/pre-trained/image_classifier/classifier.ckpt'
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
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifier.net import load_classifier
from classifier.metric import compute_all_metrics, plot_confusion_matrix, plot_roc_curve
from sklearn.metrics import cohen_kappa_score

def compute_kappa(gt_prediction, gene_prediction):
    """
    compute cohen's kappa - the commitment of the classifier's result in gene and gt imgs
    """
    kappa = cohen_kappa_score(gt_prediction, gene_prediction)
    return kappa

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


class ResultsDataset(Dataset):
    def __init__(self, root_dir, info_path, suffix_gene="_gen.png"):
        """
        args:
            root_dir (str): the test dir, which contains a sub-folder called "image" with "{id}_gene.png/{id}_gt.png"
            suffix_gene (str): suffix of images (_gen.png/_gt.png)
        """
        self.root_dir = os.path.join(root_dir, 'image')
        self.suffix_gene = suffix_gene

        # load clip id and its label
        self.all_info = load_SIGF_info(info_path)

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
        gene_path = os.path.join(self.root_dir, patient_id + self.suffix_gene)
        gene_img = Image.open(gene_path)
        label = self.all_info[patient_id]
        if self.transform:
            gene_img = self.transform(gene_img)
        return {
            'image': gene_img,
            'label': label,
            'image_id': patient_id
        }


def inference_results(model, dataloader):
    case_predictions = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing"):
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            ids = batch['image_id']

            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            pred_labels = torch.argmax(probs, dim=1)

            for name, gt, pred, prob in zip(ids, labels, pred_labels, probs):
                case_predictions[name] = {
                    'gt_labels': [gt.item()],
                    'pred_labels': [pred.item()],
                    'pred_probs': [prob[1].item()]
                }

    return case_predictions

def processing_prediction(case_predictions):
    all_gt_labels = []
    all_pred_labels = []
    for predictions in case_predictions.values():
        all_gt_labels.extend(predictions['gt_labels'])
        all_pred_labels.extend(predictions['pred_labels'])
    return all_gt_labels, all_pred_labels

def calculate_metrics(case_predictions, save_dir, save_name):
    """
    compute metrics and save all resutls
    """
    # create folder to save metrics
    cls_results_dir = os.path.join(save_dir, 'cls_results')
    os.makedirs(cls_results_dir, exist_ok=True)

    # collect all labels and probs
    all_gt_labels = []
    all_pred_labels = []
    all_pred_probs = []

    for predictions in case_predictions.values():
        all_gt_labels.extend(predictions['gt_labels'])
        all_pred_labels.extend(predictions['pred_labels'])
        all_pred_probs.extend(predictions['pred_probs'])

    # compute metrics all-in-once
    metrics = compute_all_metrics(
        gt_labels=np.array(all_gt_labels),
        pred_labels=np.array(all_pred_labels),
        pred_probs=np.array(all_pred_probs)
    )

    metrics_dict = {
        'metric': ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'auc', 'ap'],
        'value': [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1'],
            metrics['specificity'],
            metrics['auc'],
            metrics['ap']
        ]
    }

    # save results to excel
    results_df = pd.DataFrame(metrics_dict)
    results_df.to_excel(os.path.join(cls_results_dir, f'{save_name}_test_result.xlsx'), index=False)

    # Confusion Matrix
    plot_confusion_matrix(
        metrics['cm'],
        os.path.join(cls_results_dir, f'{save_name}_confusion_matrix.png')
    )

    # ROC Curve
    plot_roc_curve(
        *metrics['roc_curve'],
        metrics['auc'],
        os.path.join(cls_results_dir, f'{save_name}_roc_curve.png')
    )

    return metrics_dict


def create_parser():
    parser = argparse.ArgumentParser(description='calculate category forecasting metrics')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of workers for dataset loading')
    parser.add_argument('--eval_path', type=str, default="/path/to/your/generated/results",
                        help='path containing the generated images (include gt and gene image)')
    parser.add_argument('--classifier_ckpt_path', type=str, default="pre-trained/image_classifier/classifier.ckpt")
    parser.add_argument('--suffix_gt', type=str, default="_gt.png", help='suffix of generated images')
    parser.add_argument('--suffix_gene', type=str, default="_gen.png", help='suffix of generated images')
    parser.add_argument('--info_path', type=str, default="./data/SIGF_info.xlsx",
                        help='path to SIGF_info.xlsx file (metainfo of SIGF_make)')

    return parser

def get_correspond_prediction(gt_prediction, gene_prediction):
    """
    get the corresponding pred results from classification on GT and GENE images
    """
    cases_id = gt_prediction.keys()
    gt_pred = []
    gene_pred = []
    for each_id in cases_id:
        gt_pred.append(gt_prediction[each_id]["pred_labels"])
        gene_pred.append(gene_prediction[each_id]["pred_labels"])

    return np.array(gt_pred), np.array(gene_pred)


def main():
    os.environ['TORCH_HOME'] = "./pre-trained"  # path to save torch pre-train ckpt
    parser = create_parser()
    args = parser.parse_args()

    model = load_classifier(args.classifier_ckpt_path, hidden_dim=(512, 512))
    print(f"cls metric working path: {args.eval_path}")

    # step1: firstly run cls metric on gt data
    dataset = ResultsDataset(
        root_dir=args.eval_path,
        suffix_gene=args.suffix_gt,
        info_path=args.info_path,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    gt_prediciton = inference_results(model, dataloader)
    save_name = "gt"
    metrics = calculate_metrics(gt_prediciton, args.eval_path, save_name)
    print("\nClassification Results on GT:")
    for metric, value in zip(metrics['metric'], metrics['value']):
        print(f"{metric}: {value:.4f}")

    # step2: next run cls metric on gene data
    dataset = ResultsDataset(
        root_dir=args.eval_path,
        suffix_gene=args.suffix_gene,
        info_path=args.info_path,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    gene_prediciton = inference_results(model, dataloader)
    save_name = "gene"
    metrics = calculate_metrics(gene_prediciton, args.eval_path, save_name)
    print("\nClassification Results on Gene:")
    for metric, value in zip(metrics['metric'], metrics['value']):
        print(f"{metric}: {value:.4f}")
    print("Agreement Metrics:")

    gt_pred, gene_pred = get_correspond_prediction(gene_prediciton, gt_prediciton)
    kappa = compute_kappa(gt_pred,gene_pred)
    print("cohen kappa:", kappa)

    # step3: calculate cohen's kappa
    # create saving path and read existing results
    cls_results_dir = os.path.join(args.eval_path, 'cls_results')
    excel_path = os.path.join(cls_results_dir, f'{save_name}_test_result.xlsx')
    existing_df = pd.read_excel(excel_path)

    # add kappa
    additional_metrics = pd.DataFrame({
        'metric': ['Cohen Kappa'],
        'value': [kappa]
    })

    # combine all results
    results_df = pd.concat([existing_df, additional_metrics], ignore_index=True)

    # save all results
    results_df.to_excel(excel_path, index=False)

    print("All done!")

if __name__ == '__main__':
    main()