"""
metric - classification related metrics
"""
import numpy as np
from typing import Tuple, Dict, Union
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt


def compute_accuracy(gt_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Compute classification accuracy
    Args:
        gt_labels: ground truth labels (0 or 1)
        pred_labels: predicted labels (0 or 1)
    Returns:
        accuracy score
    """
    return np.mean(gt_labels == pred_labels)


def compute_precision(gt_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Compute precision (TP / (TP + FP))
    Args:
        gt_labels: ground truth labels (0 or 1)
        pred_labels: predicted labels (0 or 1)
    Returns:
        precision score
    """
    TP = np.sum((gt_labels == 1) & (pred_labels == 1))
    FP = np.sum((gt_labels == 0) & (pred_labels == 1))
    return TP / (TP + FP + 1e-7)  # Add small epsilon to avoid division by zero


def compute_recall(gt_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Compute recall (TP / (TP + FN))
    Args:
        gt_labels: ground truth labels (0 or 1)
        pred_labels: predicted labels (0 or 1)
    Returns:
        recall score
    """
    TP = np.sum((gt_labels == 1) & (pred_labels == 1))
    FN = np.sum((gt_labels == 1) & (pred_labels == 0))
    return TP / (TP + FN + 1e-7)


def compute_f1_score(gt_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Compute F1 score (2 * precision * recall / (precision + recall))
    Args:
        gt_labels: ground truth labels (0 or 1)
        pred_labels: predicted labels (0 or 1)
    Returns:
        F1 score
    """
    precision = compute_precision(gt_labels, pred_labels)
    recall = compute_recall(gt_labels, pred_labels)
    return 2 * precision * recall / (precision + recall + 1e-7)


def compute_confusion_matrix(gt_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix
    Args:
        gt_labels: ground truth labels (0 or 1)
        pred_labels: predicted labels (0 or 1)
    Returns:
        2x2 confusion matrix: [[TN, FP], [FN, TP]]
    """
    return confusion_matrix(gt_labels, pred_labels)


def compute_specificity(gt_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Compute specificity (TN / (TN + FP))
    Args:
        gt_labels: ground truth labels (0 or 1)
        pred_labels: predicted labels (0 or 1)
    Returns:
        specificity score
    """
    TN = np.sum((gt_labels == 0) & (pred_labels == 0))
    FP = np.sum((gt_labels == 0) & (pred_labels == 1))
    return TN / (TN + FP + 1e-7)


def compute_roc_curve(gt_labels: np.ndarray, pred_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve and AUC score
    Args:
        gt_labels: ground truth labels (0 or 1)
        pred_probs: predicted probabilities for class 1
    Returns:
        fpr: false positive rates
        tpr: true positive rates
        auc_score: area under curve score
    """
    fpr, tpr, _ = roc_curve(gt_labels, pred_probs)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score


def compute_pr_curve(gt_labels: np.ndarray, pred_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Precision-Recall curve and AP score
    Args:
        gt_labels: ground truth labels (0 or 1)
        pred_probs: predicted probabilities for class 1
    Returns:
        precision: precision values
        recall: recall values
        ap_score: average precision score
    """
    precision, recall, _ = precision_recall_curve(gt_labels, pred_probs)
    ap_score = auc(recall, precision)
    return precision, recall, ap_score


def compute_all_metrics(gt_labels: np.ndarray, pred_labels: np.ndarray, pred_probs: np.ndarray = None) -> Dict[
    str, Union[float, np.ndarray]]:
    """
    Compute all binary classification metrics
    Args:
        gt_labels: ground truth labels (0 or 1)
        pred_labels: predicted labels (0 or 1)
        pred_probs: predicted probabilities for class 1 (optional)
    Returns:
        dictionary containing all metrics
    """
    metrics = {
        'accuracy': compute_accuracy(gt_labels, pred_labels),
        'precision': compute_precision(gt_labels, pred_labels),
        'recall': compute_recall(gt_labels, pred_labels),
        'f1': compute_f1_score(gt_labels, pred_labels),
        'specificity': compute_specificity(gt_labels, pred_labels),
        'cm': compute_confusion_matrix(gt_labels, pred_labels)
    }

    if pred_probs is not None:
        fpr, tpr, auc_score = compute_roc_curve(gt_labels, pred_probs)
        precision, recall, ap_score = compute_pr_curve(gt_labels, pred_probs)
        metrics.update({
            'auc': auc_score,
            'ap': ap_score,
            'roc_curve': (fpr, tpr),
            'pr_curve': (precision, recall)
        })

    return metrics


def plot_confusion_matrix(confusion_mat: np.ndarray, save_path: str = None):
    """
    Plot confusion matrix
    Args:
        confusion_mat: 2x2 confusion matrix
        save_path: path to save the plot (optional)
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_mat, cmap='Blues')

    # Add numbers to cells
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(confusion_mat[i, j]),
                     ha='center', va='center')

    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.colorbar()

    if save_path:
        fig = plt.gcf()
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.savefig(save_path)
        return img_array
    else:
        fig = plt.gcf()
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close()
        return img_array



def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float, save_path: str = None):
    """
    Plot ROC curve
    Args:
        fpr: false positive rates
        tpr: true positive rates
        auc_score: AUC score
        save_path: path to save the plot (optional)
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()

    if save_path:
        fig = plt.gcf()
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.savefig(save_path)
        return img_array
    else:
        fig = plt.gcf()
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close()
        return img_array


# Usage example
if __name__ == '__main__':
    # Generate some example data
    np.random.seed(42)
    gt_labels = np.random.randint(0, 2, 1000)
    pred_labels = np.random.randint(0, 2, 1000)
    pred_probs = np.random.random(1000)

    # Compute all metrics
    metrics = compute_all_metrics(gt_labels, pred_labels, pred_probs)

    # Print results
    print("Binary Classification Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    print(f"Specificity: {metrics['specificity']:.3f}")
    print(f"AUC Score: {metrics['auc_score']:.3f}")
    print(f"AP Score: {metrics['ap_score']:.3f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

    # Plot confusion matrix and ROC curve
    plot_confusion_matrix(metrics['confusion_matrix'], 'confusion_matrix.png')
    plot_roc_curve(*metrics['roc_curve'], metrics['auc_score'], 'roc_curve.png')