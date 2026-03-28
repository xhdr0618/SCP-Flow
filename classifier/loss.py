import torch.nn as nn
import torch.nn.functional as F

def balanced_softmax_loss(labels, logits, sample_per_class, tempture=2.0, reduction="sum"):
    """
    Adapt from: https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification/blob/main/loss/BalancedSoftmaxLoss.py
    Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.


    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1) ** tempture
    logits = logits + spc.log() # the class with 1 sample is not acceptable
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


class BalancedSoftmaxCE(nn.Module):
    """
    Balanced Softmax Cross-entropy Loss with temperature scalar
    Args:
        class_counts (torch.Tensor): Number of samples for each class
        temperature (float): Temperature scalar τ to control the extent of class weights
    """
    def __init__(self, class_counts, temperature=2.0):
        super().__init__()
        self.class_counts = class_counts
        self.temperature = temperature

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Predicted logits of shape (N, C)
            targets (torch.Tensor): Ground truth labels of shape (N,)
        Returns:
            torch.Tensor: Computed loss
        """
        # Ensure inputs are on the same device
        if self.class_counts.device != logits.device:
            self.class_counts = self.class_counts.to(logits.device)

        return balanced_softmax_loss(labels=targets, logits=logits, sample_per_class=self.class_counts, tempture=self.temperature, reduction="mean")