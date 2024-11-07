import numpy as np
import torch
import torch.nn as nn
from detection_utils import compute_bbox_targets

class LossFunc(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(LossFunc, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, classifications, regressions, anchors, gt_clss, gt_bboxes):

        device = classifications.device
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            targets_cls = gt_clss[j, :, :]
            targets_bbox = gt_bboxes[j, :, :]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            positive_indices = (targets_cls > 0).view(-1)
            non_negative_indices = (targets_cls >= 0).view(-1)
            num_positive_anchors = positive_indices.sum()

            if num_positive_anchors == 0:
                alpha_factor = torch.ones_like(classification) * self.alpha
                alpha_factor = 1.0 - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
                bce = -(torch.log(1.0 - classification))
                cls_loss = focal_weight * bce
                classification_losses.append(cls_loss.sum())
                regression_losses.append(torch.tensor(0).float().to(device))
                continue

            targets = torch.zeros_like(classification)
            targets[positive_indices, targets_cls[positive_indices] - 1] = 1

            alpha_factor = torch.ones_like(targets) * self.alpha
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(targets_cls.view(-1, 1), -1), cls_loss, torch.zeros_like(cls_loss))
            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            targets_bbox = targets_bbox[positive_indices, :]
            bbox_reg_target = compute_bbox_targets(anchor[positive_indices, :].reshape(-1, 4), targets_bbox.reshape(-1, 4))
            targets = bbox_reg_target.to(device)
            regression_diff = torch.abs(targets - regression[positive_indices, :])
            regression_losses.append(regression_diff.mean())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)

