import torch
import torch.nn.functional as F


def batch_hard_triplet_loss(features, labels, margin=0.3):
    features = F.normalize(features, p=2, dim=1)
    distances = torch.cdist(features, features, p=2)

    labels = labels.view(-1, 1)
    positive_mask = labels.eq(labels.t())
    negative_mask = ~positive_mask

    positive_mask.fill_diagonal_(False)

    hardest_positive = distances.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
    hardest_negative = distances.masked_fill(~negative_mask, float("inf")).min(dim=1).values

    valid = torch.isfinite(hardest_positive) & torch.isfinite(hardest_negative)
    if not valid.any():
        return features.sum() * 0.0

    return F.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()
