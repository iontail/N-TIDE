import torch
import torch.nn.functional as F
from itertools import combinations
# ===============
#    수정 필요   
# ===============

def compute_metrics(logits, true_labels, group_labels, features):
    # Accuracy 
    preds = torch.argmax(logits, dim=1)
    accuracy_correct = (preds == true_labels).sum().item()

    # Bias metrics
    groups = torch.unique(group_labels)
    masks = {g.item(): (group_labels == g) for g in groups}
    num_classes = logits.shape[1]

    bias_results = {}
    for c in range(num_classes):
        # One-vs-Rest
        bin_labels = (true_labels == c).long()
        scores_c = logits[:, c]
        preds_c = (scores_c >= 0).long()

        tpr = {}
        fpr = {}
        pr  = {}
        for g in groups:
            m = masks[g.item()]
            tp = ((preds_c == 1) & (bin_labels == 1) & m).sum().item()
            fn = ((preds_c == 0) & (bin_labels == 1) & m).sum().item()
            fp = ((preds_c == 1) & (bin_labels == 0) & m).sum().item()
            tn = ((preds_c == 0) & (bin_labels == 0) & m).sum().item()

            tpr[g.item()] = tp / (tp + fn + 1e-8)
            fpr[g.item()] = fp / (fp + tn + 1e-8)
            pr[g.item()]  = (tp + fp) / (m.sum().item() + 1e-8)

        delta_tpr = max(tpr.values()) - min(tpr.values())
        delta_fpr = max(fpr.values()) - min(fpr.values())
        delta_pr  = max(pr.values())  - min(pr.values())

        mu = {}
        for g in groups:
            mu[g.item()] = features[masks[g.item()]].mean(dim=0)

        max_dist = 0.0
        for g1, g2 in combinations(groups.tolist(), 2):
            dist = torch.norm(mu[g1] - mu[g2], p=2).item()
            if dist > max_dist:
                max_dist = dist

        bias_results[f'class_{c}'] = {
            'equalized_odds_tpr_difference': delta_tpr,
            'equalized_odds_fpr_difference': delta_fpr,
            'equal_opportunity_difference': delta_tpr,
            'demographic_parity_difference': delta_pr,
            'representation_bias_distance': max_dist
        }

    result = {'accuracy_correct': accuracy_correct}
    result.update(bias_results)
    return result
