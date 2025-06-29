import torch
import torch.nn.functional as F
from itertools import combinations

def compute_bias_metrics(logits, labels, group_labels, features):
    # Bias metrics
    bias_results = {
        'equal_opportunity_difference': 0.0,
        'equalized_odds_difference': 0.0,
        'demographic_parity_difference': 0.0,
        'representation_bias_distance': 0.0
    }
    
    preds = torch.argmax(logits, dim=1)
    groups = torch.unique(group_labels)
    masks = {g.item(): (group_labels == g) for g in groups}
    num_classes = logits.shape[1]
    
    for c in range(num_classes):
        # One-vs-Rest
        bin_labels = (labels == c).long()
        preds_c = (preds == c).long()

        tpr = {} # True Postive Rate
        fpr = {} # False Positive Rate
        pr  = {} # Positive Prediction Rate
        for g in groups:
            m = masks[g.item()]
            
            tp = ((preds_c == 1) & (bin_labels == 1) & m).sum().item()
            fn = ((preds_c == 0) & (bin_labels == 1) & m).sum().item()
            fp = ((preds_c == 1) & (bin_labels == 0) & m).sum().item()
            tn = ((preds_c == 0) & (bin_labels == 0) & m).sum().item()

            tpr[g.item()] = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
            fpr[g.item()] = fp / (fp + tn + 1e-8) if (fp + tn) > 0 else 0.0
            
            group_count = m.sum().item()
            pr[g.item()] = (tp + fp) / (group_count + 1e-8) if group_count > 0 else 0.0

        # Equal Opportunity Difference: max TPR gap between groups
        tpr_diff = max(tpr.values()) - min(tpr.values())

        # Demographic Parity Difference: max PR gap between groups
        demographic_parity_diff  = max(pr.values())  - min(pr.values())

        # Equalized Odds Difference: mean of TPR and FPR differences between groups
        # Representation Bias Distance: max cosine distance between group-wise feature means                
        equalized_odds_diff = 0.0
        representation_bias_dist = 0.0
        
        mu = {}
        for g in groups:
            mu[g.item()] = features[masks[g.item()]].mean(dim=0)

        for g1, g2 in combinations(groups.tolist(), 2):
            dist = 1 - F.cosine_similarity(mu[g1], mu[g2], dim=0).item()
            if dist > representation_bias_dist:
                representation_bias_dist = dist
                
            eod = 0.5 * (abs(tpr[g1] - tpr[g2]) + abs(fpr[g1] - fpr[g2]))
            if eod > equalized_odds_diff:
                equalized_odds_diff = eod
                
        bias_results['equal_opportunity_difference'] += tpr_diff
        bias_results['equalized_odds_difference'] += equalized_odds_diff
        bias_results['demographic_parity_difference'] += demographic_parity_diff
        bias_results['representation_bias_distance'] += representation_bias_dist

    # Average over all classes (mean of bias metrics computed per class)
    for key in bias_results:
        bias_results[key] /= num_classes
        
    return bias_results
