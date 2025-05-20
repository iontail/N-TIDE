import os
import pandas as pd
import itertools
from sklearn.metrics import accuracy_score, recall_score

# UTKFace race index to label mapping
RACE_ID_TO_LABEL = {
    0: 'White',
    1: 'Black',
    2: 'Asian',
    3: 'Indian',
    4: 'Others'
}

def calculate_bias_metrics(results_df, args):
    """
    Calculates bias metrics based on inference results.

    Args:
        results_df (pd.DataFrame): DataFrame containing inference results
                                   (filename, actual labels, predicted label).
        cfg: Configuration object containing dataset and model settings.

    Returns:
        dict: A dictionary containing calculated bias metrics.
    """

    target_idx = args.dataset_target_label_idx
    num_classes = args.num_classes
    predicted_col = f'predicted_label_idx_{target_idx}'

    # Determine the actual label column based on the target index
    if target_idx == 0:
        actual_col = 'actual_age'
        # Note: Age is often treated as regression or multi-class.
        # Bias metrics like EOD might be less standard here without defining a 'positive' age group.
        # We'll focus on Accuracy Disparity for age if it's the target.
        print("Warning: Calculating bias metrics for age. EOD might be less meaningful.")
    elif target_idx == 1:
        actual_col = 'actual_gender'
    elif target_idx == 2:
        actual_col = 'actual_race'
    else:
        raise ValueError(f"Invalid TARGET_LABEL_IDX: {target_idx}")

    # Define sensitive attribute (e.g., race) - Assuming race for now
    # You could make this configurable if needed
    sensitive_attribute_col = 'actual_race'
    groups = results_df[sensitive_attribute_col].unique()
    groups.sort()

    metrics = {}

    # 1. Accuracy Disparity
    accuracy_per_group = {}
    for group in groups:
        group_df = results_df[results_df[sensitive_attribute_col] == group]
        group_label = RACE_ID_TO_LABEL.get(group, f'Unknown_{group}') # Get label from mapping

        if not group_df.empty:
            accuracy = accuracy_score(group_df[actual_col], group_df[predicted_col])
            accuracy_per_group[group_label] = accuracy # Use label as key
        else:
            accuracy_per_group[group_label] = None # Or 0 or NaN

    metrics['accuracy_disparity'] = accuracy_per_group
    if len(accuracy_per_group) > 1:
         valid_accuracies = [v for v in accuracy_per_group.values() if v is not None]
         if valid_accuracies:
              metrics['max_accuracy_difference'] = max(valid_accuracies) - min(valid_accuracies)
         else:
              metrics['max_accuracy_difference'] = None

    # 2. Equal Opportunity Difference (EOD) - Meaningful for binary/multi-class classification
    # Calculates TPR difference between groups for the positive class (assuming binary or focusing on one class)
    # For simplicity, let's assume the 'positive' class is label '1' if binary,
    # or requires specific definition if multi-class.
    # This example assumes binary classification (like gender) or focuses on class 1 if multi-class.
    positive_class_label = 1 # Adjust if your positive class is different

    if num_classes >= 2 and target_idx != 0: # EOD is more standard for classification tasks
        tpr_per_group = {}
        for group in groups:
            group_df = results_df[(results_df[sensitive_attribute_col] == group) & (results_df[actual_col] == positive_class_label)]
            group_label = RACE_ID_TO_LABEL.get(group, f'Unknown_{group}') # Get label from mapping

            if not group_df.empty:
                # Recall for the positive class is TPR
                tpr = recall_score(group_df[actual_col], group_df[predicted_col], pos_label=positive_class_label, zero_division=0)
                tpr_per_group[group_label] = tpr # Use label as key
            else:
                tpr_per_group[group_label] = None # No positive samples in this group
        metrics['positive_class_label'] = positive_class_label # Store which class was considered positive
        metrics['tpr_per_group'] = tpr_per_group

        # Calculate max EOD between any two groups
        max_eod = 0.0
        valid_tprs = [v for v in tpr_per_group.values() if v is not None]
        if len(valid_tprs) > 1:
             for tpr1, tpr2 in itertools.combinations(valid_tprs, 2):
                  max_eod = max(max_eod, abs(tpr1 - tpr2))
        metrics['max_equal_opportunity_difference'] = max_eod

    return metrics