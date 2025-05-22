import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    # Seed config
    parser.add_argument('--seed', type=int, default=42)

    # Dataset config
    parser.add_argument('--dataset_name', type=str, default='UTKFace')
    parser.add_argument('--dataset_path', type=str, default='py97/UTKFace-Cropped')
    parser.add_argument('--dataset_split_ratio', type=float, nargs=3, default=[0.8, 0.1, 0.1])
    parser.add_argument('--dataset_target_label_idx', type=int, default=1)
    parser.add_argument('--gender_num_classes', type=int, default=2)
    parser.add_argument('--race_num_classes', type=int, default=5)

    # Data Augmentation
    parser.add_argument('--expand_prob', type=float, default=0.5)
    parser.add_argument('--expand_max_ratio', type=float, default=4)

    # Model config
    parser.add_argument('--clip_backbone', type=str, default='RN50')
    parser.add_argument('--clip_clf_dim', type=int, default=512)
    parser.add_argument('--return_features', type=bool, default=True)

    # Train config
    parser.add_argument('--lr_steps', type=int, nargs='+', default=[20000, 25000, 30000])
    parser.add_argument('--max_steps', type=int, default=150000)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--bf16', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--scheduler', type=str, default='Cosine')
    parser.add_argument('--is_train', type=bool, default=True)

    # Loss Weights
    parser.add_argument('--weight_equal_r', type=float, default=0.01)
    parser.add_argument('--weight_smooth', type=float, default=0.5)
    parser.add_argument('--weight_rc', type=float, default=0.001)
    parser.add_argument('--weight_mc', type=float, default=0.1)

    args = parser.parse_args()
    return args
