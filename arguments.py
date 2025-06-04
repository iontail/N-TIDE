import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='N-TIDE: Debiasing Unimodal Vision Models via Neutral Text Inversion with CLIP')

    # Seed config
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")

    # Dataset config
    parser.add_argument('--dataset_name', type=str, choices=["UTKFace", "FairFace"], default='FairFace', help="Name of the dataset")

    # -- UTKFace
    parser.add_argument('--utkface_split_ratio', type=float, nargs=3, default=[0.7, 0.15, 0.15], help="UTKFace Train/Val/Test split ratio")
    parser.add_argument('--utkface_race_class', type=str, nargs=5, default=['White', 'Black', 'Asian', 'Indian', 'Others'], help="Race classes of UTKFace dataset (ordered by label indices)")
    
    # -- FairFace
    parser.add_argument('--fairface_split_ratio', type=float, nargs=2, default=[0.85, 0.15], help="FairFace Train/Val split ratio")
    parser.add_argument('--fairface_race_class', type=str, nargs=7, default=['East Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic', 'Southeast Asian'], help="Race classes of FairFace dataset (ordered by label indices)")
    parser.set_defaults(is_fairface_race_7=True)
    parser.add_argument('--is_fairface_race_4', dest='is_fairface_race_7', action='store_false', help='Use 4-class FairFace race classes')

    # -- Augmentation
    parser.add_argument('--train_transform_type', type=str, default='strong', choices=['strong', 'weak'], help="Training augmentation strategy")

    # Model config
    parser.add_argument('--clip_null_text', type=str, default='', help="Text prompt for CLIP model (Default: Null-text)")
    parser.add_argument('--clip_backbone', type=str, default='RN50', help="Backbone used in CLIP model's image encoder")
    parser.add_argument('--feature_dim', type=int, default=512, help="Dimensionality of feature representation")

    # Train config
    parser.add_argument('--train_mode', type=str, choices=['baseline', 'offline_teacher', 'offline_student'], default='baseline', help="Training mode type")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--num_epochs', type=int, default=25, help="Number of training epochs")
    parser.add_argument('--bf16', action='store_true', help="Enable bfloat16 precision training")

    # -- CLIP model
    parser.add_argument('--c_optimizer', type=str, default='AdamW', help="Optimizer for CLIP model")
    parser.add_argument('--c_scheduler', type=str, default='Cosine', help="Scheduler for CLIP model")
    parser.add_argument('--c_learning_rate', type=float, default=1e-3, help="Learning rate for CLIP model")
    parser.add_argument('--c_weight_decay', type=float, default=1e-4, help="Weight decay for CLIP model")
    parser.add_argument('--c_eta_min', type=float, default=1e-5, help="Minimum LR for CLIP scheduler")

    # -- CV model
    parser.add_argument('--m_optimizer', type=str, default='AdamW', help="Optimizer for CV model")
    parser.add_argument('--m_scheduler', type=str, default='Cosine', help="Scheduler for CV model")
    parser.add_argument('--m_learning_rate', type=float, default=1e-4, help="Learning rate for CV model")
    parser.add_argument('--m_weight_decay', type=float, default=1e-2, help="Weight decay for CV model")
    parser.add_argument('--m_eta_min', type=float, default=1e-5, help="Minimum LR for CV scheduler")

    # -- Loss
    parser.add_argument('--gender_smoothing', type=float, default=0.05, help="Label smoothing factor for gender classification")
    parser.add_argument('--race_smoothing', type=float, default=0.2, help="Label smoothing factor for race classification")
    parser.add_argument('--lambda_g', type=float, default=1, help="Weight for Gender Classification loss")
    parser.add_argument('--lambda_r', type=float, default=2, help="Weight for Race Classification loss") 
    parser.add_argument('--lambda_t', type=float, default=2, help="Weight for Teacher loss, CLIP model's Align loss")
    parser.add_argument('--lambda_s', type=float, default=2, help="Weight for Student loss, CV models' KD lss")

    # -- Etc
    parser.set_defaults(is_train=True)
    parser.add_argument('--is_test', dest='is_train', action='store_false', help="Run in evaluation (test) mode")
    parser.set_defaults(use_wandb=True)
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false', help="Disable Wandb logging")
    parser.add_argument('--checkpoint_dir', type=str, default='./ckpt', help="Directory to save checkpoints")

    args = parser.parse_args()
    return args
