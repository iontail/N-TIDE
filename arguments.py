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

    # Model config
    parser.add_argument('--clip_backbone', type=str, default='RN50', help="Backbone used in CLIP model's image encoder")
    parser.add_argument('--neutral_init', type=str, choices=['random', 'person'], default='person', help="Init method for neutral vector: 'random' (CLIP-style) or 'person' (token from 'A photo of a person')")
    parser.add_argument('--feature_dim', type=int, default=512, help="Dimensionality of feature representation")

    # Train config
    parser.add_argument('--experiment_type', type=str, choices=['baseline', 'offline_teacher', 'offline_student'], default='offline_teacher', help="Experimnet training type")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--num_epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--bias_attribute', type=str, choices=['gender', 'race'], default='race', help="Attribute to analyze for bias; the other attribute will be used as the classification target")
   
    # -- CLIP model (Teacher)
    parser.add_argument('--t_optimizer', type=str, default='AdamW', help="Opti mizer for CLIP model")
    parser.add_argument('--t_scheduler', type=str, default='Cosine', help="Scheduler for CLIP model")
    parser.add_argument('--t_learning_rate', type=float, default=1e-4, help="Learning rate for CLIP model")
    parser.add_argument('--t_weight_decay', type=float, default=1e-5, help="Weight decay for CLIP model")
    parser.add_argument('--t_eta_min', type=float, default=1e-5, help="Minimum LR for CLIP scheduler")

    # -- CV model (Student)
    parser.add_argument('--s_optimizer', type=str, default='AdamW', help="Optimizer for CV model")
    parser.add_argument('--s_scheduler', type=str, default='Cosine', help="Scheduler for CV model")
    parser.add_argument('--s_backbone_lr', type=float, default=1e-5, help="Learning rate for CV model's backbone")
    parser.add_argument('--s_learning_rate', type=float, default=1e-4, help="Learning rate for CV model")
    parser.add_argument('--s_weight_decay', type=float, default=1e-2, help="Weight decay for CV model")
    parser.add_argument('--s_eta_min', type=float, default=1e-5, help="Minimum LR for CV scheduler")

    # -- Loss
    parser.add_argument('--gender_smoothing', type=float, default=0.0, help="Label smoothing factor for gender classification")
    parser.add_argument('--race_smoothing', type=float, default=0.1, help="Label smoothing factor for race classification")
    parser.add_argument('--lambda_t', type=float, default=0, help="Weight for Teacher loss, CLIP model's Align loss")
    parser.add_argument('--lambda_s', type=float, default=0, help="Weight for Student loss, CV models' KD lss")

    # -- Etc
    parser.set_defaults(is_train=True)
    parser.add_argument('--is_test', dest='is_train', action='store_false', help="Run in evaluation (test) mode")
    parser.set_defaults(use_wandb=True)
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false', help="Disable Wandb logging")
    parser.add_argument('--checkpoint_dir', type=str, default='./ckpt', help="Directory to save checkpoints")
    parser.add_argument('--teacher_ckpt_path', type=str, default=None, help="Path to teacher model checkpoint (required for training student)")
    parser.add_argument('--infer_ckpt_path', type=str, default=None, help="Path to model checkpoint for inference (required for test mode)")

    args = parser.parse_args()
    
    if args.is_train and args.experiment_type == 'offline_student' and args.teacher_ckpt_path is None:
        parser.error("'teacher_ckpt_path' must be specified when 'experiment_type' is 'offline_student'")

    if not args.is_train and args.infer_ckpt_path is None:
        parser.error("'infer_ckpt_path' must be specified when running in test mode (is_train=False)")
        
    return args
