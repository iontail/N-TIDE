import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    # Seed config
    parser.add_argument('--seed', type=int, default=42)

    # Dataset config
    parser.add_argument('--dataset_name', type=str, default='UTKFace')
    parser.add_argument('--dataset_path', type=str, default='py97/UTKFace-Cropped')
    parser.add_argument('--dataset_split_ratio', type=float, nargs=3, default=[0.8, 0.1, 0.1])
    parser.add_argument('--gender_classes', type=str, nargs=2, default=['man', 'woman'])
    parser.add_argument('--race_classes', type=str, nargs=5, default=['White', 'Black', 'Asian', 'Indian', 'Others'])

    # Data Augmentation

    # Model config
    parser.add_argument('--clip_text_prompt', type=str, default='A photo of a')
    parser.add_argument('--clip_backbone', type=str, default='RN50')
    parser.add_argument('--feature_dim', type=int, default=512)

    # Train config
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--bf16', type=bool, default=False)
    
    parser.add_argument('--distill_mode', type=str, choices=['offline', 'online'], default='offline')
    parser.add_argument('--finetune_model', type=str, choices=['teacher', 'student'], default='teacher')

    # -- CLIP
    parser.add_argument('--c_learning_rate', type=float, default=1e-3)
    parser.add_argument('--c_weight_decay', type=float, default=1e-4)
    parser.add_argument('--c_optimizer', type=str, default='Adam')
    parser.add_argument('--c_scheduler', type=str, default='Cosine')
    parser.add_argument('--c_eta_min', type=float, default=1e-5)

    # -- CV model
    parser.add_argument('--m_learning_rate', type=float, default=1e-3)
    parser.add_argument('--m_weight_decay', type=float, default=1e-4)
    parser.add_argument('--m_optimizer', type=str, default='Adam')
    parser.add_argument('--m_scheduler', type=str, default='Cosine')
    parser.add_argument('--m_eta_min', type=float, default=1e-5)

    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--use_wandb', type=bool, default=False)
    parser.add_argument('--checkpoint_dir', type=str, default='./ckpt')

    # Loss Weights
    parser.add_argument('--c_lambda', type=int, default=0.5)
    parser.add_argument('--m_lambda', type=int, default=0.5)
    parser.add_argument('--alpha', type=int, default=0.5)


    args = parser.parse_args()
    return args
