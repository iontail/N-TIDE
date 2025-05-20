import os
from torch.utils.data.dataloader import default_collate 
from src.datasets.dataset_UTKFace import UTKFace_Dataset, augment_train, augment_test

def get_dataset(args):
    """
    return UTKFace dataset and loader

    Args:
        args: 
              - is_train (bool): 학습 또는 테스트 데이터 로드 여부 플래그.
              - utkface_path (str): UTKFace 데이터셋의 루트 디렉토리 경로.
              - test_split_ratio (float, optional):  기본값 0.2.
              - random_state (int, optional)
    """
    test_split_ratio = getattr(args, 'test_split_ratio', 0.2)
    random_state = getattr(args, 'random_state', 42)

    if not hasattr(args, 'utkface_path') or not args.utkface_path:
        raise ValueError("There is no argument about \'utkface_path\'.")
    
    if args.is_train:
        train_path = os.path.join(args.utkface_path, 'train')
        train_dataset = UTKFace_Dataset(root_dir = train_path, train_mode = True, transform = augment_train,
                                        test_split_ratio=test_split_ratio, random_state = random_state)
        valid_dataset = UTKFace_Dataset(root_dir = train_path, train_mode = False, transform = augment_test,
                                        test_split_ratio = test_split_ratio, random_state = random_state)
        # 기본 default_collate 함수 사용
        data_collator = default_collate

        return train_dataset, valid_dataset, data_collator
    else:
        test_path = os.path.join(args.utkface_path, 'test')
        test_dataset = UTKFace_Dataset(root_dir=test_path, train_mode=False, transform=augment_test, test_split_ratio=test_split_ratio, random_state=random_state)
        data_collator = default_collate

        return test_dataset, None, data_collator