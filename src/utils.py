import torch
from torch import nn

def initialize_weights(m):
    """
    Initialize the weights of the model layers.

    Args:
    m (nn.Module): A module in the neural network

    Note:
    - Uses Kaiming initialization for Conv2d and Linear layers
    - For BatchNorm2d, weight is set to 1 and bias to 0
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

import albumentations as A
from albumentations.pytorch import ToTensorV2
# -- 1. 데이터 증강 (Augmentation) 파이프라인 추가 및 정규화 수정 --
def get_transforms(resize, mean, std):
    transform_list = [
        A.Resize(resize[0], resize[1], p=1), # 이미지 크기 조절
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0), # 0-255 이미지를 입력받아 정규화
        # A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ToTensorV2(), # NumPy 배열을 PyTorch 텐서로 변환
    ]

    return A.Compose(transform_list)