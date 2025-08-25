import torch.nn as nn

# DiceLoss 구현 예시 (필요시 추가)
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred와 target은 (N, C, H, W) 형태여야 함
        # pred는 sigmoid 통과 후 0~1 값
        intersection = (pred * target).sum(dim=[2,3]) # H, W 차원 합산
        union = pred.sum(dim=[2,3]) + target.sum(dim=[2,3])
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return (1 - dice).mean() # 각 채널/배치에 대한 평균