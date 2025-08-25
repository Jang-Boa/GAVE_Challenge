import os
from PIL import Image 
import tqdm
import cv2
import torch
import numpy as np
<<<<<<< HEAD
from utils import get_transforms
=======
from .utils import get_transforms
>>>>>>> d4b3098 (update)

thresh = 0.5
IMAGENET_MEAN = 0.0 
IMAGENET_STD = 1.0 
resize_scale = (1024, 1536)
av_kernel = np.ones((3, 3), np.uint8)
transforms = get_transforms(resize=resize_scale, mean=IMAGENET_MEAN, std=IMAGENET_STD)

def inference_avv(image_file, vessel_model, support_model, av_model, device, save_result_path):
    image = np.array(Image.open(image_file).convert('RGB')) # 컬러 이미지로 로드 (3채널)
    file_name = image_file.split('/')[-1]
    augmented = transforms(image=image)
    image = augmented['image'] # PyTorch 텐서 (C, H, W)
    x = image.unsqueeze(dim=0).to(device)

    pred_vessel = vessel_model(x)
    support_ves = support_model(x)
    x_2 = torch.cat([x, support_ves], dim=1)
    pred_av = av_model(x_2)
    
    pred_artery = pred_av.data[0].permute(1,2,0).detach().cpu().numpy()[:,:,0]
    pred_vein = pred_av.data[0].permute(1,2,0).detach().cpu().numpy()[:,:,1]
    pred_vessel = pred_vessel.data[0].permute(1,2,0).detach().cpu().numpy()[:,:,0]
    
    pred_artery = cv2.erode(pred_artery, av_kernel, iterations=1)
    pred_vein = cv2.erode(pred_vein, av_kernel, iterations=1)
<<<<<<< HEAD
=======
    pred_vessel = cv2.erode(pred_vessel, av_kernel, iterations=1)
>>>>>>> d4b3098 (update)

    pred_final = np.stack([pred_artery, pred_vessel, pred_vein], axis=-1)

    test_mask = (pred_final * 255).astype(np.uint8)
    mask_save = Image.fromarray(test_mask)
    mask_save.save(os.path.join(save_result_path, file_name), 'png')

def inference_optic(image_file, optic_model, device, save_result_path):
    inputs = np.array(Image.open(image_file))
    file_name = image_file.split('/')[-1]
    
    image = transforms(image=inputs)['image']
    with torch.no_grad():
        x = image.unsqueeze(axis=0).to(device)
        test_pred = optic_model(x)
    test_pred = torch.sigmoid(test_pred)
    
    # Probability
    img = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    pred = torch.zeros([1, 1, resize_scale[0], resize_scale[1]]).cpu()
    pred[test_pred > thresh] = 1
    pred[test_pred <= thresh] = 0
    pred = pred.data[0].data[0].detach().cpu().numpy()
    test_mask = (pred * 255).astype(np.uint8)
    optic_save = Image.fromarray(test_mask)
    optic_save.save(os.path.join(save_result_path, file_name), 'png')
     