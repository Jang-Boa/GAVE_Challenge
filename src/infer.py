
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GAVE Challenge - Inference Script 
"""

import os
import glob
import shutil
import argparse
from typing import List

import torch
import tqdm

# Third-party & local imports used in your original code
import numpy as np
from PIL import Image  # noqa: F401 (may be used inside local modules)
import matplotlib.pyplot as plt  # noqa: F401
import cv2  # noqa: F401
import segmentation_models_pytorch as smp  # noqa: F401

from loss import DiceLoss  # noqa: F401
from resnet import ResNet, ResNetEncoder, BottleNeck, ResNet_Classifier
from unetdecoder import UNet
from avr import get_result, process_all_images, AVR, load_values
from inference_module import inference_avv, inference_optic


def parse_args():
    parser = argparse.ArgumentParser(description="GAVE Inference Pipeline (Folder only)")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing input images (searched recursively)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./BOA",
        help="Root directory to save outputs",
    )

    # Weights
    parser.add_argument("--vessel-weight", type=str, default="Vessel_model.pth")
    parser.add_argument("--vessel-support-weight", type=str, default="M-FunFound_Vessel.pth")
    parser.add_argument("--av-weight", type=str, default="AV_model.pth")
    parser.add_argument("--optic-weight", type=str, default="optic_mask_1024.pth")

    # Device
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (if using CUDA)")

    # Thresholds & options
    parser.add_argument("--vessel-thresh", type=float, default=0.5, help="Threshold for vessel binarization")
    parser.add_argument(
        "--image-exts",
        type=str,
        default="png,jpg,jpeg,tif,tiff,bmp",
        help="Comma-separated list of image extensions to search for",
    )

    return parser.parse_args()


def discover_images(image_dir: str, exts: List[str]) -> List[str]:
    """
    Accepts a directory path only. Returns sorted list of image files
    matching given extensions (case-insensitive).
    """
    if not os.path.isdir(image_dir):
        raise NotADirectoryError(f"--images must be a directory: {image_dir}")

    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(image_dir, f"**/*.{ext}"), recursive=True))

    files = [f for f in files if os.path.isfile(f)]
    files.sort()
    if not files:
        raise RuntimeError(f"No images found under directory: {image_dir}")
    return files

def get_device(use_cuda: bool, device_index: int) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device(f"cuda:{device_index}")
    return torch.device("cpu")

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def safe_rmtree(path: str):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
    except Exception as e:
        print(f"[WARN] Failed to remove {path}: {e}")


def safe_remove(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"[WARN] Failed to remove {path}: {e}")

def main():
    args = parse_args()

    # Resolve device
    device = get_device(args.use_cuda, args.device)
    print(f"[INFO] Using device: {device}")

    # Discover images
    exts = [e.strip().lower() for e in args.image_exts.split(",") if e.strip()]
    valid_list = discover_images(args.input, exts)
    print(f"[INFO] Found {len(valid_list)} images.")

    SAVE_RESULT_PATH = os.path.abspath(args.output)
    safe_makedirs(SAVE_RESULT_PATH)

    # Output subfolders
    save_avv = os.path.join(SAVE_RESULT_PATH, "Task1_2")
    save_optic = os.path.join(SAVE_RESULT_PATH, "Optic")
    safe_makedirs(save_avv)
    safe_makedirs(save_optic)

    # -----------------------------
    # Load Models
    # -----------------------------
    print("[INFO] Loading models ...")
    #### Load Vessel Model Weight 
    encoder_model = ResNet()
    vessel_model = UNet(encoder_model, num_classes=1)
    vessel_model.load_state_dict(torch.load(args.vessel_weight, weights_only=False))
    vessel_model.to(device)
    vessel_model.eval()
    
    #### Load AV Support Model Weight 
    support_model = UNet(encoder_model, num_classes=1)
    support_model.load_state_dict(torch.load(args.vessel_support_weight, weights_only=False))
    support_model.to(device)
    support_model.eval()
    
    #### Load AV Model Weight
    en = ResNetEncoder(Block=BottleNeck, block_num=[3, 4, 6, 3], num_channels=4)
    en = ResNet_Classifier(en, num_classes=2)
    av_model = UNet(en, num_classes=2)
    av_model.load_state_dict(torch.load(args.av_weight, weights_only=False))
    av_model.to(device)
    av_model.eval()
    
    #### Load Optic Model Weight
    optic_model = torch.load(args.optic_weight, weights_only=False)
    optic_model.to(device)
    optic_model.eval() 
    print("[INFO] Finished loading all models.")

    # -----------------------------
    # Inference Loop
    # -----------------------------
    for image_file in tqdm.tqdm(valid_list, desc="Inferencing"):
        inference_avv(image_file, vessel_model, support_model, av_model, device, save_avv)
        inference_optic(image_file, optic_model, device, save_optic)

    # -----------------------------
    # Post-processing & AVR
    # -----------------------------
    print("[INFO] Post-processing ...")
    # Vessel get_result
    in_dir = os.path.join(SAVE_RESULT_PATH, "Task1_2")
    out_dir = os.path.join(SAVE_RESULT_PATH, "Vessel")
    get_result(in_dir, out_dir, args.vessel_thresh)

    # Optic contour
    in_dir = os.path.join(SAVE_RESULT_PATH, "Optic")
    out_dir = os.path.join(SAVE_RESULT_PATH, "Contour")
    process_all_images(in_dir, out_dir)

    # AVR
    av_dir = os.path.join(SAVE_RESULT_PATH, "Vessel")
    disc_dir = os.path.join(SAVE_RESULT_PATH, "Contour")
    comparison_dir = os.path.join(SAVE_RESULT_PATH, "Vis")
    report_path = os.path.join(SAVE_RESULT_PATH, "AVRValue.txt")
    AVR(av_dir, disc_dir, comparison_dir, report_path)

    # Collect & save final values
    pred_values = load_values(report_path)
    if isinstance(pred_values, dict) and "AVR:" in pred_values:
        del pred_values["AVR:"]
    pred_items = sorted(pred_values.items()) if isinstance(pred_values, dict) else []

    final_dir = os.path.join(SAVE_RESULT_PATH, "Task3")
    safe_makedirs(final_dir)
    final_txt = os.path.join(final_dir, "AVR.txt")
    with open(final_txt, "w", encoding="utf-8") as f:
        for filename, value in pred_items:
            f.write(f"{filename} {value}\n")
    print(f"[INFO] Saved final AVR values to: {final_txt}")

    # Cleanup temps
    print("[INFO] Cleaning up temporary folders ...")
    for p in ["Optic", "Contour", "Vessel", "Vis"]:
        safe_rmtree(os.path.join(SAVE_RESULT_PATH, p))
    safe_remove(os.path.join(SAVE_RESULT_PATH, "AVRValue.txt"))

    print("[INFO] Complete!")

if __name__ == "__main__":
    main()