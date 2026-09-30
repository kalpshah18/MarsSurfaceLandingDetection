import os
import json
import numpy as np
import pandas as pd
import cv2
import time
from pipeline import compute_gradient_hazard, augment_shadow, compute_fft_texture, fuse_and_select, maximal_rectangle
from evaluate import evaluate_tile

# Baselines
def baseline_std(image, win_size=15, thresh=20):
    mean, std = cv2.meanStdDev(image)
    blur = cv2.blur(image.astype(np.float32)**2, (win_size, win_size))
    mean_sq = cv2.blur(image.astype(np.float32), (win_size, win_size))**2
    std_map = np.sqrt(np.maximum(0, blur - mean_sq))
    return std_map > thresh

def baseline_laplacian(image, thresh=50):
    lap = cv2.Laplacian(image, cv2.CV_32F, ksize=3)
    var_map = cv2.blur(lap**2, (15, 15))
    return var_map > thresh

def run_ablations(dtm_path, ori_path, gsd=1.0):
    ori = cv2.imread(ori_path, cv2.IMREAD_GRAYSCALE)
    dtm = np.load(dtm_path)
    
    results = {}
    
    # A4 (v0 full)
    G_A1 = compute_gradient_hazard(ori)
    G_A2, shadow = augment_shadow(ori, np.copy(G_A1))
    texture, _ = compute_fft_texture(ori)
    
    def eval_mask(safe_mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(safe_mask, connectivity=8)
        filtered = np.zeros_like(safe_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 1500:
                filtered[labels == i] = 1
        rect = maximal_rectangle(filtered)
        
        # simple eval
        box_unsafe_fraction = 0.0
        if rect[2] > 0:
            box_unsafe_fraction = 0.0 # simplified for ablation script
        return rect[2]*rect[3] # returned area
        
    # Simplified logging for ablations
    return {
        'A1': eval_mask((G_A1 < 0.35).astype(np.uint8)),
        'A4': eval_mask(((1 - G_A2) * texture > 0.35).astype(np.uint8))
    }

if __name__ == "__main__":
    print("Ablations script created.")
