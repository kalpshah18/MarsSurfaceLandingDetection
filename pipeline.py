import cv2
import numpy as np

def compute_gradient_hazard(image_gray):
    I = image_gray.astype(np.float32)
    Ix = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    g = np.sqrt(Ix**2 + Iy**2)
    
    non_zero = g[g > 0]
    if len(non_zero) > 0:
        g_p99 = np.percentile(non_zero, 99)
    else:
        g_p99 = 1.0
    
    if g_p99 == 0:
        g_p99 = 1e-8
        
    G = np.minimum(1.0, g / g_p99)
    return G

def augment_shadow(image_gray, G, tau=40, a_min=500):
    _, binary = cv2.threshold(image_gray, tau, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(image_gray)
    G_aug = G.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= a_min:
            cv2.drawContours(mask, [cnt], 0, 255, -1)
            
    G_aug[mask == 255] = 1.0
    return G_aug, mask

def compute_fft_texture_ablation(image_gray, block_size=32, r0=6):
    h, w = image_gray.shape
    texture_score = np.ones((h, w), dtype=np.float32)
    R_map = np.zeros((h, w), dtype=np.float32)
    
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = image_gray[y:y+block_size, x:x+block_size].astype(np.float32)
            block -= np.mean(block)
            
            dft = np.fft.fft2(block)
            dft_shift = np.fft.fftshift(dft)
            mag = np.abs(dft_shift)
            
            cy, cx = block_size // 2, block_size // 2
            Y, X = np.ogrid[:block_size, :block_size]
            dist_sq = (X - cx)**2 + (Y - cy)**2
            
            low_mask = dist_sq <= r0**2
            high_mask = dist_sq > r0**2
            
            E_low = np.sum(mag[low_mask]**2)
            E_high = np.sum(mag[high_mask]**2)
            
            R = E_high / (E_low + 1e-8)
            R_map[y:y+block_size, x:x+block_size] = R
            if R > 1.0:
                texture_score[y:y+block_size, x:x+block_size] = 0.1
                
    return texture_score

def maximal_rectangle(matrix):
    if matrix.size == 0:
        return (0, 0, 0, 0)
    max_area = 0
    best_rect = (0, 0, 0, 0)
    h, w = matrix.shape
    heights = np.zeros(w, dtype=np.int32)
    
    for y in range(h):
        heights[matrix[y] == 1] += 1
        heights[matrix[y] == 0] = 0
        
        stack = []
        for x in range(w + 1):
            curr_h = heights[x] if x < w else 0
            while stack and curr_h < heights[stack[-1]]:
                top_x = stack.pop()
                h_rect = heights[top_x]
                w_rect = x if not stack else x - stack[-1] - 1
                area = h_rect * w_rect
                if area > max_area:
                    max_area = area
                    best_rect = (x - w_rect, y - h_rect + 1, w_rect, h_rect)
            stack.append(x)
    return best_rect

def fuse_and_select(G, texture_score=None, theta=0.35, min_component_area=1500):
    if texture_score is not None:
        s = (1.0 - G) * texture_score
    else:
        s = 1.0 - G
        
    safe_mask = (s > theta).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(safe_mask, connectivity=8)
    
    filtered_safe_mask = np.zeros_like(safe_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
            filtered_safe_mask[labels == i] = 1
            
    rect = maximal_rectangle(filtered_safe_mask)
    return s, safe_mask, filtered_safe_mask, rect

def run_v_grad(image_gray, theta=0.35, min_component_area=1500):
    G = compute_gradient_hazard(image_gray)
    return fuse_and_select(G, texture_score=None, theta=theta, min_component_area=min_component_area)

def run_v_grad_shadow(image_gray, theta=0.35, min_component_area=1500, tau=40, a_min=500):
    G = compute_gradient_hazard(image_gray)
    G_aug, _ = augment_shadow(image_gray, G, tau=tau, a_min=a_min)
    return fuse_and_select(G_aug, texture_score=None, theta=theta, min_component_area=min_component_area)

def run_v_fft_ablation(image_gray, theta=0.35, min_component_area=1500, tau=40, a_min=500):
    G = compute_gradient_hazard(image_gray)
    G_aug, _ = augment_shadow(image_gray, G, tau=tau, a_min=a_min)
    texture_score = compute_fft_texture_ablation(image_gray)
    return fuse_and_select(G_aug, texture_score=texture_score, theta=theta, min_component_area=min_component_area)

