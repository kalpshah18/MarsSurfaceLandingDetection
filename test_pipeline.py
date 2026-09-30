import unittest
import numpy as np
import cv2
from pipeline import compute_gradient_hazard, augment_shadow, compute_fft_texture, maximal_rectangle

def brute_force_maximal_rectangle(matrix):
    if matrix.size == 0: return (0, 0, 0, 0)
    h, w = matrix.shape
    max_area = 0
    best_rect = (0, 0, 0, 0)
    for y1 in range(h):
        for x1 in range(w):
            if matrix[y1, x1] == 0: continue
            for y2 in range(y1, h):
                for x2 in range(x1, w):
                    sub = matrix[y1:y2+1, x1:x2+1]
                    if np.all(sub == 1):
                        area = (y2 - y1 + 1) * (x2 - x1 + 1)
                        if area > max_area:
                            max_area = area
                            best_rect = (x1, y1, x2 - x1 + 1, y2 - y1 + 1)
    return best_rect

class TestPipeline(unittest.TestCase):
    def test_maximal_rectangle(self):
        np.random.seed(42)
        for _ in range(200):
            mask = np.random.choice([0, 1], size=(15, 15), p=[0.3, 0.7])
            rect = maximal_rectangle(mask)
            bf_rect = brute_force_maximal_rectangle(mask)
            area = rect[2] * rect[3]
            bf_area = bf_rect[2] * bf_rect[3]
            self.assertEqual(area, bf_area)

    def test_sobel_gradient(self):
        # Synthetic ramp (slope of 2 per pixel in x)
        ramp = np.tile(np.arange(0, 100, 2, dtype=np.uint8), (50, 1))
        G = compute_gradient_hazard(ramp)
        self.assertTrue(np.all(G >= 0))
        # Gradient in interior should be uniform
        # Check interior gradient consistency (ignore borders)
        self.assertTrue(np.var(G[2:-2, 2:-2]) < 1e-4)
        
        # Checkerboard
        cb = np.zeros((50, 50), dtype=np.uint8)
        cb[::2, ::2] = 255
        cb[1::2, 1::2] = 255
        G_cb = compute_gradient_hazard(cb)
        self.assertTrue(np.max(G_cb) > 0.9) # High gradient at edges

    def test_fft_texture(self):
        # White noise
        np.random.seed(42)
        noise = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        t_score_n, R_map_n = compute_fft_texture(noise)
        self.assertTrue(np.mean(R_map_n) > 1.0) # Should be rough
        
        # Smooth gradient
        ramp = np.tile(np.arange(0, 64, dtype=np.uint8), (64, 1))
        t_score_s, R_map_s = compute_fft_texture(ramp)
        self.assertTrue(np.mean(R_map_s) < 1.0) # Should be smooth
        
        # Low-amplitude noise vs high-amplitude noise
        noise_low = np.random.randint(100, 110, (32, 32), dtype=np.uint8)
        noise_high = (noise_low - 100) * 25 # Scale up
        _, R_map_low = compute_fft_texture(noise_low)
        _, R_map_high = compute_fft_texture(noise_high)
        # Ratio should be invariant to amplitude scaling
        self.assertAlmostEqual(R_map_low[0,0], R_map_high[0,0], places=4)

    def test_shadow_module(self):
        img = np.ones((100, 100), dtype=np.uint8) * 100
        # Draw dark disc (value 0) of radius 15, area ~ 706 px
        cv2.circle(img, (50, 50), 15, 0, -1)
        G = np.zeros((100, 100), dtype=np.float32)
        G, mask = augment_shadow(img, G, tau=40, a_min=500)
        self.assertTrue(np.sum(mask > 0) > 600) # Should detect disc
        self.assertEqual(G[50, 50], 1.0)
        
        # Draw small dark disc radius 5, area ~ 78 px
        img2 = np.ones((100, 100), dtype=np.uint8) * 100
        cv2.circle(img2, (50, 50), 5, 0, -1)
        G2 = np.zeros((100, 100), dtype=np.float32)
        G2, mask2 = augment_shadow(img2, G2, tau=40, a_min=500)
        self.assertTrue(np.sum(mask2 > 0) == 0) # Should ignore small

    def test_slope_computation(self):
        # Tilted plane DTM
        # Equation: z = ax + by + c
        # Slope angle = arctan(sqrt(a^2 + b^2))
        x, y = np.meshgrid(np.arange(50), np.arange(50))
        # Resolution 1m/px, let a = 0.5, b = 0
        z = 0.5 * x
        expected_slope = np.degrees(np.arctan(0.5))
        
        # Compute slope using simple finite difference
        # baseline = 5m (since 1m/px, 5 pixels)
        baseline = 5
        dz_dx = (z[:, baseline:] - z[:, :-baseline]) / baseline
        dz_dy = (z[baseline:, :] - z[:-baseline, :]) / baseline
        
        # pad back
        slope_rad = np.arctan(np.sqrt(dz_dx[:-baseline, :]**2 + dz_dy[:, :-baseline]**2))
        slope_deg = np.degrees(slope_rad)
        
        self.assertAlmostEqual(np.mean(slope_deg), expected_slope, places=1)

if __name__ == '__main__':
    unittest.main()
