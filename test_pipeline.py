import unittest
import numpy as np
import cv2
from pipeline import compute_gradient_hazard, augment_shadow, maximal_rectangle
from footprint_metrics import EPSILON, footprint_ok, square_fit, safe_landing, rates

# The FFT texture module was removed from the headline pipeline; pipeline.py now
# only exposes compute_fft_texture_ablation, which does not return the R map
# that test_fft_texture checks. That test is skipped rather than rewritten.
try:
    from pipeline import compute_fft_texture
except ImportError:
    compute_fft_texture = None

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
        # Interior gradient is uniform (ignore the blur + Sobel border)
        self.assertTrue(np.var(G[4:-4, 4:-4]) < 1e-4)

        # A sharp step edge saturates the hazard
        step = np.full((50, 50), 60, dtype=np.uint8)
        step[:, 25:] = 200
        self.assertTrue(np.max(compute_gradient_hazard(step)) > 0.99)

        # p99 normalization (ablation) still available
        cb = np.zeros((50, 50), dtype=np.uint8)
        cb[::2, ::2] = 255
        cb[1::2, 1::2] = 255
        self.assertTrue(np.max(compute_gradient_hazard(cb, normalization='p99')) > 0.9)

    def test_gradient_reference_is_physical(self):
        # A ramp whose slope equals g_ref / 8 per pixel gives G = 1 in the interior.
        from pipeline import gradient_reference
        base = np.full((60, 60), 128, dtype=np.uint8)
        g_ref = gradient_reference(base, 30.0)
        self.assertAlmostEqual(g_ref, 8 * 128 / np.tan(np.radians(30)) * np.sin(np.radians(15)) / 5, places=4)
        ramp = (128 + (np.arange(60) - 30) * (g_ref / 8.0)).astype(np.float32)
        img = np.tile(ramp, (60, 1))
        G = compute_gradient_hazard(np.clip(img, 0, 255).astype(np.uint8), 30.0)
        self.assertGreater(G[30, 30], 0.95)

    @unittest.skipIf(compute_fft_texture is None,
                     'compute_fft_texture no longer exists in pipeline.py (FFT module is an ablation only)')
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

class TestFootprintCriteria(unittest.TestCase):
    def test_epsilon_fixed(self):
        self.assertEqual(EPSILON, 0.05)

    def test_square_box_passes_all(self):
        self.assertTrue(footprint_ok(25, 25, 25))
        self.assertTrue(square_fit(25, 25, 25))
        self.assertTrue(safe_landing(25, 25, 0.0, 25))

    def test_one_pixel_short_fails(self):
        self.assertFalse(footprint_ok(24, 26, 25))   # 624 < 625
        self.assertFalse(square_fit(24, 40, 25))
        self.assertFalse(safe_landing(24, 40, 0.0, 25))

    def test_elongated_box_passes_area_fails_square_fit(self):
        # 10 x 100 = 1000 px >= 25^2 = 625, but cannot hold a 25 x 25 square.
        self.assertTrue(footprint_ok(10, 100, 25))
        self.assertFalse(square_fit(10, 100, 25))
        self.assertFalse(square_fit(100, 10, 25))
        self.assertFalse(safe_landing(10, 100, 0.0, 25))

    def test_frac_unsafe_exactly_epsilon_passes(self):
        self.assertTrue(safe_landing(30, 30, 0.05, 25))
        self.assertTrue(safe_landing(30, 30, EPSILON, 25))

    def test_frac_unsafe_just_above_epsilon_fails(self):
        self.assertFalse(safe_landing(30, 30, 0.05 + 1e-9, 25))
        self.assertFalse(safe_landing(30, 30, 0.051, 25))

    def test_epsilon_from_pixel_counts(self):
        # 45 unsafe pixels in a 30 x 30 box is exactly 5%.
        frac = 45 / (30 * 30)
        self.assertTrue(safe_landing(30, 30, frac, 25))
        self.assertFalse(safe_landing(30, 30, 46 / (30 * 30), 25))

    def test_no_box(self):
        self.assertFalse(footprint_ok(0, 0, 1))
        self.assertFalse(square_fit(0, 0, 1))
        self.assertFalse(safe_landing(0, 0, float('nan'), 1))
        self.assertFalse(safe_landing(40, 40, float('nan'), 25))

    def test_rates_denominator_is_all_tiles(self):
        boxes = [
            {'w': 30, 'h': 30, 'frac_unsafe': 0.0},          # passes all three
            {'w': 10, 'h': 100, 'frac_unsafe': 0.0},         # area only
            {'w': 30, 'h': 30, 'frac_unsafe': 0.5},          # area + square, unsafe
            {'w': 0, 'h': 0, 'frac_unsafe': float('nan')},   # no box
        ]
        r = rates(boxes, 25)
        self.assertEqual(r, {'footprint_ok': 0.75, 'square_fit': 0.5, 'safe_landing': 0.25})

    def test_safe_landing_implies_square_fit_implies_footprint_ok(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            w, h = (int(v) for v in rng.integers(0, 60, 2))
            fu = float(rng.random())
            L = int(rng.integers(1, 60))
            if safe_landing(w, h, fu, L):
                self.assertTrue(square_fit(w, h, L))
            if square_fit(w, h, L):
                self.assertTrue(footprint_ok(w, h, L))


if __name__ == '__main__':
    unittest.main()
