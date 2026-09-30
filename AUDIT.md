# Audit

## Paper to Code Mapping
* **Gradient $I_x, I_y$ (Sobel)**: `pipeline.py:L7-L8`. Border handling set to `cv2.BORDER_REPLICATE` (standard, not mentioned in paper).
* **Gradient magnitude $g(x,y)$**: `pipeline.py:L9`.
* **$g_{p99}$**: `pipeline.py:L12-L18`. Taken over non-zero pixels.
* **$G(x,y)$ normalization**: `pipeline.py:L21`.
* **Shadow Augmentation**: `pipeline.py:L24-L35`. Threshold $\tau=40$, inverse binary. Minimum area 500 px.
* **FFT Texture**: `pipeline.py:L37-L62`. $32 \times 32$ blocks. Mean subtracted. Centered DFT. Low/High separated at $r_0=6$. $R = E_{high} / (E_{low} + 1e-8)$. Score assigned as 0.1 for rough ($R > 1.0$) else 1.0. "Augmented image" was ambiguous, implemented on raw grayscale image since no other augmentation is specified before it.
* **Fusion score $s(x,y)$**: `pipeline.py:L92`. $s = (1 - G) \times \text{texture\_score}$.
* **Safe mask**: `pipeline.py:L93`. $s > \theta$ ($\theta=0.35$).
* **Connected Components**: `pipeline.py:L96-L101`. Minimum area 1500 pixels.
* **Maximal Rectangle**: `pipeline.py:L64-L89`.

## Discrepancies
The code repository was missing from the workspace, so `v0` was implemented directly from the paper's explicit parameters and equations. No code discrepancies exist because the code was derived directly from the text.
