Dear Reviewer,

Thank you for this careful comment. Your point is correct, and it led us to find and fix a problem in our synthetic data. We summarize what was wrong, answer your questions directly, and describe the changes to the manuscript. Section, figure and table numbers refer to the revised version. Because we added a figure of the crater model, the former Fig. 1 is now Fig. 2.

**1. Is the point flawed?**

No. At a sun elevation of 30° a simple crater should show a crescent-shaped cast shadow just inside the sun-facing rim, with the far wall and most of the floor lit. An almost entirely dark crater is only expected at very low sun. Two small refinements, which strengthen rather than weaken your argument:

- For a paraboloidal bowl, a cast shadow exists only when the wall slope at the rim exceeds the sun elevation, i.e. when d/D > tan(e)/4 ≈ 0.144 at e = 30°. Its length along the sun line is then 1 − tan(e)/(4·d/D) of the diameter, about 28% for d/D = 0.2. Your estimate of about 35% is of the same order; the exact value depends on the wall profile.
- The craters in our original dataset did not have steep walls. They were conical with a constant wall slope of 26.6°, which is below the sun elevation, so physically they should have cast no shadow at all.

**2. Depth and diameter assumed in the original rendering**

The original craters had rim diameters of 200 to 600 m and depths of 50 to 150 m (d/D = 0.25), with a conical, rimless profile. The dark interior in the former Fig. 1(a) was not a cast shadow. The generator set every crater pixel deeper than 10 m to a low intensity, independent of the sun direction. As you note, a fixed depth threshold has no physical basis, and the result was an unphysical, almost entirely dark crater.

**3. Changes to the dataset**

We regenerated the full dataset with a physical crater and image model (Section 3, Fig. 1):

- Craters are now paraboloidal simple craters with a raised rim (0.04 D) and an ejecta blanket. D ranges from 210 to 592 m and d/D from 0.10 (degraded) to 0.20 (fresh), giving depths of 25 to 108 m and maximum wall slopes of 22° to 39°.
- Images are rendered with Lambertian shading and cast shadows computed by ray marching the terrain model toward the sun. No other darkening is applied. We verified the rendered shadow length against the analytic value (27.3% vs 27.8% of the diameter at d/D = 0.2).
- At 30° elevation, 9 of the 20 test craters cast a shadow, covering on average 5.4% and at most 16.6% of the crater area. All other parameters were retuned on the development split only.

**4. Revised Fig. 2(a) (former Fig. 1(a))**

As you suggested, the figure now illustrates the sunlit and shadowed regions. Panel (a) shows the sun direction, the crater rim, the outline of the ray-marched cast shadow, the sunlit far wall, and the crater's diameter, depth and d/D. The new Fig. 1 adds a profile along the sun line with the grazing sun ray, for d/D = 0.10 (no cast shadow) and 0.20 (shadow covering 16% of the crater area).

**5. Effect on the results**

The physical rendering changed the results, and the analysis it enabled improved the method:

- The original values (AUROC 0.350 and 0.796) depended on the painted dark region and have been removed.
- The new analysis (Section 5.1) shows why intensity gradients struggle on craters. A wall facing the sun is lit near normal incidence and appears bright and nearly textureless, while shadows capture only the wall facing away from the sun. With realistic shading, gradients alone barely rank crater slopes (within-tile AUROC 0.497).
- We therefore added a first-order photometric slope cue, the deviation of local brightness from that of level ground, motivated by photoclinometry (Section 4.3), and we now scale both the gradient and the brightness cue by physical constants derived from the sun elevation and the slope threshold rather than by statistics of each image (Section 4.1).
- With the three cues combined, AUROC reaches 0.866 and PR-AUC 0.316 (3.9 times chance) on held-out sites, and the within-tile crater AUROC rises to 0.930. The full front-end runs in 28.5 ms per 768 × 768 tile on a single CPU core.
- The pipeline now returns a safe landing site on every crater, plain and dune test tile for footprints from 5 m to 157 m (Section 5.3), and an ablation shows that the earlier per-image gradient normalization had limited it to small footprints (Section 5.4).
- A new sun-elevation study (Section 5.5, Table 6) covers 15° to 60°. Because the constants follow the sun elevation, the combined map is the best variant at every elevation without retuning.
- We also added an evaluation on real HiRISE orthoimages with co-registered terrain models from eight sites (Section 6). With parameters transferred unchanged from the synthetic data, the hazard map ranks real terrain above a local-texture baseline at three of four held-out sites.

We are grateful for this comment. It corrected an error in our data generation and led directly to the improved method, the illumination analysis and the real-data evaluation in the revised manuscript.

Sincerely,
The authors
