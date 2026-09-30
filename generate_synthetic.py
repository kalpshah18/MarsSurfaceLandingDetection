import os
import json
import numpy as np
import cv2
import pandas as pd
from pipeline import run_v0

def generate_synthetic_site(site_name, seed, num_tiles=20, tile_size=768, gsd=1.0):
    np.random.seed(seed)
    os.makedirs(f"data/synthetic/{site_name}", exist_ok=True)
    tiles = []
    
    for i in range(num_tiles):
        # Base DTM
        x, y = np.meshgrid(np.arange(tile_size), np.arange(tile_size))
        
        # 4 types of terrain based on site_name
        if 'crater' in site_name:
            # Crater: deep bowl
            cx, cy = np.random.randint(200, 568, 2)
            radius = np.random.randint(100, 300)
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            z = np.zeros((tile_size, tile_size))
            inside = dist < radius
            z[inside] = - (radius - dist[inside]) * 0.5 # slope up to 0.5 (~26 deg)
            z += np.random.randn(*z.shape) * 0.1 # noise
        elif 'rocky' in site_name:
            z = np.random.randn(tile_size, tile_size) * 0.5
            # Add some rocks
            for _ in range(50):
                rx, ry = np.random.randint(0, tile_size, 2)
                rr = np.random.randint(2, 10)
                rdist = np.sqrt((x - rx)**2 + (y - ry)**2)
                z[rdist < rr] += np.random.rand() * 2
        elif 'smooth' in site_name:
            # Plane
            z = 0.1 * x + 0.05 * y + np.random.randn(tile_size, tile_size) * 0.05
        else:
            # Dunes
            z = np.sin(x / 20.0) * 2.0 + np.random.randn(tile_size, tile_size) * 0.05
            
        # Synthetic ORI based on shading
        dz_dx = np.gradient(z, axis=1) / gsd
        dz_dy = np.gradient(z, axis=0) / gsd
        
        # Illumination
        sun_azimuth = np.radians(45)
        sun_elevation = np.radians(30)
        
        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(z)
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        nx /= norm; ny /= norm; nz /= norm
        
        lx = np.cos(sun_elevation) * np.cos(sun_azimuth)
        ly = np.cos(sun_elevation) * np.sin(sun_azimuth)
        lz = np.sin(sun_elevation)
        
        intensity = nx * lx + ny * ly + nz * lz
        intensity = np.clip(intensity, 0, 1)
        
        ori = (intensity * 255).astype(np.uint8)
        
        # Add some dark spots for shadows if not crater
        if 'crater' in site_name:
            ori[z < -10] = np.minimum(ori[z < -10], 30) # artificial shadow
            
        dtm_path = f"data/synthetic/{site_name}/dtm_{i}.npy"
        ori_path = f"data/synthetic/{site_name}/ori_{i}.png"
        np.save(dtm_path, z)
        cv2.imwrite(ori_path, ori)
        
        tiles.append({
            'site': site_name,
            'dtm': dtm_path,
            'ori': ori_path,
            'tile_id': f"{site_name}_{i}"
        })
        
    return tiles

def create_synthetic_dataset():
    sites = ['crater_1', 'crater_2', 'rocky_1', 'rocky_2', 'smooth_1', 'smooth_2', 'dunes_1', 'dunes_2']
    all_tiles = []
    for i, site in enumerate(sites):
        all_tiles.extend(generate_synthetic_site(site, seed=42+i))
        
    # Split by site
    dev_sites = ['crater_1', 'rocky_1', 'smooth_1', 'dunes_1']
    test_sites = ['crater_2', 'rocky_2', 'smooth_2', 'dunes_2']
    
    splits = {'DEV': [], 'TEST': []}
    for t in all_tiles:
        if t['site'] in dev_sites:
            splits['DEV'].append(t['tile_id'])
        else:
            splits['TEST'].append(t['tile_id'])
            
    with open('splits.json', 'w') as f:
        json.dump(splits, f, indent=4)
        
    manifest = pd.DataFrame(all_tiles)
    manifest.to_csv('data/MANIFEST.csv', index=False)
    
if __name__ == "__main__":
    create_synthetic_dataset()
    print("Synthetic dataset created.")
