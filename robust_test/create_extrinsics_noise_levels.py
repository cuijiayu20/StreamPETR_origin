#!/usr/bin/env python
# ------------------------------------------------------------------------
# Create Extrinsics Noise PKL files for different noise levels
# Generates L0-L4 level noise data for robustness testing
# ------------------------------------------------------------------------
# Usage:
#   python robust_test/create_extrinsics_noise_levels.py \
#       --root-path ./data/nuscenes \
#       --base-pkl data/nuscenes/nuscenes_infos_val_with_noise.pkl \
#       --output-dir data/nuscenes/extrinsics_levels
# ------------------------------------------------------------------------

import argparse
import os
import sys
import copy
import pickle
import random
import math
from math import cos, sin, acos

import numpy as np
import mmcv
from pyquaternion import Quaternion

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# Noise level definitions
# Level: (rotation_deg, translation_cm)
NOISE_LEVELS = {
    'L0': (0.0, 0.0),    # Clean baseline
    'L1': (0.5, 0.3),    # Standard online calibration error
    'L2': (1.0, 0.5),    # Mild drift
    'L3': (2.0, 1.0),    # Moderate drift (common test point)
    'L4': (5.0, 2.0),    # Severe misalignment (stress test)
}


def set_seed(seed=0):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_random_axis():
    """Generate a random unit vector on the unit sphere."""
    u_x = random.random()
    u_y = random.random()
    theta = acos(1 - 2 * u_x)
    phi = 2 * math.pi * u_y
    
    x = sin(theta) * cos(phi)
    y = sin(theta) * sin(phi)
    z = cos(theta)
    return [x, y, z]


def get_noise_rot_mat(noise_deg):
    """Generate a random rotation matrix with given noise magnitude.
    
    Args:
        noise_deg: Maximum rotation angle in degrees
    
    Returns:
        3x3 rotation matrix
    """
    if noise_deg == 0:
        return np.eye(3)
    
    rot_axis = get_random_axis()
    x, y, z = rot_axis
    
    # Random angle between 0 and noise_deg
    noise_theta = noise_deg * random.random()
    noise_theta = noise_theta / 180.0 * math.pi  # Convert to radians
    
    # Random sign
    if random.choice([True, False]):
        noise_theta *= -1
    
    c = cos(noise_theta)
    s = sin(noise_theta)
    
    rot_mat = np.array([
        [x*x*(1 - c) + c, x*y*(1 - c) + z*s, x*z*(1 - c) - y*s],
        [x*y*(1 - c) - z*s, y*y*(1 - c) + c, y*z*(1 - c) + x*s],
        [x*z*(1 - c) + y*s, y*z*(1 - c) - x*s, z*z*(1 - c) + c]
    ])
    
    return rot_mat


def get_noise_trans(noise_cm):
    """Generate a random translation vector with given noise magnitude.
    
    Args:
        noise_cm: Maximum translation in centimeters
    
    Returns:
        3D translation vector in meters
    """
    if noise_cm == 0:
        return np.array([0.0, 0.0, 0.0])
    
    noise_m = noise_cm * 0.01  # Convert cm to meters
    
    trans = []
    for _ in range(3):
        val = noise_m * random.random()
        if random.choice([True, False]):
            val *= -1
        trans.append(val)
    
    return np.array(trans)


def apply_noise_to_sensor2ego(s2e_r, s2e_t, noise_r, noise_t):
    """Apply rotation and translation noise to sensor2ego transform.
    
    Args:
        s2e_r: Original sensor2ego rotation (quaternion as list)
        s2e_t: Original sensor2ego translation (list)
        noise_r: Noise rotation matrix (3x3)
        noise_t: Noise translation vector (3,)
    
    Returns:
        Noisy s2e_r (quaternion), noisy s2e_t (array)
    """
    s2e_r_mat = Quaternion(s2e_r).rotation_matrix
    
    # Apply rotation noise: noise_r @ s2e_r_mat
    s2e_r_mat_noisy = noise_r @ s2e_r_mat
    s2e_r_noisy = Quaternion(matrix=s2e_r_mat_noisy)
    s2e_r_noisy = np.array(s2e_r_noisy.elements)
    
    # Apply translation noise
    s2e_t_noisy = noise_t + np.array(s2e_t) @ noise_r.T
    
    return s2e_r_noisy, s2e_t_noisy


def compute_sensor2lidar(s2e_r, s2e_t, e2g_r, e2g_t, l2e_r_mat, l2e_t, e2g_r_mat, e2g_t_lidar):
    """Compute sensor2lidar transform from sensor2ego and ego2global.
    
    Args:
        s2e_r: sensor2ego rotation (quaternion)
        s2e_t: sensor2ego translation
        e2g_r: ego2global rotation (quaternion)
        e2g_t: ego2global translation
        l2e_r_mat: lidar2ego rotation matrix (from LIDAR_TOP)
        l2e_t: lidar2ego translation (from LIDAR_TOP)
        e2g_r_mat: ego2global rotation matrix (from LIDAR_TOP)
        e2g_t_lidar: ego2global translation (from LIDAR_TOP)
    
    Returns:
        sensor2lidar_rotation, sensor2lidar_translation
    """
    s2e_r_mat = Quaternion(s2e_r).rotation_matrix
    e2g_r_mat_cam = Quaternion(e2g_r).rotation_matrix
    
    # Compute: sweep->ego->global->ego'->lidar
    R = (s2e_r_mat.T @ e2g_r_mat_cam.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (np.array(s2e_t) @ e2g_r_mat_cam.T + np.array(e2g_t)) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= np.array(e2g_t_lidar) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
        ) + np.array(l2e_t) @ np.linalg.inv(l2e_r_mat).T
    
    return R.T, T  # points @ R.T + T


def create_noise_for_level(base_noise_data, level_name, rot_deg, trans_cm):
    """Create noise data for a specific level.
    
    Args:
        base_noise_data: Base noise PKL data
        level_name: Level name (L0, L1, etc.)
        rot_deg: Rotation noise in degrees
        trans_cm: Translation noise in centimeters
    
    Returns:
        Modified noise data with new level's extrinsics noise
    """
    print(f'Generating {level_name}: rotation={rot_deg}°, translation={trans_cm}cm')
    
    # Deep copy the base data
    noise_data = copy.deepcopy(base_noise_data)
    camera_data = noise_data.get('camera', {})
    
    set_seed(0)  # Ensure reproducibility
    
    # For L0 (clean), we keep original sensor2lidar values
    if rot_deg == 0 and trans_cm == 0:
        for cam_file, cam_info in camera_data.items():
            extrinsics_noise = cam_info.get('noise', {}).get('extrinsics_noise', {})
            if 'sensor2lidar_rotation' in extrinsics_noise:
                # Copy clean values to noise keys
                extrinsics_noise['single_noise_sensor2lidar_rotation'] = extrinsics_noise['sensor2lidar_rotation']
                extrinsics_noise['single_noise_sensor2lidar_translation'] = extrinsics_noise['sensor2lidar_translation']
                extrinsics_noise['all_noise_sensor2lidar_rotation'] = extrinsics_noise['sensor2lidar_rotation']
                extrinsics_noise['all_noise_sensor2lidar_translation'] = extrinsics_noise['sensor2lidar_translation']
        return noise_data
    
    # For 'single' mode - each camera gets independent noise
    # Apply noise directly to sensor2lidar transform
    for cam_file, cam_info in camera_data.items():
        extrinsics_noise = cam_info.get('noise', {}).get('extrinsics_noise', {})
        
        if not extrinsics_noise:
            continue
        
        # Get original sensor2lidar (clean)
        s2l_r = extrinsics_noise.get('sensor2lidar_rotation')
        s2l_t = extrinsics_noise.get('sensor2lidar_translation')
        
        if s2l_r is None:
            continue
        
        s2l_r = np.array(s2l_r)
        s2l_t = np.array(s2l_t)
        
        # Generate and apply noise for 'single' mode
        noise_r = get_noise_rot_mat(rot_deg)
        noise_t = get_noise_trans(trans_cm)
        
        # Apply noise to sensor2lidar: noisy_s2l = noise @ s2l
        s2l_r_noisy = noise_r @ s2l_r
        s2l_t_noisy = noise_t + s2l_t @ noise_r.T
        
        extrinsics_noise['single_noise_sensor2lidar_rotation'] = s2l_r_noisy
        extrinsics_noise['single_noise_sensor2lidar_translation'] = s2l_t_noisy
    
    # For 'all' mode - all cameras in same frame get same noise
    set_seed(0)  # Reset seed for consistent 'all' noise
    processed_lidars = {}  # lidar_file -> (noise_r, noise_t)
    
    for cam_file, cam_info in camera_data.items():
        extrinsics_noise = cam_info.get('noise', {}).get('extrinsics_noise', {})
        
        if not extrinsics_noise:
            continue
        
        lidar_file = cam_info.get('lidar', {}).get('file_name', '')
        
        # Generate one noise per lidar sample (shared by all cameras in frame)
        if lidar_file not in processed_lidars:
            noise_r_all = get_noise_rot_mat(rot_deg)
            noise_t_all = get_noise_trans(trans_cm)
            processed_lidars[lidar_file] = (noise_r_all, noise_t_all)
        else:
            noise_r_all, noise_t_all = processed_lidars[lidar_file]
        
        # Get original sensor2lidar (clean)
        s2l_r = extrinsics_noise.get('sensor2lidar_rotation')
        s2l_t = extrinsics_noise.get('sensor2lidar_translation')
        
        if s2l_r is None:
            continue
        
        s2l_r = np.array(s2l_r)
        s2l_t = np.array(s2l_t)
        
        # Apply 'all' noise to sensor2lidar
        s2l_r_noisy = noise_r_all @ s2l_r
        s2l_t_noisy = noise_t_all + s2l_t @ noise_r_all.T
        
        extrinsics_noise['all_noise_sensor2lidar_rotation'] = s2l_r_noisy
        extrinsics_noise['all_noise_sensor2lidar_translation'] = s2l_t_noisy
    
    noise_data['camera'] = camera_data
    return noise_data


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create extrinsics noise PKL files for different levels')
    parser.add_argument('--base-pkl', type=str, 
                        default='data/nuscenes/nuscenes_infos_val_with_noise.pkl',
                        help='Path to base noise PKL file')
    parser.add_argument('--output-dir', type=str,
                        default='data/nuscenes/extrinsics_levels',
                        help='Output directory for level-specific PKL files')
    parser.add_argument('--levels', type=str, nargs='+',
                        default=['L0', 'L1', 'L2', 'L3', 'L4'],
                        choices=['L0', 'L1', 'L2', 'L3', 'L4'],
                        help='Noise levels to generate')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print('=' * 60)
    print('Extrinsics Noise Level PKL Generator')
    print('=' * 60)
    print(f'Base PKL: {args.base_pkl}')
    print(f'Output Dir: {args.output_dir}')
    print(f'Levels: {args.levels}')
    print('=' * 60)
    
    # Load base noise data
    if not os.path.exists(args.base_pkl):
        print(f'Error: Base PKL file not found: {args.base_pkl}')
        sys.exit(1)
    
    print('Loading base noise data...')
    base_noise_data = mmcv.load(args.base_pkl, file_format='pkl')
    print(f'  Camera entries: {len(base_noise_data.get("camera", {}))}')
    print(f'  Lidar entries: {len(base_noise_data.get("lidar", {}))}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate PKL for each level
    for level in args.levels:
        rot_deg, trans_cm = NOISE_LEVELS[level]
        
        print(f'\nProcessing {level}...')
        noise_data = create_noise_for_level(base_noise_data, level, rot_deg, trans_cm)
        
        output_file = os.path.join(args.output_dir, f'nuscenes_extrinsics_{level}.pkl')
        mmcv.dump(noise_data, output_file)
        print(f'  Saved: {output_file}')
    
    print('\n' + '=' * 60)
    print('All noise level PKL files generated successfully!')
    print('=' * 60)


if __name__ == '__main__':
    main()
