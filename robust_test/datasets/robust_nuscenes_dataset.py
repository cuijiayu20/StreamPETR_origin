# ------------------------------------------------------------------------
# Robust NuScenes Dataset for StreamPETR
# Extends CustomNuScenesDataset with noise injection capabilities
# ------------------------------------------------------------------------

import os
import numpy as np
import mmcv
from mmdet.datasets import DATASETS
from nuscenes.eval.common.utils import Quaternion

# Import the original dataset
from projects.mmdet3d_plugin.datasets.nuscenes_dataset import (
    CustomNuScenesDataset,
    convert_egopose_to_matrix_numpy,
    invert_matrix_egopose_numpy
)


@DATASETS.register_module()
class RobustNuScenesDataset(CustomNuScenesDataset):
    """NuScenes Dataset with Robustness Testing Support.
    
    Extends CustomNuScenesDataset with noise injection for robustness benchmarking.
    Supports: frame drop, extrinsics noise (without modifying original code).
    
    Args:
        noise_ann_file (str): Path to noise annotation pkl file.
        extrinsics_noise (bool): Whether to add extrinsics noise.
        extrinsics_noise_type (str): 'single' or 'all' camera noise.
        drop_frames (bool): Whether to enable frame drop.
        drop_ratio (int): Frame drop ratio (10, 20, ..., 90).
        drop_type (str): 'discrete' or 'consecutive' drop mode.
        noise_sensor_type (str): 'camera' or 'lidar'.
    """

    def __init__(self,
                 noise_ann_file='',
                 extrinsics_noise=False,
                 extrinsics_noise_type='single',
                 drop_frames=False,
                 drop_ratio=0,
                 drop_type='discrete',
                 noise_sensor_type='camera',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Noise configuration
        self.extrinsics_noise = extrinsics_noise
        self.extrinsics_noise_type = extrinsics_noise_type
        self.drop_frames = drop_frames
        self.drop_ratio = drop_ratio
        self.drop_type = drop_type
        self.noise_sensor_type = noise_sensor_type
        
        # Load noise data if needed
        if (self.extrinsics_noise or self.drop_frames) and noise_ann_file:
            noise_data = mmcv.load(noise_ann_file, file_format='pkl')
            self.noise_data = noise_data.get(noise_sensor_type, {})
            self.noise_camera_data = noise_data.get('camera', {})
        else:
            self.noise_data = {}
            self.noise_camera_data = {}
        
        # Print noise settings
        self._print_noise_settings()
    
    def _print_noise_settings(self):
        """Print current noise configuration."""
        print('=' * 50)
        print('Robust NuScenes Dataset - Noise Settings:')
        if self.drop_frames:
            print(f'  Frame Drop: ratio={self.drop_ratio}%, mode={self.drop_type}, sensor={self.noise_sensor_type}')
        if self.extrinsics_noise:
            print(f'  Extrinsics Noise: type={self.extrinsics_noise_type}')
        if not self.drop_frames and not self.extrinsics_noise:
            print('  No noise enabled (clean test)')
        print('=' * 50)

    def get_data_info(self, index):
        """Get data info with noise injection.
        
        Overrides parent method to inject noise into image paths and extrinsics.
        """
        info = self.data_infos[index]
        
        # Get ego pose matrices
        e2g_rotation = Quaternion(info['ego2global_rotation']).rotation_matrix
        e2g_translation = info['ego2global_translation']
        l2e_rotation = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        l2e_translation = info['lidar2ego_translation']
        e2g_matrix = convert_egopose_to_matrix_numpy(e2g_rotation, e2g_translation)
        l2e_matrix = convert_egopose_to_matrix_numpy(l2e_rotation, l2e_translation)
        ego_pose = e2g_matrix @ l2e_matrix
        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego_pose=ego_pose,
            ego_pose_inv=ego_pose_inv,
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            
            for cam_type, cam_info in info['cams'].items():
                img_timestamp.append(cam_info['timestamp'] / 1e6)
                cam_data_path = cam_info['data_path']
                # Handle both / and \ path separators (Windows/Linux)
                file_name = os.path.basename(cam_data_path)
                
                # Apply frame drop noise
                if self.drop_frames and self.noise_sensor_type == 'camera':
                    cam_data_path = self._apply_frame_drop(cam_data_path, file_name)
                
                image_paths.append(cam_data_path)
                
                # Get extrinsics (with optional noise)
                if self.extrinsics_noise and file_name in self.noise_camera_data:
                    noise_info = self.noise_camera_data[file_name]['noise']['extrinsics_noise']
                    noise_key = f'{self.extrinsics_noise_type}_noise_sensor2lidar'
                    cam2lidar_r = np.array(noise_info.get(f'{noise_key}_rotation', cam_info['sensor2lidar_rotation']))
                    cam2lidar_t = np.array(noise_info.get(f'{noise_key}_translation', cam_info['sensor2lidar_translation']))
                else:
                    cam2lidar_r = cam_info['sensor2lidar_rotation']
                    cam2lidar_t = cam_info['sensor2lidar_translation']
                
                cam2lidar_rt = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
                lidar2cam_rt = invert_matrix_egopose_numpy(cam2lidar_rt)
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt
                
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)
            
            if not self.test_mode:
                prev_exists = not (index == 0 or self.flag[index - 1] != self.flag[index])
            else:
                prev_exists = None

            input_dict.update(dict(
                img_timestamp=img_timestamp,
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                prev_exists=prev_exists,
            ))
        
        if not self.test_mode:
            annos = self.get_ann_info(index)
            annos.update(dict(
                bboxes=info['bboxes2d'],
                labels=info['labels2d'],
                centers2d=info['centers2d'],
                depths=info['depths'],
                bboxes_ignore=info['bboxes_ignore'])
            )
            input_dict['ann_info'] = annos
            
        return input_dict
    
    def _apply_frame_drop(self, cam_data_path, file_name):
        """Apply frame drop noise by replacing image path with previous frame."""
        if file_name not in self.noise_camera_data:
            return cam_data_path
        
        noise_info = self.noise_camera_data[file_name]['noise']['drop_frames']
        if self.drop_ratio not in noise_info:
            return cam_data_path
        
        drop_info = noise_info[self.drop_ratio][self.drop_type]
        if drop_info['stuck'] and drop_info['replace']:
            replace_file = drop_info['replace']
            cam_data_path = cam_data_path.replace(file_name, replace_file)
        
        return cam_data_path
