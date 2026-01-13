# ------------------------------------------------------------------------
# Robust Loading Pipelines for StreamPETR
# Extends image loading with mask occlusion and camera drop capabilities
# ------------------------------------------------------------------------

import os
import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadMultiViewImageWithMask:
    """Load multi-view images with occlusion mask overlay.
    
    Simulates camera lens contamination by overlaying mask images.
    
    Args:
        noise_ann_file (str): Path to noise annotation pkl file.
        mask_dir (str): Directory containing mask images.
        to_float32 (bool): Whether to convert images to float32.
        color_type (str): Color type for image loading.
    """

    def __init__(self, 
                 noise_ann_file='',
                 mask_dir='',
                 to_float32=False, 
                 color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.mask_dir = mask_dir
        
        # Load noise data
        if noise_ann_file and os.path.exists(noise_ann_file):
            noise_data = mmcv.load(noise_ann_file, file_format='pkl')
            self.noise_camera_data = noise_data.get('camera', {})
        else:
            self.noise_camera_data = {}
        
        # Pre-load mask images
        self.masks = {}
        if mask_dir and os.path.exists(mask_dir):
            self._load_masks()
    
    def _load_masks(self):
        """Pre-load all mask images."""
        for i in range(1, 17):  # 16 mask types
            mask_path = os.path.join(self.mask_dir, f'mask_{i}.jpg')
            if os.path.exists(mask_path):
                self.masks[i] = mmcv.imread(mask_path, self.color_type)
    
    def _put_mask_on_img(self, img, mask):
        """Overlay mask on image."""
        h, w = img.shape[:2]
        mask = np.rot90(mask.copy())  # Copy to avoid modifying original
        mask = mmcv.imresize(mask, (w, h), return_scale=False)
        alpha = mask / 255.0
        alpha = np.power(alpha, 3)
        img_with_mask = alpha * img + (1 - alpha) * mask
        return img_with_mask.astype(np.uint8)
    
    def __call__(self, results):
        """Load images and apply mask occlusion."""
        filenames = results['img_filename']
        img_lists = []
        
        for name in filenames:
            single_img = mmcv.imread(name, self.color_type)
            
            # Apply mask if available
            file_key = name.split('/')[-1].split('\\')[-1]  # Handle both / and \ paths
            if file_key in self.noise_camera_data:
                mask_id = self.noise_camera_data[file_key]['noise']['mask_noise'].get('mask_id', 0)
                if mask_id in self.masks:
                    single_img = self._put_mask_on_img(single_img, self.masks[mask_id])
            
            img_lists.append(single_img)
        
        if self.to_float32:
            img_lists = [img.astype(np.float32) for img in img_lists]
        
        results['filename'] = filenames
        results['img'] = img_lists
        results['img_shape'] = [img.shape for img in img_lists]
        results['ori_shape'] = [img.shape for img in img_lists]
        results['pad_shape'] = [img.shape for img in img_lists]
        
        num_channels = 1 if len(img_lists[0].shape) < 3 else img_lists[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']
        
        return results
    
    def __repr__(self):
        return f"{self.__class__.__name__}(to_float32={self.to_float32}, mask_dir='{self.mask_dir}')"


@PIPELINES.register_module()
class LoadMultiViewImageWithDrop:
    """Load multi-view images with camera drop simulation.
    
    Simulates camera failure by replacing images with black (zero) images.
    
    Args:
        drop_cameras (list): List of camera names to drop, e.g., ['CAM_FRONT'].
        to_float32 (bool): Whether to convert images to float32.
        color_type (str): Color type for image loading.
    """
    
    CAM_ORDER = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'
    ]

    def __init__(self, 
                 drop_cameras=None,
                 to_float32=False, 
                 color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.drop_cameras = drop_cameras or []
        
        # Map camera names to indices
        self.drop_indices = set()
        for cam in self.drop_cameras:
            if cam in self.CAM_ORDER:
                self.drop_indices.add(self.CAM_ORDER.index(cam))
    
    def __call__(self, results):
        """Load images and apply camera drop."""
        filenames = results['img_filename']
        img_lists = []
        
        for i, name in enumerate(filenames):
            single_img = mmcv.imread(name, self.color_type)
            
            # Replace with black image if camera is dropped
            if i in self.drop_indices:
                single_img = np.zeros_like(single_img)
            
            img_lists.append(single_img)
        
        if self.to_float32:
            img_lists = [img.astype(np.float32) for img in img_lists]
        
        results['filename'] = filenames
        results['img'] = img_lists
        results['img_shape'] = [img.shape for img in img_lists]
        results['ori_shape'] = [img.shape for img in img_lists]
        results['pad_shape'] = [img.shape for img in img_lists]
        
        num_channels = 1 if len(img_lists[0].shape) < 3 else img_lists[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']
        
        return results
    
    def __repr__(self):
        return f"{self.__class__.__name__}(drop_cameras={self.drop_cameras})"
