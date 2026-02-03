#!/usr/bin/env python
# ------------------------------------------------------------------------
# Robust Test Script for StreamPETR
# Run robustness benchmarks without modifying original code
# ------------------------------------------------------------------------
# Usage:
#   python robust_test/robust_test.py \
#       projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.py \
#       data/ckpts/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth \
#       --noise-type frame_drop \
#       --noise-pkl data/nuscenes/nuscenes_infos_val_with_noise.pkl \
#       --drop-ratio 30 --drop-mode discrete
#
#   Note: The config uses nuscenes2d_temporal_infos_val.pkl as ann_file,
#         and noise_pkl provides additional noise annotations.
# ------------------------------------------------------------------------

import argparse
import os
import sys
import copy
import json
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import mmcv
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

# Import robust modules to register them
from robust_test.datasets import RobustNuScenesDataset
from robust_test.pipelines import LoadMultiViewImageWithMask, LoadMultiViewImageWithDrop


NOISE_TYPES = ['clean', 'frame_drop', 'extrinsics', 'mask', 'camera_drop']
CAMERA_NAMES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# Extrinsics noise level definitions
# Level: (rotation_deg, translation_cm, description)
EXTRINSICS_LEVELS = {
    'L0': (0.0, 0.0, 'Clean baseline'),
    'L1': (0.5, 0.3, 'Standard online calibration error'),
    'L2': (1.0, 0.5, 'Mild drift'),
    'L3': (2.0, 1.0, 'Moderate drift'),
    'L4': (5.0, 2.0, 'Severe misalignment'),
}


def parse_args():
    parser = argparse.ArgumentParser(description='StreamPETR Robustness Test')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('checkpoint', help='Path to checkpoint file')
    parser.add_argument('--noise-type', type=str, default='clean',
                        choices=NOISE_TYPES,
                        help='Type of noise to apply')
    parser.add_argument('--noise-pkl', type=str, default='',
                        help='Path to noise annotation pkl file')
    parser.add_argument('--mask-dir', type=str, default='robust_benchmark/Occlusion_mask',
                        help='Directory containing mask images')
    
    # Frame drop options
    parser.add_argument('--drop-ratio', type=int, default=30,
                        choices=[10, 20, 30, 40, 50, 60, 70, 80, 90],
                        help='Frame drop ratio (%%)')
    parser.add_argument('--drop-mode', type=str, default='discrete',
                        choices=['discrete', 'consecutive'],
                        help='Frame drop mode')
    
    # Extrinsics noise options
    parser.add_argument('--extrinsics-type', type=str, default='single',
                        choices=['single', 'all'],
                        help='Extrinsics noise type')
    
    # Camera drop options
    parser.add_argument('--drop-cameras', type=str, nargs='+', default=[],
                        help='Cameras to drop, e.g., CAM_FRONT CAM_BACK')
    
    # Batch test mode
    parser.add_argument('--batch', action='store_true',
                        help='Run batch tests for all noise types')
    parser.add_argument('--batch-extrinsics', action='store_true',
                        help='Run batch tests for all extrinsics noise levels (L0-L4)')
    parser.add_argument('--extrinsics-level', type=str, default=None,
                        choices=['L0', 'L1', 'L2', 'L3', 'L4', 'all'],
                        help='Extrinsics noise level (L0-L4 or all)')
    parser.add_argument('--noise-dir', type=str, default='data/nuscenes/extrinsics_levels',
                        help='Directory containing level-specific noise PKL files')
    parser.add_argument('--output-dir', type=str, default='robust_test/results',
                        help='Output directory for batch test results')
    
    # General options
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    parser.add_argument('--eval', type=str, nargs='+', default=['bbox'],
                        help='Evaluation metrics')
    
    return parser.parse_args()


def modify_config_for_noise(cfg, args):
    """Modify config to inject noise settings."""
    cfg = copy.deepcopy(cfg)
    
    # Change dataset type to RobustNuScenesDataset
    for split in ['val', 'test']:
        if split in cfg.data:
            cfg.data[split]['type'] = 'RobustNuScenesDataset'
            
            # Add noise parameters
            cfg.data[split]['noise_ann_file'] = args.noise_pkl
            cfg.data[split]['drop_frames'] = (args.noise_type == 'frame_drop')
            cfg.data[split]['drop_ratio'] = args.drop_ratio
            cfg.data[split]['drop_type'] = args.drop_mode
            cfg.data[split]['extrinsics_noise'] = (args.noise_type == 'extrinsics')
            cfg.data[split]['extrinsics_noise_type'] = args.extrinsics_type
            
            # Modify pipeline for mask and camera drop
            if args.noise_type == 'mask':
                cfg.data[split]['pipeline'] = modify_pipeline_for_mask(
                    cfg.data[split]['pipeline'], args)
            elif args.noise_type == 'camera_drop':
                cfg.data[split]['pipeline'] = modify_pipeline_for_camera_drop(
                    cfg.data[split]['pipeline'], args)
    
    return cfg


def modify_pipeline_for_mask(pipeline, args):
    """Replace LoadMultiViewImageFromFiles with LoadMultiViewImageWithMask."""
    new_pipeline = []
    for step in pipeline:
        if step['type'] == 'LoadMultiViewImageFromFiles':
            new_pipeline.append(dict(
                type='LoadMultiViewImageWithMask',
                noise_ann_file=args.noise_pkl,
                mask_dir=args.mask_dir,
                to_float32=step.get('to_float32', False),
            ))
        elif step['type'] == 'MultiScaleFlipAug3D':
            # Recursively process nested transforms
            new_step = copy.deepcopy(step)
            if 'transforms' in new_step:
                new_step['transforms'] = modify_pipeline_for_mask(new_step['transforms'], args)
            new_pipeline.append(new_step)
        else:
            new_pipeline.append(step)
    return new_pipeline


def modify_pipeline_for_camera_drop(pipeline, args):
    """Replace LoadMultiViewImageFromFiles with LoadMultiViewImageWithDrop."""
    new_pipeline = []
    for step in pipeline:
        if step['type'] == 'LoadMultiViewImageFromFiles':
            new_pipeline.append(dict(
                type='LoadMultiViewImageWithDrop',
                drop_cameras=args.drop_cameras,
                to_float32=step.get('to_float32', False),
            ))
        elif step['type'] == 'MultiScaleFlipAug3D':
            # Recursively process nested transforms
            new_step = copy.deepcopy(step)
            if 'transforms' in new_step:
                new_step['transforms'] = modify_pipeline_for_camera_drop(new_step['transforms'], args)
            new_pipeline.append(new_step)
        else:
            new_pipeline.append(step)
    return new_pipeline


def run_single_test(cfg, checkpoint, args):
    """Run a single robustness test."""
    import torch
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import wrap_fp16_model
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    
    # Set device
    torch.cuda.set_device(args.gpu_id)
    
    # Modify config for noise
    cfg = modify_config_for_noise(cfg, args)
    
    # Set test mode
    cfg.data.test.test_mode = True
    
    # Build dataset
    dataset = build_dataset(cfg.data.test)
    
    # Build dataloader (use custom builder from StreamPETR)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.get('nonshuffler_sampler', dict(type='DistributedSampler')),
    )
    
    # Build model
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # Handle FP16
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    checkpoint_data = load_checkpoint(model, checkpoint, map_location='cpu')
    
    # Set CLASSES
    if 'CLASSES' in checkpoint_data.get('meta', {}):
        model.CLASSES = checkpoint_data['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    
    # Wrap with MMDataParallel
    model = MMDataParallel(model, device_ids=[args.gpu_id])
    model.eval()
    
    # Run test (custom single GPU test)
    outputs = custom_single_gpu_test(model, data_loader)
    
    # Evaluate
    eval_results = dataset.evaluate(outputs, metric=args.eval)
    
    return eval_results


def custom_single_gpu_test(model, data_loader):
    """Custom single GPU test for StreamPETR."""
    import mmcv
    
    model.eval()
    bbox_results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            
            # Handle result format
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    batch_size = len(result['bbox_results'])
                    bbox_results.extend(bbox_result)
                else:
                    batch_size = 1
                    bbox_results.append(result)
            else:
                batch_size = len(result)
                bbox_results.extend(result)
        
        for _ in range(batch_size):
            prog_bar.update()
    
    return bbox_results


def run_batch_test(cfg, checkpoint, args):
    """Run batch tests for multiple noise configurations."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Test configurations
    test_configs = [
        # Clean test (0% frame drop)
        {'noise_type': 'clean', 'name': 'clean_0'},
        
        # Frame drop tests: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%
        *[{'noise_type': 'frame_drop', 'drop_ratio': r, 'drop_mode': 'discrete',
           'name': f'frame_drop_{r}'} for r in [10, 20, 30, 40, 50, 60, 70, 80, 90]],
        
        # Extrinsics noise tests
        {'noise_type': 'extrinsics', 'extrinsics_type': 'single', 'name': 'extrinsics_single'},
        {'noise_type': 'extrinsics', 'extrinsics_type': 'all', 'name': 'extrinsics_all'},
        
        # Mask occlusion test
        {'noise_type': 'mask', 'name': 'mask_occlusion'},
    ]
    
    print(f'Running {len(test_configs)} tests...')
    print('=' * 60)
    
    for i, test_cfg in enumerate(test_configs):
        test_name = test_cfg.pop('name')
        print(f'[{i+1}/{len(test_configs)}] Running test: {test_name}')
        
        # Update args with test config
        test_args = copy.deepcopy(args)
        for key, value in test_cfg.items():
            setattr(test_args, key.replace('-', '_'), value)
        
        try:
            eval_results = run_single_test(cfg, checkpoint, test_args)
            results[test_name] = eval_results
            print(f'  Results: {eval_results}')
        except Exception as e:
            print(f'  Error: {e}')
            results[test_name] = {'error': str(e)}
        
        print('-' * 60)
    
    # Save results
    result_file = os.path.join(args.output_dir, f'batch_results_{timestamp}.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to: {result_file}')
    
    return results


def run_extrinsics_batch_test(cfg, checkpoint, args):
    """Run batch tests for all extrinsics noise levels (L0-L4).
    
    Supports two modes:
    - single: Each camera gets independent noise
    - all: All cameras share the same noise per frame
    """
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Determine which levels to test
    if args.extrinsics_level == 'all' or args.extrinsics_level is None:
        levels_to_test = ['L0', 'L1', 'L2', 'L3', 'L4']
    else:
        levels_to_test = [args.extrinsics_level]
    
    print(f'Extrinsics Type: {args.extrinsics_type}')
    print(f'Levels to test: {levels_to_test}')
    print(f'Noise Dir: {args.noise_dir}')
    print('=' * 60)
    
    # Print level definitions
    print('\nNoise Level Definitions:')
    print(f'{"Level":<6} {"Rot(°)":<10} {"Trans(cm)":<12} {"Description"}')
    print('-' * 50)
    for level in levels_to_test:
        rot, trans, desc = EXTRINSICS_LEVELS[level]
        print(f'{level:<6} {rot:<10} {trans:<12} {desc}')
    print('=' * 60)
    
    for i, level in enumerate(levels_to_test):
        rot_deg, trans_cm, desc = EXTRINSICS_LEVELS[level]
        test_name = f'extrinsics_{args.extrinsics_type}_{level}'
        
        print(f'\n[{i+1}/{len(levels_to_test)}] Testing {test_name}')
        print(f'  Rotation: {rot_deg}°, Translation: {trans_cm}cm ({desc})')
        
        # Construct noise PKL path for this level
        if level == 'L0':
            # L0 uses clean mode (no extrinsics noise)
            noise_pkl = args.noise_pkl
            use_clean = True
        else:
            # Other levels use level-specific PKL files or base PKL
            noise_pkl = os.path.join(args.noise_dir, f'nuscenes_extrinsics_{level}.pkl')
            use_clean = False
        
        print(f'  Noise PKL: {noise_pkl}')
        
        # Check if noise PKL exists
        if not use_clean and not os.path.exists(noise_pkl):
            print(f'  Warning: Noise PKL not found, using base PKL: {args.noise_pkl}')
            noise_pkl = args.noise_pkl
        
        # Create test args
        test_args = copy.deepcopy(args)
        if use_clean:
            # L0: Use clean mode (no extrinsics noise injection)
            test_args.noise_type = 'clean'
        else:
            test_args.noise_type = 'extrinsics'
        test_args.noise_pkl = noise_pkl
        
        try:
            eval_results = run_single_test(cfg, checkpoint, test_args)
            results[test_name] = {
                'level': level,
                'rotation_deg': rot_deg,
                'translation_cm': trans_cm,
                'extrinsics_type': args.extrinsics_type,
                'metrics': eval_results
            }
            
            # Print key metrics
            if 'pts_bbox_NuScenes/NDS' in eval_results:
                nds = eval_results['pts_bbox_NuScenes/NDS']
                mAP = eval_results.get('pts_bbox_NuScenes/mAP', 0)
                print(f'  Results: NDS={nds:.4f}, mAP={mAP:.4f}')
            else:
                print(f'  Results: {eval_results}')
                
        except Exception as e:
            import traceback
            print(f'  Error: {e}')
            traceback.print_exc()
            results[test_name] = {'error': str(e)}
        
        print('-' * 60)
    
    # Save results
    result_file = os.path.join(
        args.output_dir, 
        f'extrinsics_{args.extrinsics_type}_results_{timestamp}.json'
    )
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to: {result_file}')
    
    # Print summary table
    print('\n' + '=' * 60)
    print('Summary: Extrinsics Noise Level Test Results')
    print('=' * 60)
    print(f'{"Level":<6} {"Rot(°)":<8} {"Trans(cm)":<10} {"NDS":<10} {"mAP":<10}')
    print('-' * 50)
    for level in levels_to_test:
        test_name = f'extrinsics_{args.extrinsics_type}_{level}'
        if test_name in results and 'metrics' in results[test_name]:
            metrics = results[test_name]['metrics']
            nds = metrics.get('pts_bbox_NuScenes/NDS', 0)
            mAP = metrics.get('pts_bbox_NuScenes/mAP', 0)
            rot = results[test_name]['rotation_deg']
            trans = results[test_name]['translation_cm']
            print(f'{level:<6} {rot:<8} {trans:<10} {nds:<10.4f} {mAP:<10.4f}')
        else:
            print(f'{level:<6} {"Error":<28}')
    print('=' * 60)
    
    return results

def main():
    args = parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Set plugin
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.replace('/', '.')
            if _module_dir:
                importlib.import_module(_module_dir)
    
    print('=' * 60)
    print('StreamPETR Robustness Test')
    print('=' * 60)
    print(f'Config: {args.config}')
    print(f'Checkpoint: {args.checkpoint}')
    
    if args.batch_extrinsics:
        # Batch extrinsics noise level test
        print('Mode: Batch Extrinsics Noise Level Test')
        results = run_extrinsics_batch_test(cfg, args.checkpoint, args)
    elif args.batch:
        print('Mode: Batch Test')
        print(f'Noise Type: {args.noise_type}')
        results = run_batch_test(cfg, args.checkpoint, args)
    elif args.extrinsics_level is not None:
        # Single extrinsics level test
        print('Mode: Single Extrinsics Level Test')
        print(f'  Extrinsics Level: {args.extrinsics_level}')
        print(f'  Extrinsics Type: {args.extrinsics_type}')
        
        if args.extrinsics_level == 'all':
            # Test all levels
            results = run_extrinsics_batch_test(cfg, args.checkpoint, args)
        else:
            # Test single level - set appropriate noise_pkl
            level = args.extrinsics_level
            if level == 'L0':
                # L0 uses clean mode (no extrinsics noise)
                args.noise_type = 'clean'
            else:
                # Other levels use level-specific PKL
                level_pkl = os.path.join(args.noise_dir, f'nuscenes_extrinsics_{level}.pkl')
                if os.path.exists(level_pkl):
                    args.noise_pkl = level_pkl
                else:
                    print(f'Warning: Level PKL not found: {level_pkl}, using base PKL')
                args.noise_type = 'extrinsics'
            
            results = run_single_test(cfg, args.checkpoint, args)
            print('=' * 60)
            print('Results:')
            for key, value in results.items():
                print(f'  {key}: {value}')
    else:
        print('Mode: Single Test')
        print(f'Noise Type: {args.noise_type}')
        if args.noise_type == 'frame_drop':
            print(f'  Drop Ratio: {args.drop_ratio}%')
            print(f'  Drop Mode: {args.drop_mode}')
        elif args.noise_type == 'extrinsics':
            print(f'  Extrinsics Type: {args.extrinsics_type}')
        elif args.noise_type == 'camera_drop':
            print(f'  Drop Cameras: {args.drop_cameras}')
        
        results = run_single_test(cfg, args.checkpoint, args)
        print('=' * 60)
        print('Results:')
        for key, value in results.items():
            print(f'  {key}: {value}')

    
    print('=' * 60)
    print('Test completed!')
    

if __name__ == '__main__':
    main()
