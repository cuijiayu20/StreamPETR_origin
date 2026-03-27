import os
import json
import argparse
import mmcv
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Print visualization text for 4 robustness conditions')
    parser.add_argument('--token', type=str, default=None, help='Sample token to print. If None, picks the first one.')
    parser.add_argument('--info-pkl', type=str, default='data/nuscenes/nuscenes_infos_val_with_noise.pkl', help='Path to GT pkl info')
    parser.add_argument('--clean-json', type=str, default='robust_test/results/clean/pts_bbox/results_nusc.json')
    parser.add_argument('--frame-drop-json', type=str, default='robust_test/results/frame_drop_50/pts_bbox/results_nusc.json')
    parser.add_argument('--extrinsic-json', type=str, default='robust_test/results/extrinsics_L2/pts_bbox/results_nusc.json')
    parser.add_argument('--mask-json', type=str, default='robust_test/results/mask_S2/pts_bbox/results_nusc.json')
    parser.add_argument('--score-thr', type=float, default=0.25, help='Score threshold for printing predictions')
    return parser.parse_args()

def load_json(path):
    if not os.path.exists(path):
        # Fallback in case pts_bbox suffix is different
        fallback = path.replace('/pts_bbox/results_nusc.json', '.json')
        if os.path.exists(fallback):
            path = fallback
        else:
            return None
    with open(path, 'r') as f:
        return json.load(f)

def format_box(box):
    return f"[{box['translation'][0]:.2f}, {box['translation'][1]:.2f}, {box['translation'][2]:.2f}, w:{box['size'][0]:.2f}, l:{box['size'][1]:.2f}, h:{box['size'][2]:.2f}, score:{box['detection_score']:>4.2f}]"

def main():
    args = parse_args()
    
    # Load predictions
    results = {
        'Clean (正常)': load_json(args.clean_json),
        'Frame Drop 50% (50%丢帧)': load_json(args.frame_drop_json),
        'Extrinsics L2 (外参L2)': load_json(args.extrinsic_json),
        'Mask S2 (遮挡S2)': load_json(args.mask_json)
    }
    
    # Verify loaded predictions
    for name, data in results.items():
        if data is None:
            print(f"Warning: Could not load data for {name}")
    
    # Determine token
    valid_data = [d for d in results.values() if d is not None]
    if not valid_data:
        print("Error: No prediction data loaded. Are the paths correct?")
        return
        
    sample_token = args.token
    if sample_token is None:
        # Pick the first token from available results
        sample_token = list(valid_data[0]['results'].keys())[0]
        print(f"No token specified. Using token: {sample_token}")
    
    print("="*80)
    print(f"Results for Sample Token: {sample_token}")
    print("="*80)
    
    # Load GT
    if os.path.exists(args.info_pkl):
        print("Ground Truth (真实数据):")
        infos = mmcv.load(args.info_pkl)
        info_list = infos['infos'] if 'infos' in infos else infos
        
        target_info = None
        for info in info_list:
            if info.get('token') == sample_token:
                target_info = info
                break
                
        if target_info and 'gt_boxes' in target_info:
            gt_boxes = target_info['gt_boxes']
            gt_names = target_info['gt_names']
            count = 0
            for box, name in zip(gt_boxes, gt_names):
                # Using a very basic format for GT, adapt if structure differs
                # x, y, z, w, l, h, yaw, v_x, v_y
                print(f"  - {name:<10}: [{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, w:{box[3]:.2f}, l:{box[4]:.2f}, h:{box[5]:.2f}]")
                count += 1
            print(f"  Total GT objects: {count}")
        else:
            print("  [No GT found for this token in the provided info pkl]")
    else:
        print(f"Ground Truth (真实数据): [Could not load info pkl from {args.info_pkl}]")
        
    print("-"*80)
        
    for name, data in results.items():
        print(f"{name}:")
        if data is None:
            print("  [File Missing or Failed to Load]")
            continue
            
        preds = data['results'].get(sample_token, [])
        # Filter by threshold
        preds = [p for p in preds if p['detection_score'] >= args.score_thr]
        
        if not preds:
            print("  [No predictions above score threshold]")
        else:
            for p in preds:
                print(f"  - {p['detection_name']:<10}: {format_box(p)}")
            print(f"  Total predicted above {args.score_thr} threshold: {len(preds)}")
        print("-"*80)

if __name__ == '__main__':
    main()
