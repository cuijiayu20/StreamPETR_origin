import os
import json
import argparse
import mmcv

def parse_args():
    parser = argparse.ArgumentParser(description='Export visualization data to JSON')
    parser.add_argument('--tokens', type=str, nargs='*', default=None, help='List of sample tokens to export')
    parser.add_argument('--info-pkl', type=str, default='data/nuscenes/nuscenes_infos_val_with_noise.pkl')
    parser.add_argument('--clean-json', type=str, default='our_clean/results_nusc.json')
    parser.add_argument('--frame-drop-json', type=str, default='our_framedrop_50/results_nusc.json')
    parser.add_argument('--extrinsic-json', type=str, default='our_extrinsics_L2/results_nusc.json')
    parser.add_argument('--mask-json', type=str, default='our_occlusion_S2/results_nusc.json')
    parser.add_argument('--score-thr', type=float, default=0.25)
    parser.add_argument('--out-json', type=str, default='compare_results.json')
    return parser.parse_args()

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def format_box_dict(box):
    return {
        "translation": [float(x) for x in box['translation']],
        "size": [float(x) for x in box['size']],
        "rotation": [float(x) for x in box['rotation']],
        "velocity": [float(x) for x in box.get('velocity', [0.0, 0.0])],
        "name": box['detection_name'],
        "score": float(box.get('detection_score', 1.0))
    }

def get_gt_boxes_for_token(info_list, token):
    for info in info_list:
        if info.get('token') == token:
            gt_boxes = info.get('gt_boxes', [])
            gt_names = info.get('gt_names', [])
            gt_list = []
            for box, name in zip(gt_boxes, gt_names):
                gt_list.append({
                    "translation": [float(x) for x in box[0:3]],
                    "size": [float(x) for x in box[3:6]],
                    "rotation": [float(x) for x in box[6:7]] if len(box)>6 else [0.0],
                    "velocity": [float(x) for x in box[7:9]] if len(box)>8 else [0.0, 0.0],
                    "name": name,
                    "score": 1.0
                })
            return gt_list
    return []

def main():
    args = parse_args()
    
    results = {
        'clean': load_json(args.clean_json),
        'frame_drop_50': load_json(args.frame_drop_json),
        'extrinsics_L2': load_json(args.extrinsic_json),
        'mask_S2': load_json(args.mask_json)
    }
    
    info_list = []
    if os.path.exists(args.info_pkl):
        infos = mmcv.load(args.info_pkl)
        info_list = infos.get('infos', infos)
    else:
        print(f"Warning: GT pkl not found at {args.info_pkl}.")

    valid_data = [d for d in results.values() if d is not None]
    if not valid_data:
        print("Error: No JSON data loaded. Please check folder paths.")
        return

    tokens_to_process = args.tokens
    if not tokens_to_process:
        # Default top 10 tokens from the first valid JSON
        all_tokens = list(valid_data[0]['results'].keys())
        tokens_to_process = all_tokens[:10]
        print(f"No explicit tokens given. Exporting {len(tokens_to_process)} tokens to JSON.")
        
    output_data = {}
    
    for token in tokens_to_process:
        token_data = {}
        token_data['gt'] = get_gt_boxes_for_token(info_list, token)
        
        for name, data in results.items():
            if data is None:
                token_data[name] = []
                continue
                
            preds = data['results'].get(token, [])
            preds = [p for p in preds if p.get('detection_score', 0) >= args.score_thr]
            token_data[name] = [format_box_dict(p) for p in preds]
            
        output_data[token] = token_data
        
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Saved JSON comparison results to {args.out_json}")

if __name__ == '__main__':
    main()
