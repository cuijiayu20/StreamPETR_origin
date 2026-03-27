import subprocess
import os
import sys

def main():
    base_cmd = [
        sys.executable, "robust_test/robust_test.py",
        "projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e_mini.py",
        "data/ckpts/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth",
        "--noise-pkl", "data/nuscenes/nuscenes_infos_val_with_noise.pkl"
    ]
    
    tasks = [
        {
            "name": "Clean (正常)",
            "args": ["--noise-type", "clean", "--gpu-id", "0", "--res-dir", "robust_test/results/clean"],
            "log": "robust_test/results/log_clean.txt"
        },
        {
            "name": "Frame Drop 50% (50%丢帧)",
            "args": ["--noise-type", "frame_drop", "--drop-ratio", "50", "--drop-mode", "discrete", "--gpu-id", "1", "--res-dir", "robust_test/results/frame_drop_50"],
            "log": "robust_test/results/log_frame_drop.txt"
        },
        {
            "name": "Extrinsics L2 (外参L2)",
            "args": ["--noise-type", "extrinsics", "--extrinsics-level", "L2", "--gpu-id", "2", "--res-dir", "robust_test/results/extrinsics_L2"],
            "log": "robust_test/results/log_extrinsics.txt"
        },
        {
            "name": "Mask S2 (遮挡S2)",
            "args": ["--noise-type", "mask", "--alpha-exp", "2.0", "--gpu-id", "3", "--res-dir", "robust_test/results/mask_S2"],
            "log": "robust_test/results/log_mask.txt"
        }
    ]
    
    os.makedirs("robust_test/results", exist_ok=True)
    
    processes = []
    
    print("Starting 4 testing tasks in parallel...")
    for task in tasks:
        cmd = base_cmd + task["args"]
        print(f"Launching {task['name']} on GPU {task['args'][task['args'].index('--gpu-id')+1]}... Log: {task['log']}")
        f_out = open(task['log'], 'w')
        p = subprocess.Popen(cmd, stdout=f_out, stderr=subprocess.STDOUT)
        processes.append((p, task, f_out))
        
    for p, task, f_out in processes:
        p.wait()
        f_out.close()
        print(f"[{task['name']}] Finished with return code: {p.returncode}")
        
    print("All tasks completed!")

if __name__ == '__main__':
    main()
