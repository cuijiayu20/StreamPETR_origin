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
            "args": ["--noise-type", "clean", "--gpu-id", "0", "--res-dir", "our_clean"],
            "log": "our_clean.log"
        },
        {
            "name": "Frame Drop 50% (50%丢帧)",
            "args": ["--noise-type", "frame_drop", "--drop-ratio", "50", "--drop-mode", "discrete", "--gpu-id", "1", "--res-dir", "our_framedrop_50"],
            "log": "our_framedrop_50.log"
        },
        {
            "name": "Extrinsics L2 (外参L2)",
            "args": ["--noise-type", "extrinsics", "--extrinsics-level", "L2", "--gpu-id", "2", "--res-dir", "our_extrinsics_L2"],
            "log": "our_extrinsics_L2.log"
        },
        {
            "name": "Mask S2 (遮挡S2)",
            "args": ["--noise-type", "mask", "--alpha-exp", "2.0", "--gpu-id", "3", "--res-dir", "our_occlusion_S2"],
            "log": "our_occlusion_S2.log"
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
