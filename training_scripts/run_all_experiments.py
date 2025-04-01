import os
import json
from typing import List

def read_num_gaussians(result_dir: str) -> int:
    """Read number of gaussians from val_step29999.json"""
    json_path = os.path.join(result_dir, 'stats', 'val_step29999.json')
    try:
        with open(json_path, 'r') as f:
            stats = json.load(f)
            return stats['num_GS']
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None

def check_experiment_completed(result_dir: str) -> bool:
    """Check if the experiment is already completed"""
    os.listdir(os.path.join(result_dir, 'ckpts'))
    checkpoint_path = os.path.join(result_dir, 'ckpts', 'ckpt_29999_rank0.pt')
    return os.path.exists(checkpoint_path)

def main():
    try:
        # 데이터셋 목록
        datasets = [
            "bonsai",
            "bicycle",
            "counter",
            "flowers",
            "garden",
            "kitchen",
            "room",
            "stump",
            "treehill"
        ]

        # 공통 파라미터
        eval_steps = "19999 20001 30000"
        data_factor = 4
        base_data_dir = "/Bean/data/gwangjin/2025/kdgs/360_v2"
        base_result_dir = "/Bean/log/gwangjin/2025/kdgs/simplification_comparison_dif_4"
        gpu_id = "2"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        for dataset in datasets:
            print(f"\nStarting experiments for {dataset}...")
            
            # 1. Standard MS training (먼저 실행)
            ms_result_dir = f"{base_result_dir}/{dataset}_ms"
            if check_experiment_completed(ms_result_dir):
                print(f"MS training for {dataset} already completed, skipping...")
            else:
                ms_command = f"python ms_trainer.py msd \
                    --data_dir {base_data_dir}/{dataset} \
                    --data_factor {data_factor} \
                    --result_dir {ms_result_dir} \
                    --eval_steps {eval_steps} \
                    --strategy.refine_scale2d_stop_iter 15000"
                print("\nExecuting MS training command:")
                print(ms_command)
                os.system(ms_command)

            # Get number of gaussians after MS training
            num_gaussians = read_num_gaussians(ms_result_dir)
            if num_gaussians is None:
                num_gaussians = 500000
            
            print(f"\nNumber of gaussians for {dataset} after MS training: {num_gaussians}")

            # 2. Mesh simplification without reinitialization
            simp_result_dir = f"{base_result_dir}/{dataset}_render_diff"
            if check_experiment_completed(simp_result_dir):
                print(f"Simplification for {dataset} already completed, skipping...")
            else:
                simp_command = f"python ms_trainer_mesh_simp_no_reinit.py msd \
                    --data_dir {base_data_dir}/{dataset} \
                    --data_factor {data_factor} \
                    --result_dir {simp_result_dir} \
                    --eval_steps {eval_steps} \
                    --sampling \
                    --simplification_iterations 10 \
                    --target_num_gaussians {num_gaussians} \
                    --simplification_num 3"
                print("\nExecuting Simplification command:")
                print(simp_command)
                os.system(simp_command)

            # 3. MS-D training
            msd_result_dir = f"{base_result_dir}/{dataset}_ms_d"
            if check_experiment_completed(msd_result_dir):
                print(f"MS-D training for {dataset} already completed, skipping...")
            else:
                msd_command = f"python ms_d_trainer.py msd \
                    --data_dir {base_data_dir}/{dataset} \
                    --data_factor {data_factor} \
                    --result_dir {msd_result_dir} \
                    --eval_steps {eval_steps} \
                    --strategy.refine_scale2d_stop_iter 15000"
                print("\nExecuting MS-D training command:")
                print(msd_command)
                os.system(msd_command)

            # 4. Teacher training
            teacher_result_dir = f"{base_result_dir}/{dataset}_teacher"
            if check_experiment_completed(teacher_result_dir):
                print(f"Teacher training for {dataset} already completed, skipping...")
            else:
                teacher_command = f"python teacher_trainer.py default \
                    --data_dir {base_data_dir}/{dataset} \
                    --data_factor {data_factor} \
                    --result_dir {teacher_result_dir} \
                    --eval_steps {eval_steps}"
                print("\nExecuting Teacher training command:")
                print(teacher_command)
                os.system(teacher_command)

            print(f"\nCompleted experiments for {dataset}")
            print("-" * 40)

        print("\nAll experiments completed!")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting immediately...")
        exit()
    except Exception as e:
        print(f"\nUnexpected error occurred: {str(e)}")
        print("Exiting immediately...")
        exit()

if __name__ == "__main__":
    main() 