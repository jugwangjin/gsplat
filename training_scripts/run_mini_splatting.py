import os
import json
import signal
import sys
from typing import List, Dict

def read_num_gaussians(result_dir: str) -> int:
    """val_step29999.json에서 가우시안 수를 읽습니다"""
    json_path = os.path.join(result_dir, 'stats', 'val_step29999.json')
    try:
        with open(json_path, 'r') as f:
            stats = json.load(f)
            return stats['num_GS']
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None

def check_experiment_completed(result_dir: str) -> bool:
    """실험이 완료되었는지 확인합니다"""
    checkpoint_path = os.path.join(result_dir, 'ckpts', 'ckpt_29999_rank0.pt')
    return os.path.exists(checkpoint_path)

def signal_handler(signum, frame):
    """Handle keyboard interrupt signal"""
    print("\n키보드 인터럽트를 받았습니다. 즉시 종료합니다...")
    sys.exit(1)

def main():
    # Set up signal handler for keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Mini-Splatting 타겟 가우시안 수 (단위: 백만)
        target_gaussians = {
            "room": 0.39,
            "counter": 0.41,
            "kitchen": 0.43,
            "bonsai": 0.36,
            "treehill": 0.57,
            "bicycle": 0.53,
            "flowers": 0.57,
            "garden": 0.56,
            "stump": 0.61,
        }

        # 데이터셋별 data_factor 설정
        data_factors = {
            "bicycle": 4,
            "flowers": 4,
            "garden": 4,
            "stump": 4,
            "treehill": 4,
            "room": 2,
            "counter": 2,
            "kitchen": 2,
            "bonsai": 2
        }

        # 공통 파라미터
        eval_steps = "14999 15001 19999 20001 30000"
        base_data_dir = "/Bean/data/gwangjin/2025/kdgs/360_v2"
        base_result_dir = "/Bean/log/gwangjin/2025/kdgs/mini_splatting_comparison_nosample_keep_moredepth"
        gpu_id = "2"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        for dataset, target_num in target_gaussians.items():
            print(f"\n시작: {dataset} (목표 가우시안 수: {target_num}M)")
            
            # 결과 디렉토리 설정
            result_dir = f"{base_result_dir}/{dataset}_mini"
            
            if check_experiment_completed(result_dir):
                print(f"{dataset}의 학습이 이미 완료되었습니다. 건너뜁니다...")
                continue

            # Mini-Splatting 명령어 실행
            mini_command = f"python ms_trainer_mesh_simp_no_reinit.py msd \
                --data_dir {base_data_dir}/{dataset} \
                --data_factor {data_factors[dataset]} \
                --result_dir {result_dir} \
                --eval_steps {eval_steps} \
                --simplification_iterations 10 \
                --target_num_gaussians {int(target_num * 1e6)} \
                --simplification_num 3 \
                --depth_reinit_iters 2500 5000 7500 10000 12500 \
                "
            
            print("\nMini-Splatting 명령어 실행:")
            print(mini_command)
            
            # Run command and check its return code
            return_code = os.system(mini_command)
            if return_code != 0:
                print(f"\n명령어 실행 중 오류가 발생했습니다. 종료 코드: {return_code}")
                sys.exit(1)

            print(f"\n{dataset} 실험 완료")
            print("-" * 40)

        print("\n모든 실험이 완료되었습니다!")

    except KeyboardInterrupt:
        print("\n키보드 인터럽트를 받았습니다. 즉시 종료합니다...")
        sys.exit(1)
    except Exception as e:
        print(f"\n예상치 못한 오류가 발생했습니다: {str(e)}")
        print("즉시 종료합니다...")
        sys.exit(1)

if __name__ == "__main__":
    main() 