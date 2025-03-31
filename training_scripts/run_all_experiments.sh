#!/bin/bash

# 데이터셋 목록
datasets=(
    "bonsai"
    "bicycle"
    "counter"
    "flowers"
    "garden"
    "kitchen"
    "room"
    "stump"
    "treehill"
)

# 공통 파라미터
eval_steps="14900 15001 16001 17001 18001 19001 19999 20001 21001 22001 23001 24001 25001 26001 27001 28001 29001 30000"
data_factor=4
base_data_dir="/Bean/data/gwangjin/2025/kdgs/360_v2"
base_result_dir="/Bean/log/gwangjin/2025/kdgs/simplification_comparison_dif"

# 각 데이터셋에 대해 실험 실행
for dataset in "${datasets[@]}"; do
    echo "Starting experiments for ${dataset}..."
    
    # 1. Mesh simplification without reinitialization
    CUDA_VISIBLE_DEVICES=2 python ms_trainer_mesh_simp_no_reinit.py msd \
        --data_dir "${base_data_dir}/${dataset}" \
        --data_factor ${data_factor} \
        --result_dir "${base_result_dir}/${dataset}_render_diff" \
        --eval_steps ${eval_steps} \
        --sampling \
        --simplification_iterations 10 \
        --target_num_gaussians 500000 \
        --simplification_num 3

    # 2. Standard MS training
    CUDA_VISIBLE_DEVICES=2 python ms_trainer.py msd \
        --data_dir "${base_data_dir}/${dataset}" \
        --data_factor ${data_factor} \
        --result_dir "${base_result_dir}/${dataset}_ms" \
        --eval_steps ${eval_steps}

    # 3. MS-D training
    CUDA_VISIBLE_DEVICES=2 python ms_d_trainer.py msd \
        --data_dir "${base_data_dir}/${dataset}" \
        --data_factor ${data_factor} \
        --result_dir "${base_result_dir}/${dataset}_ms" \
        --eval_steps ${eval_steps}

    echo "Completed experiments for ${dataset}"
    echo "----------------------------------------"
done

echo "All experiments completed!" 