
# CUDA_VISIBLE_DEVICES=3 python ms_trainer_mesh_simp.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/simp_comp/simp_pruning_v1 --target_final_sampling_factor 0.1 --simplification_num 5 
# CUDA_VISIBLE_DEVICES=3 python ms_trainer_mesh_simp.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/simp_comp/simp_pruning_v1_ascending --ascending  --target_final_sampling_factor 0.1 --simplification_num 5 



CUDA_VISIBLE_DEVICES=3 python ms_trainer_mesh_simp.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/simp_comp/simp_pruning_v4_450k_4 --target_num_gaussians 450000 --simplification_num 4 --disable_mean
CUDA_VISIBLE_DEVICES=3 python ms_trainer_mesh_simp.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/simp_comp/simp_pruning_v4_300k_4 --target_num_gaussians 300000 --simplification_num 4 --disable_mean

CUDA_VISIBLE_DEVICES=3 python ms_trainer_mesh_simp.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/simp_comp/simp_pruning_v4_450k_2 --target_num_gaussians 450000 --simplification_num 2 --disable_mean
CUDA_VISIBLE_DEVICES=3 python ms_trainer_mesh_simp.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/simp_comp/simp_pruning_v4_300k_2 --target_num_gaussians 300000 --simplification_num 2 --disable_mean
# CUDA_VISIBLE_DEVICES=3 python ms_trainer_mesh_simp.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/simp_comp/simp_pruning_v4_450k_6 --target_num_gaussians 450000 --simplification_num 6 --disable_mean
# CUDA_VISIBLE_DEVICES=3 python ms_trainer_mesh_simp.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/simp_comp/simp_pruning_v4_300k_6 --target_final_sampling_factor 300000 --simplification_num 6 --disable_mean

# CUDA_VISIBLE_DEVICES=3 python ms_trainer_mesh_simp.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/simp_comp/simp_pruning_v3_010_10 --target_final_sampling_factor 0.1 --simplification_num 10 --disable_mean
# CUDA_VISIBLE_DEVICES=3 python ms_trainer_mesh_simp.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/simp_comp/simp_pruning_v3_005_10 --target_final_sampling_factor 0.05 --simplification_num 10 --disable_mean
# # 
# CUDA_VISIBLE_DEVICES=3 python ms_trainer_mesh_simp.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/simp_comp/simp_pruning_v3_005_15 --target_final_sampling_factor 0.05 --simplification_num 15 --disable_mean
# CUDA_VISIBLE_DEVICES=3 python ms_trainer_mesh_simp.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/simp_comp/simp_pruning_v3_005_20 --target_final_sampling_factor 0.05 --simplification_num 20 --disable_mean
# CUDA_VISIBLE_DEVICES=3 python ms_trainer_mesh_simp.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/simp_comp/simp_pruning_v1_no_mean_ascending --ascending  --target_final_sampling_factor 0.1 --simplification_num 5 --disable_mean

