
# CUDA_VISIBLE_DEVICES=3 python ms_trainer_mesh_simp.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/simp_comp/simp_pruning_v1 --target_final_sampling_factor 0.1 --simplification_num 5 
# CUDA_VISIBLE_DEVICES=3 python ms_trainer_mesh_simp.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/simp_comp/simp_pruning_v1_ascending --ascending  --target_final_sampling_factor 0.1 --simplification_num 5 


CUDA_VISIBLE_DEVICES=2 python ms_d_trainer.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/garden/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/simplification_comparison/garden_dense --eval_steps 14900 15001 16001 17001 18001 19001 19999 20001 21001 22001 23001 24001 25001 26001 27001 28001 29001 30000



CUDA_VISIBLE_DEVICES=2 python ms_trainer_mesh_simp.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/garden/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/simplification_comparison/garden_inc_mesh_simp --disable_mean --eval_steps 14900 15001 16001 17001 18001 19001 19999 20001 21001 22001 23001 24001 25001 26001 27001 28001 29001 30000

CUDA_VISIBLE_DEVICES=2 python ms_trainer_mesh_simp.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/garden/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/simplification_comparison/garden_inc_mesh_simp_samp_iter10 --disable_mean --eval_steps 14900 15001 16001 17001 18001 19001 19999 20001 21001 22001 23001 24001 25001 26001 27001 28001 29001 30000 --sampling --simplification_iterations 10


CUDA_VISIBLE_DEVICES=2 python ms_trainer.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/garden/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/simplification_comparison/garden --eval_steps 14900 15001 16001 17001 18001 19001 19999 20001 21001 22001 23001 24001 25001 26001 27001 28001 29001 30000
