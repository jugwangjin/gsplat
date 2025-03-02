CUDA_VISIBLE_DEVICES=0 python ms_d_trainer.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/ms_d/bicycle_depth_reinit --use_depth_reinit --disable_viewer

python ms_with_distillation_test.py --gpu_ids 0 3