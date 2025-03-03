import os
import subprocess
import csv
import multiprocessing
from queue import Empty
import argparse

# Global paths / parameters.
DATA_DIR = '/Bean/data/gwangjin/2025/kdgs/360_v2'
result_dir_root = '/Bean/log/gwangjin/2025/kdgs'
DATA_FACTOR = 4
max_steps = 15000

# Directory where distilled student results will be saved.
STUDENT_DIR = os.path.join(result_dir_root, 'all_datasets_distilled')

def generate_commands():
    commands = []
    
    # List all dataset directories under DATA_DIR.
    dataset_names = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    # Define lists of options.
    use_blur_splits = [True]
    use_novel_views = [True]
    # Each tuple represents (start_sampling_ratio, target_sampling_ratio)
    target_sampling_pairs = [(0.45, 0.6), (0.6, 0.75), (0.75, 0.9)]
    
    # Two different lambda combinations.
    combinations = [
        {"distill_sh_lambda": 0.25, "distill_colors_lambda": 0.25, "distill_depth_lambda": 0.25, "distill_xyzs_lambda": 0.25, "distill_quats_lambda": 0.25},
        {"distill_sh_lambda": 0, "distill_colors_lambda": 0, "distill_depth_lambda": 0, "distill_xyzs_lambda": 0, "distill_quats_lambda": 0},
    ]
    key_for_gradients = ['means2d']
    sh_coeffs_mults = [10]
    grow_grad2ds = [0.0002]
    use_densifications = [True, False]
    
    # Loop over each dataset.
    for dataset_name in dataset_names:
        data_dir = os.path.join(DATA_DIR, dataset_name)
        
        # For each dataset, generate commands for both teacher types.
        for teacher_type in ["full", "small"]:
            if teacher_type == "small":
                teacher_ckpt = f"/Bean/log/gwangjin/2025/kdgs/teachers/{dataset_name}_4_small/ckpts/ckpt_29999_rank0.pt"
                eval_steps_opt = f"--eval_steps {max_steps}"
            elif teacher_type == "full":
                teacher_ckpt = f"/Bean/log/gwangjin/2025/kdgs/teachers/{dataset_name}_4_dense/ckpts/ckpt_29999_rank0.pt"
                eval_steps_opt = f"--eval_steps 1 {max_steps}"
            
            # Build the base command for the current dataset and teacher type.
            base_command = (
                "python student_trainer_w_ms_v3.py distill2d"
                f" --teacher_ckpt {teacher_ckpt}"
                f" --data_factor {DATA_FACTOR}"
                f" --data_dir {data_dir}"
                " --disable_viewer"
                " --strategy.blur_threshold 0.002"
                " --strategy.refine_start_iter 100"
                " --strategy.refine_stop_iter 5000"
                f" --max_steps {max_steps}"
                f" {eval_steps_opt}"
                f" --save_steps {max_steps}"
            )
            
            # Loop over additional options.
            for use_densification in use_densifications:
                for use_blur_split in use_blur_splits:
                    for use_novel_view in use_novel_views:
                        for target in target_sampling_pairs:
                            start_sampling_ratio, target_sampling_ratio = target
                            for comb in combinations:
                                d_sh = comb["distill_sh_lambda"]
                                d_colors = comb["distill_colors_lambda"]
                                d_depth = comb["distill_depth_lambda"]
                                d_xyzs = comb["distill_xyzs_lambda"]
                                d_quats = comb["distill_quats_lambda"]
                                for key in key_for_gradients:
                                    # Filter out invalid options.
                                    if (d_sh == 0) and (key == 'rendered_sh_coeffs' or key == 'depths_and_sh'):
                                        continue
                                    if (d_depth == 0) and key == 'depths_and_sh':
                                        continue
                                    for sh_mult in sh_coeffs_mults:
                                        for grow_base in grow_grad2ds:
                                            # Compute multipliers if lambdas are nonzero.
                                            if d_sh > 0:
                                                sh_coeffs_mult_val = sh_mult / d_sh
                                            else:
                                                sh_coeffs_mult_val = 0
                                            if d_depth > 0 and sh_coeffs_mult_val:
                                                depths_mult_val = sh_coeffs_mult_val / d_depth * 0.5
                                            else:
                                                depths_mult_val = 0
                                            grow_grad2d = grow_base
                                            
                                            cmd = base_command
                                            if use_blur_split:
                                                cmd += " --strategy.use_blur_split"
                                            if use_novel_view:
                                                cmd += " --use_novel_view"
                                            if not use_densification:
                                                # When not using densification, override the refine_stop_iter.
                                                cmd += " --strategy.refine_stop_iter 0"
                                                start_sampling_ratio = target_sampling_ratio
                                            cmd += f" --start_sampling_ratio {start_sampling_ratio}"
                                            cmd += f" --target_sampling_ratio {target_sampling_ratio}"
                                            cmd += (f" --distill_sh_lambda {d_sh}"
                                                    f" --distill_colors_lambda {d_colors}"
                                                    f" --distill_depth_lambda {d_depth}"
                                                    f" --distill_xyzs_lambda {d_xyzs}"
                                                    f" --distill_quats_lambda {d_quats}")
                                            if sh_coeffs_mult_val:
                                                cmd += f" --strategy.sh_coeffs_mult {sh_coeffs_mult_val}"
                                            if depths_mult_val:
                                                cmd += f" --strategy.depths_mult {depths_mult_val}"
                                            cmd += f" --strategy.grow_grad2d {grow_grad2d}"
                                            cmd += f" --strategy.key_for_gradient {key}"
                                            
                                            # Construct an output directory name that captures all options.
                                            output_dir_name = (
                                                f"{dataset_name}_{teacher_type}_{DATA_FACTOR}_SMALL"
                                                f"_blur{use_blur_split}_novel{use_novel_view}_densify{use_densification}"
                                                f"_start{start_sampling_ratio}_target{target_sampling_ratio}"
                                                f"_sh{d_sh}_colors{d_colors}_depth{d_depth}_xyzs{d_xyzs}_quats{d_quats}"
                                                f"_{key}"
                                            )
                                            if sh_coeffs_mult_val:
                                                output_dir_name += f"_shmult{sh_coeffs_mult_val}"
                                            if depths_mult_val:
                                                output_dir_name += f"_depth{depths_mult_val}"
                                            output_dir_name += f"_grow2d{grow_grad2d}"
                                            
                                            output_dir = os.path.join(STUDENT_DIR, output_dir_name)
                                            cmd += f" --result_dir {output_dir}"
                                            
                                            # Only add the command if the final checkpoint file does not exist.
                                            if not os.path.exists(os.path.join(output_dir, 'ckpts', f'ckpt_{max_steps-1}_rank0.pt')):
                                                commands.append(cmd)
    return commands

def worker(gpu_id, command_queue):
    while True:
        try:
            command = command_queue.get_nowait()
        except Empty:
            break

        tokens = command.split()
        if '--result_dir' in tokens:
            idx = tokens.index('--result_dir')
            output_dir = tokens[idx + 1]
        else:
            output_dir = os.path.join(result_dir_root, "default_job")
        
        flag_file = os.path.join(output_dir, "job_running.flag")
        if os.path.exists(flag_file):
            print(f"Skipping command because flag file exists for {output_dir}: {command}")
            command_queue.task_done()
            continue

        os.makedirs(output_dir, exist_ok=True)
        with open(flag_file, "w") as f:
            f.write("Job started.\n")

        command_with_gpu = f"CUDA_VISIBLE_DEVICES={gpu_id} " + command
        print(f"Running on GPU {gpu_id}: {command_with_gpu}")
        try:
            subprocess.run(command_with_gpu, shell=True)
        except Exception as e:
            print(f"Failed on GPU {gpu_id}: {command_with_gpu}\nError: {e}")
        finally:
            if os.path.exists(flag_file):
                os.remove(flag_file)
        command_queue.task_done()

def main():
    commands = generate_commands()

    # Create a CSV file that logs all command options.
    all_keys = set()
    parsed_commands = []
    for command in commands:
        parts = command.split(' ')
        options = {}
        for i in range(len(parts)):
            if parts[i].startswith('--'):
                key = parts[i][2:]
                value = parts[i + 1] if i + 1 < len(parts) and not parts[i + 1].startswith('--') else ""
                options[key] = value
                all_keys.add(key)
        parsed_commands.append(options)

    csv_filename = 'distillation_options_all_datasets.csv'
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
        writer.writeheader()
        for options in parsed_commands:
            complete_options = {key: options.get(key, "") for key in all_keys}
            writer.writerow(complete_options)

    print(f"Total commands: {len(commands)}")

    command_queue = multiprocessing.JoinableQueue()
    for cmd in commands:
        command_queue.put(cmd)

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=int, nargs='+', required=True)
    args = parser.parse_args()
    gpu_ids = args.gpu_ids

    processes = []
    for gpu_id in gpu_ids:
        p = multiprocessing.Process(target=worker, args=(gpu_id, command_queue))
        p.start()
        processes.append(p)

    command_queue.join()
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
