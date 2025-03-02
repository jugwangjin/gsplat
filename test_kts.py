import os
import subprocess
import csv
import multiprocessing
from queue import Empty
import argparse

# Global paths / parameters.
DATA_DIR = '/Bean/data/gwangjin/2025/kdgs/360_v2'
DATASET_NAME = 'bicycle'
result_dir_root = '/Bean/log/gwangjin/2025/kdgs'
DATA_FACTOR = 4


# Directories for teacher and student.
FULL_TEACHER_CKPT = '/Bean/log/gwangjin/2025/kdgs/ms_d/bicycle_depth_reinit/ckpts/ckpt_29999_rank0.pt'
SMALL_TEACHER_CKPT = '/Bean/log/gwangjin/2025/kdgs/ms/bicycle_depth_reinit_sampling_0.5/ckpts/ckpt_29999_rank0.pt'
FULL_TEACHER_DIR = os.path.dirname(os.path.dirname(FULL_TEACHER_CKPT))
SMALL_TEACHER_DIR = os.path.dirname(os.path.dirname(SMALL_TEACHER_CKPT))
STUDENT_DIR = os.path.join(result_dir_root, 'testing_kt')

max_steps = 15000

FULL_BASE_COMMAND = (
    "python student_trainer_w_ms_v3.py distill2d"
    f" --teacher_ckpt {FULL_TEACHER_CKPT}"
    f" --data_factor {DATA_FACTOR}"
    f" --data_dir {os.path.join(DATA_DIR, DATASET_NAME)}"
    " --disable_viewer"
    " --strategy.blur_threshold 0.002"
    " --strategy.refine_start_iter 100"
    " --strategy.refine_stop_iter 5000"
    f" --max_steps {max_steps}"
    f" --eval_steps 1 {max_steps}"
    f" --save_steps {max_steps}"
)

SMALL_BASE_COMMAND = (
    "python student_trainer_w_ms_v3.py distill2d"
    f" --teacher_ckpt {SMALL_TEACHER_CKPT}"
    f" --data_factor {DATA_FACTOR}"
    f" --data_dir {os.path.join(DATA_DIR, DATASET_NAME)}"
    " --disable_viewer"
    " --strategy.blur_threshold 0.002"
    " --strategy.refine_start_iter 100"
    " --strategy.refine_stop_iter 5000"
    f" --max_steps {max_steps}"
    f" --eval_steps {max_steps}"
    f" --save_steps {max_steps}"
)



def generate_commands():
    commands = []
    # Define model-specific base commands.
    models = [
        ("SMALL", SMALL_BASE_COMMAND),
        ("FULL", FULL_BASE_COMMAND),
    ]
    
    # Define lambda keys.
    lambda_keys = [
        "distill_sh_lambda", 
        "distill_colors_lambda", 
        "distill_depth_lambda", 
        "distill_xyzs_lambda", 
        "distill_quats_lambda"
    ]
    
    # Create five configurations: one for each lambda set to 0.25 individually.
    lambda_combinations = []
    for key in lambda_keys:
        config = {k: 0 for k in lambda_keys}
        config[key] = 0.25
        lambda_combinations.append(config)
    # Add the all-zero configuration.
    lambda_combinations.append({k: 0 for k in lambda_keys})
    
    # Fixed parameters.
    use_blur_split = True       # Keep as True.
    use_novel_view = True        # Force novel view True.
    use_densification = True     # Use densification.
    target_sampling_pairs = [(0.3, 0.5)]  # Only one pair: start=0.3, target=0.5.
    key_for_gradient = "means2d" # Single key for simplicity.
    
    # Parameters for multipliers (using same defaults as before).
    sh_coeffs_mult = 10
    grow_grad2d = 0.0002
    
    for model_label, base_command in models:
        for start_sampling_ratio, target_sampling_ratio in target_sampling_pairs:
            for lambdas in lambda_combinations:
                d_sh     = lambdas["distill_sh_lambda"]
                d_colors = lambdas["distill_colors_lambda"]
                d_depth  = lambdas["distill_depth_lambda"]
                d_xyzs   = lambdas["distill_xyzs_lambda"]
                d_quats  = lambdas["distill_quats_lambda"]
                
                # Calculate sh_coeffs_mult and depths_mult based on lambdas.
                if d_sh > 0:
                    sh_coeffs_mult_val = sh_coeffs_mult / d_sh
                else:
                    sh_coeffs_mult_val = 0
                if d_depth > 0 and sh_coeffs_mult_val:
                    depths_mult_val = sh_coeffs_mult_val / d_depth * 0.5
                else:
                    depths_mult_val = 0
                
                # Build the command.
                cmd = base_command
                if use_blur_split:
                    cmd += " --strategy.use_blur_split"
                if use_novel_view:
                    cmd += " --use_novel_view"
                if not use_densification:
                    cmd += " --strategy.refine_stop_iter 0"
                    # When densification is disabled, force start_sampling_ratio to be the same as target_sampling_ratio.
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
                cmd += f" --strategy.key_for_gradient {key_for_gradient}"
                
                # Create an output directory name that embeds the configuration.
                output_dir_name = (
                    f"{DATASET_NAME}_{DATA_FACTOR}_{model_label}"
                    f"_blur{use_blur_split}_novel{use_novel_view}_densify{use_densification}"
                    f"_start{start_sampling_ratio}_target{target_sampling_ratio}"
                    f"_sh{d_sh}_colors{d_colors}_depth{d_depth}_xyzs{d_xyzs}_quats{d_quats}"
                    f"_{key_for_gradient}"
                )
                if sh_coeffs_mult_val:
                    output_dir_name += f"_shmult{sh_coeffs_mult_val}"
                if depths_mult_val:
                    output_dir_name += f"_depth{depths_mult_val}"
                output_dir_name += f"_grow2d{grow_grad2d}"
                output_dir = os.path.join(STUDENT_DIR, output_dir_name)
                cmd += f" --result_dir {output_dir}"
                
                commands.append(cmd)
    return commands

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

    csv_filename = 'distillation_options.csv'
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