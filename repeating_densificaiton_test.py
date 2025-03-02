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
max_steps = 15000

# Teacher checkpoints.
FULL_TEACHER_CKPT = '/Bean/log/gwangjin/2025/kdgs/ms_d/bicycle_depth_reinit/ckpts/ckpt_29999_rank0.pt'
SMALL_TEACHER_CKPT = '/Bean/log/gwangjin/2025/kdgs/ms/bicycle_depth_reinit_sampling_0.5/ckpts/ckpt_29999_rank0.pt'

# For saving results.
STUDENT_DIR = os.path.join(result_dir_root, 'repeating_test_v2')

def generate_commands():
    commands = []
    # Now target sampling includes 0.15, 0.33, and 0.75.
    target_samplings = [0.15, 0.33, 0.75]
    
    # ----- "Once" Experiments -----
    # Full teacher: once experiments (no iterative teacher update)
    full_once_base = (
        "python student_trainer_w_ms_v3.py distill2d"
        f" --teacher_ckpt {FULL_TEACHER_CKPT}"
        f" --data_factor {DATA_FACTOR}"
        f" --data_dir {os.path.join(DATA_DIR, DATASET_NAME)}"
        " --disable_viewer"
        " --strategy.blur_threshold 0.002"
        " --strategy.refine_start_iter 100"
        " --strategy.refine_stop_iter 5000"
        f" --max_steps {max_steps}"
        f" --eval_steps {max_steps}"
        f" --save_steps {max_steps}"
        " --distill_sh_lambda 0"
        " --distill_colors_lambda 0"
        " --distill_depth_lambda 0"
        " --distill_xyzs_lambda 0"
        " --distill_quats_lambda 0"
        " --strategy.sh_coeffs_mult 0"
        " --strategy.depths_mult 0"
        " --strategy.grow_grad2d 0.0002"
        " --strategy.key_for_gradient means2d"
        " --use_novel_view"
        " --strategy.use_blur_split"
    )
    full_once_dir_prefix = f"{DATASET_NAME}_{DATA_FACTOR}_FULL_once"
    for ts in target_samplings:
        # With densification: start = ts*0.75, target = ts.
        cmd = (full_once_base +
               f" --start_sampling_ratio {ts * 0.75}"
               f" --target_sampling_ratio {ts}"
               f" --result_dir {os.path.join(STUDENT_DIR, full_once_dir_prefix + f'_{ts}_densify')}")
        if not os.path.exists(os.path.join(STUDENT_DIR, full_once_dir_prefix + f'_{ts}_densify', 'ckpts', f'ckpt_{max_steps-1}_rank0.pt')):
            print(f"Skipping {full_once_dir_prefix + f'_{ts}_densify'}")
            commands.append(cmd)
        # Without densification: start = ts, target = ts.
        cmd = (full_once_base +
               f" --start_sampling_ratio {ts}"
               f" --target_sampling_ratio {ts}"
               f" --result_dir {os.path.join(STUDENT_DIR, full_once_dir_prefix + f'_{ts}')}")
        if not os.path.exists(os.path.join(STUDENT_DIR, full_once_dir_prefix + f'_{ts}', 'ckpts', f'ckpt_{max_steps-1}_rank0.pt')):
            print(f"Skipping {full_once_dir_prefix + f'_{ts}'}")
            commands.append(cmd)
    
    # Small teacher: once experiments (start from SMALL_TEACHER_CKPT)
    small_once_base = (
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
        " --distill_sh_lambda 0"
        " --distill_colors_lambda 0"
        " --distill_depth_lambda 0"
        " --distill_xyzs_lambda 0"
        " --distill_quats_lambda 0"
        " --strategy.sh_coeffs_mult 0"
        " --strategy.depths_mult 0"
        " --strategy.grow_grad2d 0.0002"
        " --strategy.key_for_gradient means2d"
        " --use_novel_view"
        " --strategy.use_blur_split"
    )
    small_once_dir_prefix = f"{DATASET_NAME}_{DATA_FACTOR}_SMALL_once"
    for ts in target_samplings:
        cmd = (small_once_base +
               f" --start_sampling_ratio {ts * 0.75}"
               f" --target_sampling_ratio {ts}"
               f" --result_dir {os.path.join(STUDENT_DIR, small_once_dir_prefix + f'_{ts}_densify')}")
        if not os.path.exists(os.path.join(STUDENT_DIR, small_once_dir_prefix + f'_{ts}_densify', 'ckpts', f'ckpt_{max_steps-1}_rank0.pt')):
            print(f"Skipping {small_once_dir_prefix + f'_{ts}_densify'}")
            commands.append(cmd)
        cmd = (small_once_base +
               f" --start_sampling_ratio {ts}"
               f" --target_sampling_ratio {ts}"
               f" --result_dir {os.path.join(STUDENT_DIR, small_once_dir_prefix + f'_{ts}')}")
        if not os.path.exists(os.path.join(STUDENT_DIR, small_once_dir_prefix + f'_{ts}', 'ckpts', f'ckpt_{max_steps-1}_rank0.pt')):
            print(f"Skipping {small_once_dir_prefix + f'_{ts}'}")
            commands.append(cmd)
    
    # ----- "Repeated" Experiments -----
    # For repeated experiments, we iterate over runs (e.g. 3 times) and update the teacher checkpoint.
    # We'll use a cube-root variant of each target to vary the sampling ratio.
    each_target_samplings = [v ** (1/3) for v in target_samplings]
    
    # Full teacher repeated experiments.
    full_repeated_base = (
        "python student_trainer_w_ms_v3.py distill2d"
        f" --data_factor {DATA_FACTOR}"
        f" --data_dir {os.path.join(DATA_DIR, DATASET_NAME)}"
        " --disable_viewer"
        " --strategy.blur_threshold 0.002"
        " --strategy.refine_start_iter 100"
        " --strategy.refine_stop_iter 5000"
        f" --max_steps {max_steps}"
        f" --eval_steps {max_steps}"
        f" --save_steps {max_steps}"
        " --distill_sh_lambda 0"
        " --distill_colors_lambda 0"
        " --distill_depth_lambda 0"
        " --distill_xyzs_lambda 0"
        " --distill_quats_lambda 0"
        " --strategy.sh_coeffs_mult 0"
        " --strategy.depths_mult 0"
        " --strategy.grow_grad2d 0.0002"
        " --strategy.key_for_gradient means2d"
        " --use_novel_view"
        " --strategy.use_blur_split"
    )
    full_repeated_dir_prefix = f"{DATASET_NAME}_{DATA_FACTOR}_FULL_repeated"
    teacher_ckpt = FULL_TEACHER_CKPT
    for i in range(3):
        for ts in each_target_samplings:
            cmd = (full_repeated_base +
                   f" --start_sampling_ratio {ts * 0.75}"
                   f" --target_sampling_ratio {ts}"
                   f" --teacher_ckpt {teacher_ckpt}"
                   f" --result_dir {os.path.join(STUDENT_DIR, full_repeated_dir_prefix + f'_{ts}_densify_{i}')}")
            if not os.path.exists(os.path.join(STUDENT_DIR, full_repeated_dir_prefix + f'_{ts}_densify_{i}', 'ckpts', f'ckpt_{max_steps-1}_rank0.pt')):
                print(f"Skipping {full_repeated_dir_prefix + f'_{ts}_densify_{i}'}")
                commands.append(cmd)
            # Update teacher checkpoint from this run's output.
            teacher_ckpt = os.path.join(STUDENT_DIR, full_repeated_dir_prefix + f'_{ts}_densify_{i}', 'ckpts', f'ckpt_{max_steps-1}_rank0.pt')
    teacher_ckpt = FULL_TEACHER_CKPT  # reset for non-densification runs
    for i in range(3):
        for ts in each_target_samplings:
            cmd = (full_repeated_base +
                   f" --start_sampling_ratio {ts}"
                   f" --target_sampling_ratio {ts}"
                   f" --teacher_ckpt {teacher_ckpt}"
                   f" --result_dir {os.path.join(STUDENT_DIR, full_repeated_dir_prefix + f'_{ts}_{i}')}")
            if not os.path.exists(os.path.join(STUDENT_DIR, full_repeated_dir_prefix + f'_{ts}_{i}', 'ckpts', f'ckpt_{max_steps-1}_rank0.pt')):
                print(f"Skipping {full_repeated_dir_prefix + f'_{ts}_{i}'}")
                commands.append(cmd)
            teacher_ckpt = os.path.join(STUDENT_DIR, full_repeated_dir_prefix + f'_{ts}_{i}', 'ckpts', f'ckpt_{max_steps-1}_rank0.pt')
    
    # Small teacher repeated experiments.
    small_repeated_base = (
        "python student_trainer_w_ms_v3.py distill2d"
        f" --data_factor {DATA_FACTOR}"
        f" --data_dir {os.path.join(DATA_DIR, DATASET_NAME)}"
        " --disable_viewer"
        " --strategy.blur_threshold 0.002"
        " --strategy.refine_start_iter 100"
        " --strategy.refine_stop_iter 5000"
        f" --max_steps {max_steps}"
        f" --eval_steps {max_steps}"
        f" --save_steps {max_steps}"
        " --distill_sh_lambda 0"
        " --distill_colors_lambda 0"
        " --distill_depth_lambda 0"
        " --distill_xyzs_lambda 0"
        " --distill_quats_lambda 0"
        " --strategy.sh_coeffs_mult 0"
        " --strategy.depths_mult 0"
        " --strategy.grow_grad2d 0.0002"
        " --strategy.key_for_gradient means2d"
        " --use_novel_view"
        " --strategy.use_blur_split"
    )
    small_repeated_dir_prefix = f"{DATASET_NAME}_{DATA_FACTOR}_SMALL_repeated"
    teacher_ckpt = SMALL_TEACHER_CKPT
    for i in range(3):
        for ts in each_target_samplings:
            cmd = (small_repeated_base +
                   f" --start_sampling_ratio {ts * 0.75}"
                   f" --target_sampling_ratio {ts}"
                   f" --teacher_ckpt {teacher_ckpt}"
                   f" --result_dir {os.path.join(STUDENT_DIR, small_repeated_dir_prefix + f'_{ts}_densify_{i}')}")
            if not os.path.exists(os.path.join(STUDENT_DIR, small_repeated_dir_prefix + f'_{ts}_densify_{i}', 'ckpts', f'ckpt_{max_steps-1}_rank0.pt')):
                print(f"Skipping {small_repeated_dir_prefix + f'_{ts}_densify_{i}'}")
                commands.append(cmd)
            teacher_ckpt = os.path.join(STUDENT_DIR, small_repeated_dir_prefix + f'_{ts}_densify_{i}', 'ckpts', f'ckpt_{max_steps-1}_rank0.pt')
    teacher_ckpt = SMALL_TEACHER_CKPT
    for i in range(3):
        for ts in each_target_samplings:
            cmd = (small_repeated_base +
                   f" --start_sampling_ratio {ts}"
                   f" --target_sampling_ratio {ts}"
                   f" --teacher_ckpt {teacher_ckpt}"
                   f" --result_dir {os.path.join(STUDENT_DIR, small_repeated_dir_prefix + f'_{ts}_{i}')}")
            if not os.path.exists(os.path.join(STUDENT_DIR, small_repeated_dir_prefix + f'_{ts}_{i}', 'ckpts', f'ckpt_{max_steps-1}_rank0.pt')):
                commands.append(cmd)
            teacher_ckpt = os.path.join(STUDENT_DIR, small_repeated_dir_prefix + f'_{ts}_{i}', 'ckpts', f'ckpt_{max_steps-1}_rank0.pt')
    
    return commands

def worker(gpu_id, command_queue):
    while True:
        try:
            command = command_queue.get_nowait()
        except Empty:
            break
        # Extract output directory from the command.
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

    # Split commands into once and repeated groups.
    once_commands = [cmd for cmd in commands if '_repeated_' not in cmd]
    repeated_commands = [cmd for cmd in commands if '_repeated_' in cmd]

    # Optionally, log command options in a CSV.
    all_keys = set()
    parsed_commands = []
    for command in commands:
        parts = command.split(' ')
        options = {}
        for i in range(len(parts)):
            if parts[i].startswith('--'):
                key = parts[i][2:]
                value = parts[i + 1] if (i + 1) < len(parts) and not parts[i + 1].startswith('--') else ""
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

    # Execute once experiments concurrently.
    command_queue = multiprocessing.JoinableQueue()
    for cmd in once_commands:
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

    # Execute repeated experiments sequentially using the first GPU.
    for cmd in repeated_commands:
        print(f"Sequentially running: {cmd}")
        command_with_gpu = f"CUDA_VISIBLE_DEVICES={gpu_ids[0]} " + cmd
        try:
            subprocess.run(command_with_gpu, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Sequential command failed: {cmd}\nError: {e}")

if __name__ == "__main__":
    main()
