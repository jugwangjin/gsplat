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
STUDENT_DIR = os.path.join(result_dir_root, 'testing_KT')

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
    # Define lambdas and gradient keys.
    commands = []


    # if True or not os.path.exists('/Bean/log/gwangjin/2025/kdgs/ms/bicycle_depth_reinit_sampling_0.5/ckpts/ckpt_29999_rank0.pt'):
    #     commands.append("python ms_trainer.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/ms/bicycle_depth_reinit_sampling_0.5 --use_depth_reinit --sampling_factor 0.5 --disable_viewer")

    # Make teachers for datasets

    # list dataset names 
    # 
    DATASET_DIR = '/Bean/data/gwangjin/2025/kdgs/360_v2'
    dataset_names = os.listdir(DATASET_DIR)
    dataset_names = [d for d in dataset_names if os.path.isdir(os.path.join(DATASET_DIR, d))]

    TEACHERS_BASE_DIR = '/Bean/log/gwangjin/2025/kdgs/teachers'
    

    # dense teachers
    for dataset_name in dataset_names:
        command = f"python ms_d_trainer.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/{dataset_name} --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/teachers/{dataset_name}_4_dense --use_depth_reinit --disable_viewer"
        if not os.path.exists(f'/Bean/log/gwangjin/2025/kdgs/teachers/{dataset_name}_4_dense/ckpts/ckpt_29999_rank0.pt'):
            commands.append(command)
        
    # small teachers

    for dataset_name in dataset_names:
        command = f"python ms_trainer.py ms --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/{dataset_name} --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/teachers/{dataset_name}_4_small --use_depth_reinit --disable_viewer"
        if not os.path.exists(f'/Bean/log/gwangjin/2025/kdgs/teachers/{dataset_name}_4_small/ckpts/ckpt_29999_rank0.pt'):
            commands.append(command)

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