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
TEACHER_DIR = os.path.join(result_dir_root, "teachers", f"{DATASET_NAME}_{DATA_FACTOR}")
TEACHER_CKPT = os.path.join(TEACHER_DIR, "ckpts", "ckpt_29999_rank0.pt")
STUDENT_DIR = os.path.join(result_dir_root, 'student_v6')

# Base command for student training.
BASE_COMMAND = (
    "python student_trainer.py distill2d"
    f" --teacher_ckpt {TEACHER_CKPT}"
    f" --data_factor {DATA_FACTOR}"
    f" --data_dir {os.path.join(DATA_DIR, DATASET_NAME)}"
    " --disable_viewer"
    " --apply_vis_on_teacher_sampling"
    " --strategy.blur_threshold 0.0005"
)

def generate_commands():
    # Define lambdas and gradient keys.
    key_for_gradient = ['depths_and_sh', 'rendered_sh_coeffs', 'depths',]
    BASIC_output_dir_name = f"{DATASET_NAME}_{DATA_FACTOR}"
    sh_coeffs_mults = [1.75]

    # Define different combinations of distillation lambdas.
    combinations = [
        [0.5, 0.5, 0.5, 0.5, 0.5],
        # [0.5, 0, 0, 0, 0],
        # [0, 0.5, 0, 0, 0],
        # [0, 0, 0.5, 0, 0],
        # [0, 0, 0, 0.5, 0],
        # [0, 0, 0, 0, 0.5],
        # [0, 0, 0, 0, 0]
    ]
    grow_grad2ds = [0.0002]
    commands = []

    if not os.path.exists('/Bean/log/gwangjin/2025/kdgs/ms_d/bicycle_depth_reinit/ckpts/ckpt_29999_rank0.pt'):
        commands.append("python ms_d_trainer.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/ms_d/bicycle_depth_reinit --use_depth_reinit --disable_viewer")
    if not os.path.exists('/Bean/log/gwangjin/2025/kdgs/ms_d/bicycle/ckpts/ckpt_29999_rank0.pt'):
        commands.append("python ms_d_trainer.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/ms_d/bicycle")
    # return commands
    if not os.path.exists('/Bean/log/gwangjin/2025/kdgs/ms/bicycle_depth_reinit_sampling_0.5/ckpts/ckpt_29999_rank0.pt'):
        commands.append("python ms_trainer.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/ms/bicycle_depth_reinit_sampling_0.5 --use_depth_reinit --sampling_factor 0.5 --disable_viewer")
    if not os.path.exists('/Bean/log/gwangjin/2025/kdgs/ms/bicycle_depth_reinit_sampling_0.1/ckpts/ckpt_29999_rank0.pt'):
        commands.append("python ms_trainer.py msd --data_dir /Bean/data/gwangjin/2025/kdgs/360_v2/bicycle/ --data_factor 4 --result_dir /Bean/log/gwangjin/2025/kdgs/ms/bicycle_depth_reinit_sampling_0.1 --use_depth_reinit --sampling_factor 0.1 --disable_viewer")


    strategy_thresholds = [1e-3]

    use_blur_splits = [True, False]
    use_novel_views = [True, False]  # new outer loop for novel view flag

    # If the teacher checkpoint is missing, schedule teacher training first.
    if not os.path.exists(TEACHER_CKPT):
        teacher_train_cmd = (
            f"python teacher_trainer.py default"
            f" --data_factor {DATA_FACTOR}"
            f" --data_dir {os.path.join(DATA_DIR, DATASET_NAME)}"
            " --disable_viewer"
            f" --result_dir {TEACHER_DIR}"
        )
        commands.append(teacher_train_cmd)

    # Outer loops for blur_split and novel_view.
    for use_blur_split in use_blur_splits:
        for use_novel_view in use_novel_views:
            # Build a modified base command that includes the blur_split and novel view flags.
            base_command = BASE_COMMAND
            if use_blur_split:
                base_command += " --strategy.use_blur_split"
            if use_novel_view:
                base_command += " --use_novel_view"

            # Loop over each combination of distillation lambdas.
            for combination in combinations:
                distill_sh_lambda, distill_colors_lambda, distill_depth_lambda, distill_xyzs_lambda, distill_quats_lambda = combination

                cmd_base = base_command
                cmd_base += f' --distill_colors_lambda {distill_colors_lambda}'
                cmd_base += f' --distill_depth_lambda {distill_depth_lambda}'
                cmd_base += f' --distill_xyzs_lambda {distill_xyzs_lambda}'
                cmd_base += f' --distill_quats_lambda {distill_quats_lambda}'
                cmd_base += f' --distill_sh_lambda {distill_sh_lambda}'

                # Case 1: Both sh and depth lambdas are > 0.
                if distill_sh_lambda > 0 and distill_depth_lambda > 0:
                    for gradient_key in key_for_gradient:
                        for sh_mult in sh_coeffs_mults:
                            sh_mult_val = sh_mult / distill_sh_lambda
                            depth_mult = sh_mult_val * 0.4
                            for grow_grad2d in grow_grad2ds:
                                cmd = cmd_base
                                cmd += f' --strategy.sh_coeffs_mult {sh_mult_val}'
                                cmd += f' --strategy.depths_mult {depth_mult}'
                                cmd += f' --strategy.key_for_gradient {gradient_key}'
                                cmd += f' --strategy.grow_grad2d {grow_grad2d}'
                                output_dir_name = (
                                    BASIC_output_dir_name +
                                    f'_sh{distill_sh_lambda}_colors{distill_colors_lambda}_depth{distill_depth_lambda}_'
                                    f'xyzs{distill_xyzs_lambda}_quats{distill_quats_lambda}_'
                                    f'{gradient_key}_shmult{sh_mult_val}_depth{depth_mult}_grow2d{grow_grad2d}'
                                    f'_blur{use_blur_split}_novelview{use_novel_view}'
                                )
                                output_dir = os.path.join(STUDENT_DIR, output_dir_name)
                                cmd_full = cmd + f' --result_dir {output_dir}'
                                if os.path.exists(os.path.join(output_dir, 'ckpts', 'ckpt_29999_rank0.pt')):
                                    continue
                                commands.append(cmd_full)

                # Case 2: Only sh lambda > 0.
                elif distill_sh_lambda > 0:
                    for gradient_key in key_for_gradient:
                        if gradient_key in ['depths', 'depths_and_sh']:
                            continue
                        for sh_mult in sh_coeffs_mults:
                            sh_mult_val = sh_mult / distill_sh_lambda
                            for grow_grad2d in grow_grad2ds:
                                cmd = cmd_base
                                cmd += f' --strategy.sh_coeffs_mult {sh_mult_val}'
                                cmd += f' --strategy.key_for_gradient {gradient_key}'
                                cmd += f' --strategy.grow_grad2d {grow_grad2d}'
                                output_dir_name = (
                                    BASIC_output_dir_name +
                                    f'_sh{distill_sh_lambda}_colors{distill_colors_lambda}_depth{distill_depth_lambda}_'
                                    f'xyzs{distill_xyzs_lambda}_quats{distill_quats_lambda}_'
                                    f'{gradient_key}_shmult{sh_mult_val}_grow2d{grow_grad2d}'
                                    f'_blur{use_blur_split}_novelview{use_novel_view}'
                                )
                                output_dir = os.path.join(STUDENT_DIR, output_dir_name)
                                cmd_full = cmd + f' --result_dir {output_dir}'
                                if os.path.exists(os.path.join(output_dir, 'ckpts', 'ckpt_29999_rank0.pt')):
                                    continue
                                commands.append(cmd_full)

                # Case 3: Only depth lambda > 0.
                elif distill_depth_lambda > 0:
                    for gradient_key in key_for_gradient:
                        if gradient_key in ['rendered_sh_coeffs', 'depths_and_sh']:
                            continue
                        for sh_mult in sh_coeffs_mults:
                            depth_mult = sh_mult / distill_depth_lambda * 0.4
                            for grow_grad2d in grow_grad2ds:
                                cmd = cmd_base
                                cmd += f' --strategy.depths_mult {depth_mult}'
                                cmd += f' --strategy.key_for_gradient {gradient_key}'
                                cmd += f' --strategy.grow_grad2d {grow_grad2d}'
                                output_dir_name = (
                                    BASIC_output_dir_name +
                                    f'_sh{distill_sh_lambda}_colors{distill_colors_lambda}_depth{distill_depth_lambda}_'
                                    f'xyzs{distill_xyzs_lambda}_quats{distill_quats_lambda}_'
                                    f'{gradient_key}_depth{depth_mult}_grow2d{grow_grad2d}'
                                    f'_blur{use_blur_split}_novelview{use_novel_view}'
                                )
                                output_dir = os.path.join(STUDENT_DIR, output_dir_name)
                                cmd_full = cmd + f' --result_dir {output_dir}'
                                if os.path.exists(os.path.join(output_dir, 'ckpts', 'ckpt_29999_rank0.pt')):
                                    continue
                                commands.append(cmd_full)

                # Case 4: Both lambdas are zero.
                else:
                    for gradient_key in key_for_gradient:
                        for grow in grow_grad2ds:
                            if gradient_key in ['rendered_sh_coeffs', 'depths_and_sh', 'depths']:
                                continue
                            cmd = cmd_base
                            cmd += f' --strategy.key_for_gradient {gradient_key}'
                            cmd += f' --strategy.grow_grad2d {grow}'
                            output_dir_name = (
                                BASIC_output_dir_name +
                                f'_sh{distill_sh_lambda}_colors{distill_colors_lambda}_depth{distill_depth_lambda}_'
                                f'xyzs{distill_xyzs_lambda}_quats{distill_quats_lambda}'
                                f'_{gradient_key}_grow2d{grow}'
                                f'_blur{use_blur_split}_novelview{use_novel_view}'
                            )
                            output_dir = os.path.join(STUDENT_DIR, output_dir_name)
                            cmd_full = cmd + f' --result_dir {output_dir}'
                            if not os.path.exists(os.path.join(output_dir, 'ckpts', 'ckpt_29999_rank0.pt')):
                                commands.append(cmd_full)

    # Additionally, add commands for gradient_key 'means2d'.
    extra_commands = []
    for use_blur_split in use_blur_splits:
        for use_novel_view in use_novel_views:
            base_command = BASE_COMMAND
            if use_blur_split:
                base_command += " --strategy.use_blur_split"
            if use_novel_view:
                base_command += " --use_novel_view"
            for combination in combinations:
                distill_sh_lambda, distill_colors_lambda, distill_depth_lambda, distill_xyzs_lambda, distill_quats_lambda = combination
                cmd_base = base_command
                cmd_base += f' --distill_colors_lambda {distill_colors_lambda}'
                cmd_base += f' --distill_depth_lambda {distill_depth_lambda}'
                cmd_base += f' --distill_xyzs_lambda {distill_xyzs_lambda}'
                cmd_base += f' --distill_quats_lambda {distill_quats_lambda}'
                cmd_base += f' --distill_sh_lambda {distill_sh_lambda}'
                for grow_grad2d in [0.001]:
                    cmd = cmd_base
                    cmd += f' --strategy.key_for_gradient means2d'
                    cmd += f' --strategy.grow_grad2d {grow_grad2d}'
                    output_dir_name = (
                        BASIC_output_dir_name +
                        f'_sh{distill_sh_lambda}_colors{distill_colors_lambda}_depth{distill_depth_lambda}_'
                        f'xyzs{distill_xyzs_lambda}_quats{distill_quats_lambda}_means2d_grow2d{grow_grad2d}'
                        f'_blur{use_blur_split}_novelview{use_novel_view}'
                    )
                    output_dir = os.path.join(STUDENT_DIR, output_dir_name)
                    cmd_full = cmd + f' --result_dir {output_dir}'
                    if not os.path.exists(os.path.join(output_dir, 'ckpts', 'ckpt_29999_rank0.pt')):
                        extra_commands.append(cmd_full)
    commands.extend(extra_commands)

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
