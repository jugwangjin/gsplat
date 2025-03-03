import os
import subprocess
import csv
import multiprocessing
from queue import Empty
import argparse

# Global paths / parameters.
DATA_DIR = './data/360_v2'
DATASET_NAME = 'bicycle'
result_dir_root = './gsplat_students'
DATA_FACTOR = 4


# Directories for teacher and student.
FULL_TEACHER_CKPT = './gsplat_teachers/bicycle_4_ms/ckpts/ckpt_29999_rank0.pt'
FULL_TEACHER_DIR = os.path.dirname(os.path.dirname(FULL_TEACHER_CKPT))
STUDENT_DIR = os.path.join(result_dir_root, 'kt_test_on_ms2')

max_steps = 10000

FULL_BASE_COMMAND = (
    "python student_trainer_prune_and_densify.py distill2d"
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
)


def generate_commands():
    # Define lambdas and gradient keys.
    commands = []

    # Define lists of options.
    use_blur_splits = [False]
    use_novel_views = [False]
    # Target sampling is given as a pair (start, target). We format it as "start-target".
    target_sampling_pairs = [(0.7, 0.8)]
    
    # Each combination is a dictionary of lambdas.
    combinations = [
        {"distill_sh_lambda": 0, "distill_colors_lambda": 0, "distill_depth_lambda": 0, "distill_xyzs_lambda": 0, "distill_quats_lambda": 0, "distill_scales_lambda": 0, "distill_opacities_lambda": 0},

    ]
    key_for_gradients = ['means2d']
    # Even if there is only one value here, we include it to be explicit.
    sh_coeffs_mults = [10]
    grow_grad2ds = [0.0002]
    # start_sampling_ratio: float = 0.75

    use_densifications = [True, False]   

    # target_sampling_ratio: float = 0.9
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
                        d_scales = comb["distill_scales_lambda"]
                        d_opacities = comb["distill_opacities_lambda"]
                        for key in key_for_gradients:
                            # Filter out an invalid option: if all lambdas are zero and key is 'rendered_sh_coeffs'
                            if (d_sh == 0) and (key == 'rendered_sh_coeffs' or key == 'depths_and_sh'):
                                continue
                            if (d_depth == 0) and key == 'depths_and_sh':
                                continue

                            for sh_mult in sh_coeffs_mults:
                                for grow_base in grow_grad2ds:
                                    # Compute sh_coeffs_mult value if distill_sh_lambda > 0; otherwise set to zero.
                                    if d_sh > 0:
                                        sh_coeffs_mult_val = sh_mult / d_sh
                                    else:
                                        sh_coeffs_mult_val = 0
                                    # Compute depths_mult value if distill_depth_lambda > 0.
                                    if d_depth > 0 and sh_coeffs_mult_val:
                                        depths_mult_val = sh_coeffs_mult_val / d_depth * 0.5
                                    else:
                                        depths_mult_val = 0
                                    # Adjust grow_grad2d based on the sum of lambdas.
                                    grow_grad2d = grow_base

                                    # Build the command by including all options.
                                    cmd = FULL_BASE_COMMAND
                                    if use_blur_split:
                                        cmd += " --strategy.use_blur_split"
                                    if use_novel_view:
                                        cmd += " --use_novel_view"
                                    if not use_densification:
                                        cmd += " --strategy.refine_stop_iter 0"
                                        start_sampling_ratio = target_sampling_ratio
                                    cmd += f" --start_sampling_ratio {start_sampling_ratio}"
                                    cmd += f" --target_sampling_ratio {target_sampling_ratio}"
                                    cmd += (f" --distill_sh_lambda {d_sh}"
                                            f" --distill_colors_lambda {d_colors}"
                                            f" --distill_depth_lambda {d_depth}"
                                            f" --distill_xyzs_lambda {d_xyzs}"
                                            f" --distill_quats_lambda {d_quats}"
                                            f" --distill_scales_lambda {d_scales}"
                                            f" --distill_opacities_lambda {d_opacities}"
                                            )
                                    # Include sh and depth multipliers only if they are nonzero.
                                    if sh_coeffs_mult_val:
                                        cmd += f" --strategy.sh_coeffs_mult {sh_coeffs_mult_val}"
                                    if depths_mult_val:
                                        cmd += f" --strategy.depths_mult {depths_mult_val}"
                                    cmd += f" --strategy.grow_grad2d {grow_grad2d}"
                                    cmd += f" --strategy.key_for_gradient {key}"

                                    # Create an output directory name that embeds all options.
                                    output_dir_name = (
                                        f"{DATASET_NAME}_{DATA_FACTOR}_FULL"
                                        f"_blur{use_blur_split}_novel{use_novel_view}_densify{use_densification}"
                                        f"_start{start_sampling_ratio}_target{target_sampling_ratio}"
                                        f"_sh{d_sh}_colors{d_colors}_depth{d_depth}_xyzs{d_xyzs}_quats{d_quats}_scales{d_scales}_opacities{d_opacities}"
                                        f"_{key}"
                                    )
                                    if sh_coeffs_mult_val:
                                        output_dir_name += f"_shmult{sh_coeffs_mult_val}"
                                    if depths_mult_val:
                                        output_dir_name += f"_depth{depths_mult_val}"
                                    output_dir_name += f"_grow2d{grow_grad2d}"
                                    output_dir = os.path.join(STUDENT_DIR, output_dir_name)
                                    cmd += f" --result_dir {output_dir}"

                                    if d_scales > 0 or d_opacities > 0 or not os.path.exists(os.path.join(output_dir, 'ckpts', f'ckpt_{max_steps-1}_rank0.pt')):
                                        commands.append(cmd)




    # Define lists of options.
    use_blur_splits = [True]
    use_novel_views = [True]
    # Target sampling is given as a pair (start, target). We format it as "start-target".
    target_sampling_pairs = [(0.7, 0.8)]
    
    # Each combination is a dictionary of lambdas.
    combinations = [
        {"distill_sh_lambda": 0.25, "distill_colors_lambda": 0.25, "distill_depth_lambda": 0.25, "distill_xyzs_lambda": 0.25, "distill_quats_lambda": 0.25, "distill_scales_lambda": 0.25, "distill_opacities_lambda": 0.25},
        {"distill_sh_lambda": 0, "distill_colors_lambda": 0, "distill_depth_lambda": 0, "distill_xyzs_lambda": 0, "distill_quats_lambda": 0, "distill_scales_lambda": 0, "distill_opacities_lambda": 0},
        {"distill_sh_lambda": 0.25, "distill_colors_lambda": 0, "distill_depth_lambda": 0, "distill_xyzs_lambda": 0, "distill_quats_lambda": 0, "distill_scales_lambda": 0, "distill_opacities_lambda": 0},
        {"distill_sh_lambda": 0, "distill_colors_lambda": 0.25, "distill_depth_lambda": 0, "distill_xyzs_lambda": 0, "distill_quats_lambda": 0, "distill_scales_lambda": 0, "distill_opacities_lambda": 0},
        {"distill_sh_lambda": 0, "distill_colors_lambda": 0, "distill_depth_lambda": 0.25, "distill_xyzs_lambda": 0, "distill_quats_lambda": 0, "distill_scales_lambda": 0, "distill_opacities_lambda": 0},
        {"distill_sh_lambda": 0, "distill_colors_lambda": 0, "distill_depth_lambda": 0, "distill_xyzs_lambda": 0.25, "distill_quats_lambda": 0, "distill_scales_lambda": 0, "distill_opacities_lambda": 0},
        {"distill_sh_lambda": 0, "distill_colors_lambda": 0, "distill_depth_lambda": 0, "distill_xyzs_lambda": 0, "distill_quats_lambda": 0.25, "distill_scales_lambda": 0, "distill_opacities_lambda": 0},
        {"distill_sh_lambda": 0, "distill_colors_lambda": 0, "distill_depth_lambda": 0, "distill_xyzs_lambda": 0, "distill_quats_lambda": 0, "distill_scales_lambda": 0.25, "distill_opacities_lambda": 0},
        {"distill_sh_lambda": 0, "distill_colors_lambda": 0, "distill_depth_lambda": 0, "distill_xyzs_lambda": 0, "distill_quats_lambda": 0, "distill_scales_lambda": 0, "distill_opacities_lambda": 0.25},
        {"distill_sh_lambda": 0, "distill_colors_lambda": 0.25, "distill_depth_lambda": 0, "distill_xyzs_lambda": 0.25, "distill_quats_lambda": 0, "distill_scales_lambda": 0, "distill_opacities_lambda": 0},

    ]
    key_for_gradients = ['means2d']
    # Even if there is only one value here, we include it to be explicit.
    sh_coeffs_mults = [10]
    grow_grad2ds = [0.0002]
    # start_sampling_ratio: float = 0.75

    use_densifications = [True]   

    # target_sampling_ratio: float = 0.9
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
                        d_scales = comb["distill_scales_lambda"]
                        d_opacities = comb["distill_opacities_lambda"]
                        for key in key_for_gradients:
                            # Filter out an invalid option: if all lambdas are zero and key is 'rendered_sh_coeffs'
                            if (d_sh == 0) and (key == 'rendered_sh_coeffs' or key == 'depths_and_sh'):
                                continue
                            if (d_depth == 0) and key == 'depths_and_sh':
                                continue

                            for sh_mult in sh_coeffs_mults:
                                for grow_base in grow_grad2ds:
                                    # Compute sh_coeffs_mult value if distill_sh_lambda > 0; otherwise set to zero.
                                    if d_sh > 0:
                                        sh_coeffs_mult_val = sh_mult / d_sh
                                    else:
                                        sh_coeffs_mult_val = 0
                                    # Compute depths_mult value if distill_depth_lambda > 0.
                                    if d_depth > 0 and sh_coeffs_mult_val:
                                        depths_mult_val = sh_coeffs_mult_val / d_depth * 0.5
                                    else:
                                        depths_mult_val = 0
                                    # Adjust grow_grad2d based on the sum of lambdas.
                                    grow_grad2d = grow_base

                                    # Build the command by including all options.
                                    cmd = FULL_BASE_COMMAND
                                    if use_blur_split:
                                        cmd += " --strategy.use_blur_split"
                                    if use_novel_view:
                                        cmd += " --use_novel_view"
                                    if not use_densification:
                                        cmd += " --strategy.refine_stop_iter 0"
                                        start_sampling_ratio = target_sampling_ratio
                                    cmd += f" --start_sampling_ratio {start_sampling_ratio}"
                                    cmd += f" --target_sampling_ratio {target_sampling_ratio}"
                                    cmd += (f" --distill_sh_lambda {d_sh}"
                                            f" --distill_colors_lambda {d_colors}"
                                            f" --distill_depth_lambda {d_depth}"
                                            f" --distill_xyzs_lambda {d_xyzs}"
                                            f" --distill_quats_lambda {d_quats}"
                                            f" --distill_scales_lambda {d_scales}"
                                            f" --distill_opacities_lambda {d_opacities}"
                                            )
                                    # Include sh and depth multipliers only if they are nonzero.
                                    if sh_coeffs_mult_val:
                                        cmd += f" --strategy.sh_coeffs_mult {sh_coeffs_mult_val}"
                                    if depths_mult_val:
                                        cmd += f" --strategy.depths_mult {depths_mult_val}"
                                    cmd += f" --strategy.grow_grad2d {grow_grad2d}"
                                    cmd += f" --strategy.key_for_gradient {key}"

                                    # Create an output directory name that embeds all options.
                                    output_dir_name = (
                                        f"{DATASET_NAME}_{DATA_FACTOR}_FULL"
                                        f"_blur{use_blur_split}_novel{use_novel_view}_densify{use_densification}"
                                        f"_start{start_sampling_ratio}_target{target_sampling_ratio}"
                                        f"_sh{d_sh}_colors{d_colors}_depth{d_depth}_xyzs{d_xyzs}_quats{d_quats}_scales{d_scales}_opacities{d_opacities}"
                                        f"_{key}"
                                    )
                                    if sh_coeffs_mult_val:
                                        output_dir_name += f"_shmult{sh_coeffs_mult_val}"
                                    if depths_mult_val:
                                        output_dir_name += f"_depth{depths_mult_val}"
                                    output_dir_name += f"_grow2d{grow_grad2d}"
                                    output_dir = os.path.join(STUDENT_DIR, output_dir_name)
                                    cmd += f" --result_dir {output_dir}"

                                    if d_scales > 0 or d_opacities > 0 or not os.path.exists(os.path.join(output_dir, 'ckpts', f'ckpt_{max_steps-1}_rank0.pt')):
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