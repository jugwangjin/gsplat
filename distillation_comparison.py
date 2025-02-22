import os
import subprocess

BASE_COMMAND = "CUDA_VISIBLE_DEVICES=1 python student_trainer.py distill2d --teacher_ckpt gsplat_teachers/bicycle_4/ckpts/ckpt_29999_rank0.pt --data_factor 4 --data_dir data/360_v2/bicycle/ --disable_viewer --strategy.grow_grad2d 0.0009 --apply_vis_on_teacher_sampling"
STUDENT_DIR = "gsplat_students_v7"

def main():
    distill_lambda_keys = ['sh', 'colors', 'depth', 'xyzs', 'quats']
    key_for_gradient = ['depths_and_sh', 'rendered_sh_coeffs', 'sh_coeffs', 'depths', ]

    BASIC_output_dir_name = 'bicycle_4'

    sh_coeffs_mults = [1, 2, 4, 8]

    combinations = []

    combinations.append([5e-1, 0, 5e-1, 0, 0])
    combinations.append([5e-1, 0, 0, 0, 0])
    combinations.append([0, 5e-1, 0, 0, 0])
    combinations.append([0, 0, 5e-1, 0, 0])
    combinations.append([0, 0, 0, 5e-1, 0])
    combinations.append([0, 0, 0, 0, 5e-1])

    combinations.append([1, 0, 1, 0, 0])
    combinations.append([1, 0, 0, 0, 0])
    combinations.append([0, 1, 0, 0, 0])
    combinations.append([0, 0, 1, 0, 0])
    combinations.append([0, 0, 0, 1, 0])
    combinations.append([0, 0, 0, 0, 1])

    grow_grad2ds = [0.0002]

    commands = []
    # Ensure teacher checkpoint exists, if not run teacher training.
    if not os.path.exists("gsplat_teachers/bicycle_4/ckpts/ckpt_29999_rank0.pt"):
        commands.append("CUDA_VISIBLE_DEVICES=1 python teacher_trainer.py default --data_factor 4 --data_dir data/360_v2/bicycle --disable_viewer --result_dir gsplat_teachers/bicycle_4")

    for combination in combinations:
        # Extract all five lambda parameters.
        distill_sh_lambda = combination[0]
        distill_colors_lambda = combination[1]
        distill_depth_lambda = combination[2]
        distill_xyzs_lambda = combination[3]
        distill_quats_lambda = combination[4]

        base_command = BASE_COMMAND
        base_command += f' --distill_colors_lambda {distill_colors_lambda}'
        base_command += f' --distill_depth_lambda {distill_depth_lambda}'
        base_command += f' --distill_xyzs_lambda {distill_xyzs_lambda}'
        base_command += f' --distill_quats_lambda {distill_quats_lambda}'
        base_command += f' --distill_sh_lambda {distill_sh_lambda}'

        # Case 1: Both sh and depth lambdas are > 0
        if distill_sh_lambda > 0 and distill_depth_lambda > 0:
            for gradient_key in key_for_gradient:
                # All keys valid since both lambdas are nonzero
                for sh_mult in sh_coeffs_mults:
                    depth_mult = sh_mult 
                    for grow_grad2d in grow_grad2ds:
                        cmd = base_command
                        cmd += f' --strategy.sh_coeffs_mult {sh_mult}'
                        cmd += f' --strategy.depths_mult {depth_mult}'
                        cmd += f' --strategy.key_for_gradient {gradient_key}'
                        cmd += f' --strategy.grow_grad2d {grow_grad2d}'
                        
                        output_dir_name = (
                            BASIC_output_dir_name +
                            f'_sh{distill_sh_lambda}_colors{distill_colors_lambda}_depth{distill_depth_lambda}_xyzs{distill_xyzs_lambda}_quats{distill_quats_lambda}_'
                            f'{gradient_key}_sh{sh_mult}_depth{depth_mult}_grow2d{grow_grad2d}'
                        )
                        output_dir = os.path.join(STUDENT_DIR, output_dir_name)
                        cmd += f' --result_dir {output_dir}'

                        if os.path.exists(os.path.join(output_dir, 'ckpts', 'ckpt_9999_rank0.pt')):
                            continue
                        commands.append(cmd)

        # Case 2: Only sh lambda > 0
        elif distill_sh_lambda > 0:
            for gradient_key in key_for_gradient:
                # Skip gradient keys that require a depth component
                if gradient_key in ['depths', 'depths_and_sh']:
                    continue
                for sh_mult in sh_coeffs_mults:
                    for grow_grad2d in grow_grad2ds:
                        cmd = base_command
                        cmd += f' --strategy.sh_coeffs_mult {sh_mult}'
                        cmd += f' --strategy.key_for_gradient {gradient_key}'
                        cmd += f' --strategy.grow_grad2d {grow_grad2d}'

                        output_dir_name = (
                            BASIC_output_dir_name +
                            f'_sh{distill_sh_lambda}_colors{distill_colors_lambda}_depth{distill_depth_lambda}_xyzs{distill_xyzs_lambda}_quats{distill_quats_lambda}_'
                            f'{gradient_key}_sh{sh_mult}_grow2d{grow_grad2d}'
                        )
                        output_dir = os.path.join(STUDENT_DIR, output_dir_name)
                        cmd += f' --result_dir {output_dir}'
                        if os.path.exists(os.path.join(output_dir, 'ckpts', 'ckpt_9999_rank0.pt')):
                            continue
                        commands.append(cmd)

        # Case 3: Only depth lambda > 0
        elif distill_depth_lambda > 0:
            for gradient_key in key_for_gradient:
                # Only keys that relate to depth are allowed
                if gradient_key in ['rendered_sh_coeffs', 'depths_and_sh']:
                    continue
                for depth_mult in sh_coeffs_mults:
                    for grow_grad2d in grow_grad2ds:
                        cmd = base_command
                        cmd += f' --strategy.depths_mult {depth_mult}'
                        cmd += f' --strategy.key_for_gradient {gradient_key}'
                        cmd += f' --strategy.grow_grad2d {grow_grad2d}'

                        output_dir_name = (
                            BASIC_output_dir_name +
                            f'_sh{distill_sh_lambda}_colors{distill_colors_lambda}_depth{distill_depth_lambda}_xyzs{distill_xyzs_lambda}_quats{distill_quats_lambda}_'
                            f'{gradient_key}_depth{depth_mult}_grow2d{grow_grad2d}'
                        )
                        output_dir = os.path.join(STUDENT_DIR, output_dir_name)
                        cmd += f' --result_dir {output_dir}'
                        if os.path.exists(os.path.join(output_dir, 'ckpts', 'ckpt_9999_rank0.pt')):
                            continue
                        commands.append(cmd)

        # Case 4: Both lambdas are zero -> produce one unified command (to avoid duplicates)
        else:
            cmd = base_command
            output_dir_name = (
                BASIC_output_dir_name +
                f'_sh{distill_sh_lambda}_colors{distill_colors_lambda}_depth{distill_depth_lambda}_xyzs{distill_xyzs_lambda}_quats{distill_quats_lambda}'
            )
            output_dir = os.path.join(STUDENT_DIR, output_dir_name)
            cmd += f' --result_dir {output_dir}'
            if os.path.exists(os.path.join(output_dir, 'ckpts', 'ckpt_9999_rank0.pt')):
                continue
            commands.append(cmd)

    # Add gradient_key as 'means2d', no need for mults.
    combinations.append([0, 0, 0, 0, 0])
    for combination in combinations:        
        distill_sh_lambda = combination[0]
        distill_colors_lambda = combination[1]
        distill_depth_lambda = combination[2]
        distill_xyzs_lambda = combination[3]
        distill_quats_lambda = combination[4]

        base_command = BASE_COMMAND
        base_command += f' --distill_colors_lambda {distill_colors_lambda}'
        base_command += f' --distill_depth_lambda {distill_depth_lambda}'
        base_command += f' --distill_xyzs_lambda {distill_xyzs_lambda}'
        base_command += f' --distill_quats_lambda {distill_quats_lambda}'
        base_command += f' --distill_sh_lambda {distill_sh_lambda}'

        for gradient_key in ['means2d']:
            # All keys valid since both lambdas are nonzero
            grow_grad2ds = [0.0008, 0.001]
            for grow_grad2d in grow_grad2ds:
                cmd = base_command
                cmd += f' --strategy.key_for_gradient {gradient_key}'
                cmd += f' --strategy.grow_grad2d {grow_grad2d}'
                
                output_dir_name = (
                    BASIC_output_dir_name +
                    f'_sh{distill_sh_lambda}_colors{distill_colors_lambda}_depth{distill_depth_lambda}_xyzs{distill_xyzs_lambda}_quats{distill_quats_lambda}_'
                    f'{gradient_key}_grow2d{grow_grad2d}'
                )
                output_dir = os.path.join(STUDENT_DIR, output_dir_name)
                cmd += f' --result_dir {output_dir}'
                if os.path.exists(os.path.join(output_dir, 'ckpts', 'ckpt_9999_rank0.pt')):
                    continue
                commands.append(cmd)

    # Parse commands, extract all options, and save as a CSV file.
    import csv
    all_keys = set()
    parsed_commands = []

    for command in commands:
        command_parts = command.split(' ')
        options = {}

        for i in range(len(command_parts)):
            if command_parts[i].startswith('--'):
                key = command_parts[i][2:]  # Remove '--'
                value = command_parts[i + 1] if i + 1 < len(command_parts) and not command_parts[i + 1].startswith('--') else ""
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

    print(len(commands))
    try:
        for command in commands:
            print(command)
            subprocess.run(command, shell=True)
    except KeyboardInterrupt:
        print('Interrupted')
    except Exception as e:
        print(e)

    return

if __name__ == "__main__":
    main()
