import os
import subprocess


BASE_COMMAND = "python student_trainer.py distillation --teacher_ckpt ../gsplat_teachers/bicycle_4/ckpts/ckpt_29999_rank0.pt --data_factor 4 --data_dir data/360_v2/bicycle/ --disable_viewer"
STUDENT_DIR = "../gsplat_students_v3"
    # refine_every: int = 200

def main():
    options = {'reinit': 'disable_depth_reinit',
               'weight': 'apply_vis_on_teacher_sampling',
               'depth': 'depth_lambda',
                'sampling': 'teacher_sampling_ratio'
               }

# --reset_every 1000
    possible_values = {
        'reinit': [True, False],
        'weight': [True],
        'disable_pruning': [True],
        'depth': [0, 0.01, 0.001, 0.05, 0.005],
        'sampling': [0.05, 0.1],
        # 'sampling': [0.1, 0.05]
    }


    BASIC_output_dir_name = 'bicycle_4_oftensplit'

    commands = []
    output_dirs = []
    options = []
    # make combinations
    for reinit in possible_values['reinit']:
        for weight in possible_values['weight']:
            for depth in possible_values['depth']:
                for sampling in possible_values['sampling']:
                    for disable_pruning in possible_values['disable_pruning']:
                        output_dir_name = BASIC_output_dir_name
                        command = BASE_COMMAND
                        if reinit:
                            command += ' --disable_depth_reinit'
                            output_dir_name += '_reinit'
                        if weight:
                            command += ' --apply_vis_on_teacher_sampling'
                            output_dir_name += '_weight'
                        
                        command += ' --depth_lambda {}'.format(depth)
                        if depth > 0:
                            output_dir_name += '_{}'.format(depth)
                        
                        command += ' --teacher_sampling_ratio {}'.format(sampling)
                        output_dir_name += '_sampling_{}'.format(sampling)
                        
                        if disable_pruning:
                            command += ' --strategy.disable_pruning'
                            output_dir_name += '_disable_pruning'


                        output_dir = os.path.join(STUDENT_DIR, output_dir_name)

                        command += ' --result_dir {}'.format(output_dir)

                        # print(command)
                        commands.append(command)
                        output_dirs.append(output_dir)
                        options.append({'reinit': reinit, 'weight': weight, 'depth': depth, 'sampling': sampling, 'disable_pruning': disable_pruning})

    print(len(commands))
    # exit()

    try:
        for command in commands:
            # print(command)
            # print command - BASE_COMMAND
            diff_from_base = command.replace(BASE_COMMAND, '')
            print(diff_from_base)
            subprocess.run(command, shell=True)
    except KeyboardInterrupt:
        print('Interrupted')
    except Exception as e:
        print(e)


    # grads = ['0008', '0010', '0012', '0016']
    # for grad in grads:

    #     command = f"python teacher_trainer.py default --data_factor 4 --data_dir data/360_v2/bicycle/ --result_dir ../gsplat_students/bicycle_4_default_{grad} --disable_viewer --strategy.refine_start_iter 20000 --strategy.refine_start_iter 500 --strategy.refine_stop_iter 3000 --strategy.grow_grad2d 0.{grad}"
    #     subprocess.run(command, shell=True)

    # after running all, collect the results. 
    # the quantitative results are saved at: 
    # output_dir/stats/val_step4999.json
    # the format:
    #{"psnr": 23.801332473754883, "ssim": 0.6488418579101562, "lpips": 0.29714885354042053, "ellipse_time": 0.023675689697265623, "num_GS": 574409}
    # collect the results as csv file.

    return


if __name__ == "__main__":
    main()