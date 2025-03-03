import os

import subprocess


from pathlib import Path


dataset_path = './data/360_v2'
result_path = '../../gsplat_teachers'
SCALE: int = 4
BASE_COMMAND = f"python kd_trainer.py default --data_dir {dataset_path}/{{}}/ --data_factor {SCALE} --result_dir ../../gsplat_teachers/{{}}_{SCALE} --disable_viewer"

def main():
    dataset_names = [d for d in os.listdir(dataset_path) if os.path.exists(os.path.join(dataset_path, d, 'poses_bounds.npy'))]

    command_list = []   

    for dataset in dataset_names:
        data_dir = os.path.join(dataset_path, dataset)
        scale = SCALE
        result_dir = os.path.join(result_path, f'{dataset}_{scale}')

        command = f'python kd_trainer.py default --data_dir {data_dir} --data_factor {scale} --result_dir {result_dir} --disable_viewer'
        command_list.append(command)

    # print the command lists 
    print('Found the following datasets:')
    for command in command_list:
        print(command)
    
    try:
        for command in command_list:
            print(f"Running command: {command}")
            subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
    except KeyboardInterrupt:
        print("Interrupted")
        return


if __name__=='__main__':
    main()