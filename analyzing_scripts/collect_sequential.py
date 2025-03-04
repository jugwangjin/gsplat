#!/usr/bin/env python
import os
import csv
import json

# Path to the training results directory from the training script.
# Adjust this path as needed.
STUDENT_DIR = '/Bean/log/gwangjin/2025/kdgs/students_sequential/sequential'
# Name for the output CSV file.
OUTPUT_CSV = 'distillation_analysis_results.csv'
# Set MAX_STEPS as used in training.
MAX_STEPS = 10000

def parse_directory_name(dir_name):
    """
    Parse the directory name to extract configuration parameters for the training script.

    Expected naming format:
    {DATASET}_{DATA_FACTOR}_FULL_blur{use_blur_split}_novel{use_novel_view}_densify{use_densification}
    _start{start_sampling_ratio}_target{target_sampling_ratio}_kt{kt}_{distill_loss_terms}_{gradient_key}_grow2d{grow_grad2d}

    Example:
    bicycle_4_FULL_blurTrue_novelFalse_densifyTrue_start0.75_target0.9_kt0.5_l2_means2d_grow2d0.0002
    """
    parts = dir_name.split('_')
    config = {}
    try:
        # Expecting 12 parts according to the naming scheme
        if len(parts) == 12 and parts[2] == "FULL":
            config['dataset'] = parts[0]
            config['data_factor'] = parts[1]
            config['mode'] = 'FULL'
            config['use_blur_split'] = parts[3].replace("blur", "")
            config['use_novel_view'] = parts[4].replace("novel", "")
            config['use_densification'] = parts[5].replace("densify", "")
            config['start_sampling_ratio'] = parts[6].replace("start", "")
            config['target_sampling_ratio'] = parts[7].replace("target", "")
            config['kt'] = parts[8].replace("kt", "")
            config['distill_loss_terms'] = parts[9]
            config['gradient_key'] = parts[10]
            config['grow_grad2d'] = parts[11].replace("grow2d", "")
        else:
            # If the directory name doesn't match the expected format, store it raw.
            config['raw_dir_name'] = dir_name
    except IndexError:
        config['raw_dir_name'] = dir_name
    return config

def read_json_metrics(results_dir, max_steps=MAX_STEPS):
    """
    Look for a JSON results file (e.g. stats/val_step{max_steps-1}.json) in the given directory and
    parse it to extract metrics. Only the following keys are extracted:
      psnr, ssim, lpips, num_GS
    """
    json_metrics = {}
    json_file = os.path.join(results_dir, "stats", f"val_step{max_steps - 1}.json")
    if os.path.exists(json_file):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            for key in ["psnr", "ssim", "lpips", "num_GS"]:
                if key in data:
                    json_metrics[key] = data[key]
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    return json_metrics

def main():
    all_rows = []
    if not os.path.exists(STUDENT_DIR):
        print(f"Directory {STUDENT_DIR} does not exist.")
        return

    for entry in os.listdir(STUDENT_DIR):
        run_dir = os.path.join(STUDENT_DIR, entry)
        if os.path.isdir(run_dir):
            # Extract configuration from directory name.
            config = parse_directory_name(entry)
            # Read metrics from the JSON file inside the stats/ folder.
            json_metrics = read_json_metrics(run_dir)
            # Combine configuration and metrics.
            row = {**config, **json_metrics}
            row['run_directory'] = entry
            all_rows.append(row)

    if not all_rows:
        print("No result directories found.")
        return

    # Determine CSV fieldnames (union of all keys).
    fieldnames = sorted({key for row in all_rows for key in row.keys()})
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Analysis CSV written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
