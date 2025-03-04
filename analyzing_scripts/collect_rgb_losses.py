#!/usr/bin/env python
import os
import csv
import json

# Path to the training results directory (adjust if needed)
STUDENT_DIR = './gsplat_students/rgb_loss_test3'
# Name for the output CSV file
OUTPUT_CSV = 'rgb_loss_analysis_results.csv'
# Set MAX_STEPS as used in training (e.g. 10000)
MAX_STEPS = 10000

def parse_directory_name(dir_name):
    """
    Parse the directory name to extract configuration parameters.
    Supports two naming formats:
    
    1. Simple (zero) format:
       {DATASET}_{DATA_FACTOR}_FULL_zero_grow2d{grow_grad2d}
       
    2. Detailed format:
       {DATASET}_{DATA_FACTOR}_FULL
       _blur{use_blur_split}_novel{use_novel_view}_densify{use_densification}
       _start{start_sampling_ratio}_target{target_sampling_ratio}
       _sh{distill_sh_lambda}_colors{distill_colors_lambda}_depth{distill_depth_lambda}_xyzs{distill_xyzs_lambda}
       _quats{distill_quats_lambda}_scales{distill_scales_lambda}_opacities{distill_opacities_lambda}
       _image{image_lambda}_{distill_loss_terms}_{gradient_key}
       _grow2d{grow_grad2d}
    """
    parts = dir_name.split('_')
    config = {}
    try:
        # Basic info: dataset and data factor
        config['dataset'] = parts[0]
        config['data_factor'] = parts[1]
        # Check for the simple "zero" format
        if len(parts) == 5 and parts[3] == "zero":
            config['mode'] = 'zero'
            # parts[4] is "grow2d{grow_grad2d}"
            config['grow_grad2d'] = parts[4].replace("grow2d", "")
        # Detailed format expected to have at least 19 parts.
        elif len(parts) >= 19:
            config['use_blur_split']      = parts[3].replace("blur", "")
            config['use_novel_view']      = parts[4].replace("novel", "")
            config['use_densification']   = parts[5].replace("densify", "")
            config['start_sampling_ratio'] = parts[6].replace("start", "")
            config['target_sampling_ratio']= parts[7].replace("target", "")
            config['distill_sh_lambda']   = parts[8].replace("sh", "")
            config['distill_colors_lambda'] = parts[9].replace("colors", "")
            config['distill_depth_lambda']  = parts[10].replace("depth", "")
            config['distill_xyzs_lambda']   = parts[11].replace("xyzs", "")
            config['distill_quats_lambda']  = parts[12].replace("quats", "")
            config['distill_scales_lambda'] = parts[13].replace("scales", "")
            config['distill_opacities_lambda'] = parts[14].replace("opacities", "")
            config['image_lambda']        = parts[15].replace("image", "")
            config['distill_loss_terms']  = parts[16]
            config['gradient_key']        = parts[17]
            config['grow_grad2d']         = parts[18].replace("grow2d", "")
        else:
            # If the format does not match any known pattern, store the raw directory name.
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
            # Parse configuration from directory name
            config = parse_directory_name(entry)
            # Read metrics from the JSON file inside the stats/ folder
            json_metrics = read_json_metrics(run_dir)
            # Combine configuration and metrics
            row = {**config, **json_metrics}
            row['run_directory'] = entry
            all_rows.append(row)

    if not all_rows:
        print("No result directories found.")
        return

    # Determine CSV fieldnames (union of all keys)
    fieldnames = sorted({key for row in all_rows for key in row.keys()})
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Analysis CSV written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
