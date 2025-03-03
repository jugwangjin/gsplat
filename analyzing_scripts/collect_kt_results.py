#!/usr/bin/env python
import os
import csv
import re

# Path to the student results (change if needed)
STUDENT_DIR = './gsplat_students/kt_test_on_ms'
# Name for the output CSV file
OUTPUT_CSV = 'analysis_results.csv'

def parse_directory_name(dir_name):
    """
    Parse the directory name to extract configuration parameters.
    Expected naming format:
      {DATASET}_{DATA_FACTOR}_FULL_blur{use_blur}_novel{use_novel}_densify{use_densify}
      _start{start_sampling}_target{target_sampling}
      _sh{distill_sh}_colors{distill_colors}_depth{distill_depth}_xyzs{distill_xyzs}
      _quats{distill_quats}_scales{distill_quats}_opacities{distill_quats}_{key}
      [ _shmult{sh_coeffs_mult} ] [ _depth{depths_mult} ] _grow2d{grow_grad2d}
    """
    # Split on underscore
    parts = dir_name.split('_')
    config = {}
    try:
        # Basic info: dataset, factor and fixed string "FULL"
        config['dataset'] = parts[0]
        config['data_factor'] = parts[1]
        # Next three parts: blur, novel and densify flags
        config['use_blur_split'] = parts[3].replace("blur", "")
        config['use_novel_view'] = parts[4].replace("novel", "")
        config['use_densification'] = parts[5].replace("densify", "")
        # Sampling ratios
        config['start_sampling_ratio'] = parts[6].replace("start", "")
        config['target_sampling_ratio'] = parts[7].replace("target", "")
        # Lambdas
        config['distill_sh_lambda'] = parts[8].replace("sh", "")
        config['distill_colors_lambda'] = parts[9].replace("colors", "")
        config['distill_depth_lambda'] = parts[10].replace("depth", "")
        config['distill_xyzs_lambda'] = parts[11].replace("xyzs", "")
        config['distill_quats_lambda'] = parts[12].replace("quats", "")
        config['distill_scales_lambda'] = parts[13].replace("scales", "")
        config['distill_opacities_lambda'] = parts[14].replace("opacities", "")
        # Key for gradient computation
        config['gradient_key'] = parts[15]
        # Look for additional optional parameters:
        for part in parts[16:]:
            if part.startswith("shmult"):
                config['sh_coeffs_mult'] = part.replace("shmult", "")
            elif part.startswith("depth"):
                config['depths_mult'] = part.replace("depth", "")
            elif part.startswith("grow2d"):
                config['grow_grad2d'] = part.replace("grow2d", "")
    except IndexError:
        # In case the naming pattern differs, store the full name for reference.
        config['raw_dir_name'] = dir_name
    return config

def read_metrics(results_dir):
    """
    Look for a results file (e.g. results.txt) in the given directory and
    parse it to extract metrics.
    The file is expected to have lines formatted as:
      MetricName: value
    """
    metrics = {}
    results_file = os.path.join(results_dir, "results.txt")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    key, value = line.split(":", 1)
                    metrics[key.strip()] = value.strip()
    return metrics

def main():
    all_rows = []
    # Loop over each run directory in the student results directory.
    if not os.path.exists(STUDENT_DIR):
        print(f"Directory {STUDENT_DIR} does not exist.")
        return

    for entry in os.listdir(STUDENT_DIR):
        run_dir = os.path.join(STUDENT_DIR, entry)
        if os.path.isdir(run_dir):
            config = parse_directory_name(entry)
            metrics = read_metrics(run_dir)
            # Combine configuration and metric values.
            row = {**config, **metrics}
            row['run_directory'] = entry
            all_rows.append(row)

    if not all_rows:
        print("No result directories found.")
        return

    # Determine the CSV fieldnames (union of all keys)
    fieldnames = sorted({key for row in all_rows for key in row.keys()})
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Analysis CSV written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
