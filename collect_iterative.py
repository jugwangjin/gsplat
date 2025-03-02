import os
import json
import csv
import argparse
import re

def parse_subdir_name(subdir_name):
    """
    Parse a subdirectory name produced by the new training script.
    
    Expected naming patterns include:
      bicycle_4_FULL_once{target}_densify
      bicycle_4_FULL_once{target}
      bicycle_4_SMALL_repeated{target}_densify_{i}
      bicycle_4_SMALL_repeated{target}_{i}
    
    This function extracts:
      - teacher_type ("full" if "FULL" appears, "small" if "SMALL" appears)
      - experiment_type ("once" or "repeated")
      - target_sampling_ratio (float; the first number encountered)
      - densification (bool; True if "densify" appears in the name)
    """
    params = {}
    # Determine teacher type from the name.
    if "FULL" in subdir_name.upper():
        params["teacher_type"] = "full"
    elif "SMALL" in subdir_name.upper():
        params["teacher_type"] = "small"
    else:
        params["teacher_type"] = ""
    
    # Determine experiment type.
    if "repeated" in subdir_name:
        params["experiment_type"] = "repeated"
    elif "once" in subdir_name:
        params["experiment_type"] = "once"
    else:
        params["experiment_type"] = ""
    
    # Extract target sampling ratio (the first number in the name).
    m = re.search(r'([0-9]*\.?[0-9]+)', subdir_name)
    if m:
        try:
            params["target_sampling_ratio"] = float(m.group(1))
        except:
            params["target_sampling_ratio"] = m.group(1)
    else:
        params["target_sampling_ratio"] = None

    # Determine if densification was used.
    params["densification"] = ("densify" in subdir_name.lower())
    
    return params

def analyze_dir(base_dir, json_filename):
    """
    Scan a given base directory for subdirectories that contain a valid stats JSON file.
    For each such directory, parse the subdirectory name using parse_subdir_name and combine
    the extracted parameters with the contents of the JSON file.
    """
    results = []
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        stats_dir = os.path.join(subdir_path, "stats")
        stats_file = os.path.join(stats_dir, json_filename)
        if os.path.exists(stats_file):
            print(f"Found stats file: {stats_file}")
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
            except Exception as e:
                print(f"Error reading {stats_file}: {e}")
                continue

            entry = {"subdirectory": subdir}
            # Parse parameters from the directory name.
            entry.update(parse_subdir_name(subdir))
            # Update with statistics (e.g. PSNR, SSIM, etc.).
            entry.update(stats)
            results.append(entry)
    return results

def analyze_training_results(results_dir, output_csv):
    """
    Analyze training results found in results_dir by scanning all subdirectories for a stats JSON file.
    Writes a combined CSV with all extracted parameters and statistics.
    """
    # Adjust the JSON filename as needed. Here we assume student runs produce "val_step14999.json".
    json_filename = "val_step14999.json"
    results = analyze_dir(results_dir, json_filename)
    
    if not results:
        print("No valid JSON files found.")
        return

    # Sort results by teacher type and target_sampling_ratio (and optionally by PSNR/SSIM).
    def safe_float(val):
        try:
            return float(val)
        except:
            return 0.0

    results.sort(key=lambda x: (
        x.get("teacher_type", ""),
        safe_float(x.get("target_sampling_ratio", 0)),
        safe_float(x.get("psnr", 0)),
        safe_float(x.get("ssim", 0))
    ), reverse=True)

    header = [
        "subdirectory",
        "teacher_type",
        "experiment_type",
        "densification",
        "target_sampling_ratio",
        "psnr",
        "ssim",
        "lpips",
        "ellipse_time",
        "num_GS"
    ]
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for item in results:
            row = {key: item.get(key, "") for key in header}
            writer.writerow(row)
    
    print(f"Combined CSV saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze training results from the new training script."
    )
    parser.add_argument('--results_dir', type=str, default='../repeating_test',
                        help="Path to the directory containing training result subdirectories")
    parser.add_argument('--output_csv', type=str, default='iterative_combined_results.csv',
                        help="Path to output combined CSV file (default: iterative_combined_results.csv)")
    
    args = parser.parse_args()
    analyze_training_results(args.results_dir, args.output_csv)

if __name__ == "__main__":
    main()
