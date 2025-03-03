#!/usr/bin/env python
import os
import csv
import json
import glob
import re
import argparse

def extract_step_from_filename(filename):
    """
    Extract the step number from the filename.
    Expected filename format: val_step{step}.json
    """
    basename = os.path.basename(filename)
    match = re.search(r"val_step(\d+)\.json", basename)
    if match:
        return int(match.group(1))
    return None

def parse_json_file(json_file):
    """
    Load the JSON file and extract the desired metrics.
    """
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        data = {}
    # Add the step information extracted from the filename.
    step = extract_step_from_filename(json_file)
    if step is not None:
        data["step"] = step
    # Optionally, you could add the file path to help identify the source.
    data["result_file"] = json_file
    return data

def main():
    parser = argparse.ArgumentParser(description="Aggregate JSON stats into a CSV file.")
    parser.add_argument("--given_dir", type=str, required=True,
                        help="Directory containing the stats folder with JSON files.")
    parser.add_argument("--output_csv", type=str, default="aggregated_results.csv",
                        help="Output CSV file name (default: aggregated_results.csv)")
    args = parser.parse_args()

    given_dir = args.given_dir
    output_csv = args.output_csv

    # Create the glob pattern: {given_dir}/stats/val_step*.json
    json_pattern = os.path.join(given_dir, "stats", "val_step*.json")
    json_files = sorted(glob.glob(json_pattern))

    if not json_files:
        print(f"No JSON files found with pattern: {json_pattern}")
        return

    all_rows = []
    for json_file in json_files:
        row = parse_json_file(json_file)
        if row:
            all_rows.append(row)

    if not all_rows:
        print("No valid JSON files parsed.")
        return

    # Determine CSV fieldnames (union of all keys from the JSON files)
    fieldnames = sorted({key for row in all_rows for key in row.keys()})
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Aggregated CSV written to {output_csv}")

if __name__ == "__main__":
    main()
