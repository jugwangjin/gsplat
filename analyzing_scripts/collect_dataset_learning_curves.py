import os
import json
import re
import argparse
import shutil
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend suitable for command line systems
import matplotlib.pyplot as plt

def get_step_psnr(subdir):
    """
    For a given subdirectory, look into the 'stats' folder and collect
    (step, psnr) pairs from files named 'val_step{n_step}.json'.
    """
    stats_dir = os.path.join(subdir, 'stats')
    if not os.path.isdir(stats_dir):
        return []  # Return an empty list if there is no 'stats' folder

    data = []
    for filename in os.listdir(stats_dir):
        if filename.startswith('val_step') and filename.endswith('.json'):
            match = re.search(r'val_step(\d+)\.json', filename)
            if match:
                step = int(match.group(1))
                file_path = os.path.join(stats_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        quality = json.load(f)
                    psnr = quality.get("psnr")
                    if psnr is not None:
                        data.append((step, psnr))
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    data.sort(key=lambda x: x[0])
    return data

def copy_render_images(subdir, stats_data, images_output_dir, errors_output_dir):
    """
    For each valid n_step from the stats_data, look in the 'renders' directory
    for images with index 0 and 10 (e.g., val_step{n_step}_0000.png and
    val_step{n_step}_0010.png). If the images exist, copy them into the provided
    images_output_dir, renaming them so that the subdirectory name is at the end.
    """
    renders_dir = os.path.join(subdir, 'renders')
    errors_dirs = os.path.join(subdir, 'errors')
    if not os.path.isdir(renders_dir):
        return

    # Get a label from the subdirectory name.
    method_label = os.path.basename(subdir)
    
    for step, _ in stats_data:
        for img_index in [0, 10]:
            # Format the image file name, assuming 4-digit formatting for the index.
            src_filename = f"val_step{step}_{img_index:04d}.png"
            src_filepath = os.path.join(renders_dir, src_filename)
            if os.path.isfile(src_filepath):
                # New filename: step_index_imgIndex_methodLabel.png
                dst_filename = f"{step}_index{img_index:04d}_{method_label}.png"
                dst_filepath = os.path.join(images_output_dir, dst_filename)
                try:
                    shutil.copy(src_filepath, dst_filepath)
                    print(f"Copied {src_filepath} to {dst_filepath}")
                except Exception as e:
                    print(f"Error copying {src_filepath} to {dst_filepath}: {e}")
            else:
                print(f"File {src_filepath} not found, skipping.")

            error_filename = f"val_step{step}_{img_index:04d}_error.png"
            error_filepath = os.path.join(errors_dirs, error_filename)
            if os.path.isfile(error_filepath):
                # New filename: step_index_imgIndex_methodLabel.png
                dst_filename = f"{step}_index{img_index:04d}_{method_label}.png"
                dst_filepath = os.path.join(errors_output_dir, dst_filename)
                try:
                    shutil.copy(error_filepath, dst_filepath)
                    print(f"Copied {error_filepath} to {dst_filepath}")
                except Exception as e:
                    print(f"Error copying {error_filepath} to {dst_filepath}: {e}")

def read_num_gaussians(result_dir: str) -> int:
    """Read number of gaussians from val_step29999.json"""
    json_path = os.path.join(result_dir, 'stats', 'val_step29999.json')
    try:
        with open(json_path, 'r') as f:
            stats = json.load(f)
            return stats['num_GS']
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None

def plot_learning_curve(dataset, base_dir, output_dir):
    """
    Plot learning curve for a specific dataset
    """
    plt.figure(figsize=(10, 6))
    
    # Create output directories
    images_output_dir = os.path.join(output_dir, dataset, 'images')
    errors_output_dir = os.path.join(output_dir, dataset, 'errors')
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(errors_output_dir, exist_ok=True)
    
    # Find all subdirectories for this dataset
    subdirectories = [os.path.join(base_dir, d) 
                     for d in os.listdir(base_dir) 
                     if os.path.isdir(os.path.join(base_dir, d)) and dataset in d]
    
    # Filter out unwanted directories
    subdirectories = [d for d in subdirectories 
                     if (not 'iter10' in d) 
                     and not d.endswith('25') 
                     and not d.endswith('06') 
                     and not d.endswith('reinit') 
                     and not d.endswith('no_reinit_2') 
                     and not 'prog' in os.path.basename(d)]
    
    print(f"\nProcessing {dataset}...")
    print(f"Found subdirectories: {subdirectories}")

    for idx, subdir in enumerate(subdirectories):
        stats_data = get_step_psnr(subdir)
        if not stats_data:
            continue
        
        # Plot the learning curve
        steps, psnrs = zip(*stats_data)
        base_label = os.path.basename(subdir)
        
        # 가우시안 수 추출
        num_gaussians = read_num_gaussians(subdir)
        if num_gaussians is None:
            num_gaussians = 500000  # 기본값
        gaussian_count = num_gaussians / 1_000_000  # 백만 단위로 변환
        
        # 최종 PSNR 값 추출
        final_psnr = psnrs[-1]
        
        label = f"{base_label} ({gaussian_count:.2f}M, PSNR: {final_psnr:.2f})"
        color = plt.cm.tab10(idx % 10)
        plt.plot(steps, psnrs, label=label, color=color, alpha=0.6, linewidth=1.0, marker='x', markersize=4)
        
        if not args.copy_images:
            continue
        copy_render_images(subdir, stats_data, images_output_dir, errors_output_dir)
    
    plt.xlabel('Step')
    plt.ylabel('PSNR')
    plt.title(f'Learning Curve for {dataset} (Step vs PSNR)')
    plt.ylim(18)
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    output_image = os.path.join(output_dir, f'{dataset}_learning_curve.png')
    plt.savefig(output_image)
    plt.close()
    print(f"Plot saved to {output_image}")

def main():
    # 데이터셋 목록
    datasets = [
        "bonsai",
        "bicycle",
        "counter",
        "flowers",
        "garden",
        "kitchen",
        "room",
        "stump",
        "treehill"
    ]
    
    base_dir = args.base_dir
    output_dir = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each dataset
    for dataset in datasets:
        plot_learning_curve(dataset, base_dir, output_dir)
        print(f"\nCompleted processing {dataset}")
        print("-" * 40)
    
    print("\nAll datasets processed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot learning curves for each dataset from json stats and copy select rendered images.")
    parser.add_argument("--base-dir", required=True, help="Base directory containing subdirectories with stats and renders folders.")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots and images.")
    parser.add_argument("--copy_images", action='store_true', help="Copy the selected rendered images to the output directory.")
    args = parser.parse_args()
    
    main() 