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

def copy_render_images(subdir, stats_data, images_output_dir):
    """
    For each valid n_step from the stats_data, look in the 'renders' directory
    for images with index 0 and 10 (e.g., val_step{n_step}_0000.png and
    val_step{n_step}_0010.png). If the images exist, copy them into the provided
    images_output_dir, renaming them so that the subdirectory name is at the end.
    """
    renders_dir = os.path.join(subdir, 'renders')
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

def main(base_dir, output_image, images_output_dir):
    """
    For a given base directory, this function finds all subdirectories,
    collects the (step, psnr) data from each subdirectory's 'stats' folder,
    plots a learning curve (step vs. PSNR) for each method, and copies the 
    corresponding rendered images (for index 0 and 10) to a flat output directory.
    """
    plt.figure(figsize=(10, 6))
    
    # Create the images output directory if it doesn't exist.
    os.makedirs(images_output_dir, exist_ok=True)
    
    # List all subdirectories in the base directory.
    subdirectories = [os.path.join(base_dir, d) 
                      for d in os.listdir(base_dir) 
                      if os.path.isdir(os.path.join(base_dir, d))]
    
    for subdir in subdirectories:
        stats_data = get_step_psnr(subdir)
        if not stats_data:
            continue  # Skip if no stats data was found.
        
        # Plot the learning curve for this method.
        steps, psnrs = zip(*stats_data)
        label = os.path.basename(subdir)
        plt.plot(steps, psnrs, label=label)
        
        # Copy the rendered images corresponding to index 0 and 10 for each valid step.
        copy_render_images(subdir, stats_data, images_output_dir)
    
    plt.xlabel('Step')
    plt.ylabel('PSNR')
    plt.title('Learning Curve (Step vs PSNR)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file.
    plt.savefig(output_image)
    print(f"Plot saved to {output_image}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot learning curves (step vs PSNR) from json stats and copy select rendered images.")
    parser.add_argument("--base-dir", required=True, help="Base directory containing subdirectories with stats and renders folders.")
    parser.add_argument("--output", default='learning_curve.png', help="Output image file name for the learning curve plot (e.g., plot.png)")
    parser.add_argument("--images-output-dir", default='learning_curve', help="Output directory to copy the selected rendered images.")
    args = parser.parse_args()
    
    main(args.base_dir, args.output, args.images_output_dir)
