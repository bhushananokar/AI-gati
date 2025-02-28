import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

# ========== CONFIGURATION ==========
# Set your dataset path here
DATASET_PATH = "/content/drive/MyDrive/your_dataset_path"  # Change this to your actual dataset path

# Other config options
MAX_DEPTH = None      # Maximum depth to display in the tree (None for unlimited)
VISUALIZE = True      # Whether to generate visualization plots
# ===================================

def count_files_by_extension(directory):
    """Count files by extension in the given directory and all subdirectories."""
    extension_counts = defaultdict(int)
    total_files = 0

    for root, _, files in os.walk(directory):
        for file in files:
            total_files += 1
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext:
                extension_counts[file_ext] += 1
            else:
                extension_counts['no extension'] += 1

    return extension_counts, total_files

def get_dir_stats(directory):
    """Get statistics about the directory structure."""
    stats = {
        'total_directories': 0,
        'total_files': 0,
        'max_depth': 0,
        'dir_with_most_files': ('', 0),
        'dir_with_most_subdirs': ('', 0),
        'files_by_level': defaultdict(int),
        'dirs_by_level': defaultdict(int),
        'class_distribution': defaultdict(lambda: defaultdict(int))
    }

    start_depth = directory.count(os.sep)

    # Check if this is likely a dataset with train/val/test structure
    possible_dataset = False
    try:
        root_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        if set(root_dirs) & {'train', 'val', 'test', 'validation'}:
            possible_dataset = True
            print(f"Dataset structure detected with splits: {[d for d in root_dirs if d in ['train', 'val', 'test', 'validation']]}")
    except:
        pass

    for root, dirs, files in os.walk(directory):
        # Calculate current depth
        current_depth = root.count(os.sep) - start_depth
        stats['max_depth'] = max(stats['max_depth'], current_depth)

        # Count directories and files at this level
        stats['total_directories'] += len(dirs)
        stats['dirs_by_level'][current_depth] += len(dirs)
        stats['total_files'] += len(files)
        stats['files_by_level'][current_depth] += len(files)

        # Track directory with most files and subdirectories
        if len(files) > stats['dir_with_most_files'][1]:
            stats['dir_with_most_files'] = (root, len(files))

        if len(dirs) > stats['dir_with_most_subdirs'][1]:
            stats['dir_with_most_subdirs'] = (root, len(dirs))

        # If this looks like a dataset, count files per class per split
        if possible_dataset:
            try:
                path_parts = Path(root).relative_to(directory).parts
                if len(path_parts) >= 1 and path_parts[0] in ['train', 'val', 'test', 'validation']:
                    split = path_parts[0]
                    if len(path_parts) >= 2:
                        # This is likely a class folder
                        class_name = path_parts[1]
                        stats['class_distribution'][split][class_name] = len(files)
            except:
                pass

    return stats

def print_directory_tree(directory, max_depth=None, indent="", last=True, depth=0, file_count=True, count_empty=False, only_show_directories=False):
    """Print a directory tree structure."""
    # Skip if we've reached max_depth
    if max_depth is not None and depth > max_depth:
        return

    path = os.path.basename(directory)

    # Handle root directory case
    if path == "":
        path = directory

    file_count_str = ""
    if file_count:
        try:
            dir_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            if not only_show_directories or (count_empty and not dir_files):
                file_count_str = f" ({len(dir_files)} files)"
        except:
            file_count_str = " (error accessing files)"

    # Print the current directory with proper indentation
    connector = "└── " if last else "├── "
    print(f"{indent}{connector}{path}{file_count_str}")

    # Prepare indentation for subdirectories
    indent += "    " if last else "│   "

    # Get subdirectories
    try:
        subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        subdirs.sort()

        # Print each subdirectory
        for i, subdir in enumerate(subdirs):
            is_last = i == len(subdirs) - 1
            print_directory_tree(
                os.path.join(directory, subdir),
                max_depth,
                indent,
                is_last,
                depth + 1,
                file_count,
                count_empty,
                only_show_directories
            )
    except Exception as e:
        print(f"{indent}└── Error: {str(e)}")

def plot_distribution(stats):
    """Plot various distribution charts based on the collected stats."""
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Dataset Structure Analysis', fontsize=16)

    # Plot 1: Files and directories by level
    ax1 = fig.add_subplot(221)
    levels = sorted(list(set(list(stats['files_by_level'].keys()) + list(stats['dirs_by_level'].keys()))))
    file_counts = [stats['files_by_level'][level] for level in levels]
    dir_counts = [stats['dirs_by_level'][level] for level in levels]

    width = 0.35
    ax1.bar(np.array(levels) - width/2, file_counts, width, label='Files')
    ax1.bar(np.array(levels) + width/2, dir_counts, width, label='Directories')
    ax1.set_xlabel('Directory Level')
    ax1.set_ylabel('Count')
    ax1.set_title('Files and Directories by Level')
    ax1.legend()

    # Plot 2: Class distribution by split (if available)
    if stats['class_distribution']:
        ax2 = fig.add_subplot(222)

        # Convert the nested defaultdict to a DataFrame
        distribution_data = []

        for split, classes in stats['class_distribution'].items():
            for class_name, count in classes.items():
                distribution_data.append({
                    'Split': split,
                    'Class': class_name,
                    'Count': count
                })

        df = pd.DataFrame(distribution_data)

        # If there are too many classes, we'll show the top N classes by total count
        max_classes_to_show = 10
        if len(df['Class'].unique()) > max_classes_to_show:
            top_classes = df.groupby('Class')['Count'].sum().nlargest(max_classes_to_show).index
            df = df[df['Class'].isin(top_classes)]

        # Pivot the data for plotting
        pivot_df = df.pivot(index='Class', columns='Split', values='Count')
        pivot_df.plot(kind='bar', ax=ax2)
        ax2.set_title(f'Sample Distribution by Class and Split (Top {max_classes_to_show})')
        ax2.set_ylabel('Number of Samples')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot 3: File extension distribution
    ax3 = fig.add_subplot(223)
    try:
        extensions, _ = count_files_by_extension(os.path.dirname(stats['dir_with_most_files'][0]))
        if extensions:
            ext_names = list(extensions.keys())
            ext_counts = list(extensions.values())

            # Sort by count in descending order
            sorted_indices = np.argsort(ext_counts)[::-1]
            ext_names = [ext_names[i] for i in sorted_indices]
            ext_counts = [ext_counts[i] for i in sorted_indices]

            # Show only top 10 extensions if there are many
            if len(ext_names) > 10:
                ext_names = ext_names[:10]
                ext_counts = ext_counts[:10]

            ax3.bar(ext_names, ext_counts)
            ax3.set_title('File Extensions Distribution (Top 10)')
            ax3.set_ylabel('Count')
            ax3.tick_params(axis='x', rotation=45)
    except:
        ax3.text(0.5, 0.5, 'Could not generate extension distribution',
                horizontalalignment='center', verticalalignment='center')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def analyze_folder_structure(directory, max_depth=None, visualize=True):
    """Analyze and print folder structure with statistics."""
    try:
        directory = os.path.abspath(directory)

        if not os.path.isdir(directory):
            print(f"Error: '{directory}' is not a valid directory")
            return

        print(f"\n=== FOLDER STRUCTURE: {directory} ===\n")
        print_directory_tree(directory, max_depth, file_count=True, only_show_directories=False)

        print("\n=== DIRECTORY STATISTICS ===\n")
        stats = get_dir_stats(directory)

        print(f"Total directories: {stats['total_directories']}")
        print(f"Total files: {stats['total_files']}")
        print(f"Maximum directory depth: {stats['max_depth']}")
        print(f"Directory with most files: {stats['dir_with_most_files'][0]} ({stats['dir_with_most_files'][1]} files)")
        print(f"Directory with most subdirectories: {stats['dir_with_most_subdirs'][0]} ({stats['dir_with_most_subdirs'][1]} subdirectories)")

        # Print file extensions
        extensions, total_files = count_files_by_extension(directory)
        print("\nFile extensions:")
        for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_files) * 100
            print(f"  {ext}: {count} files ({percentage:.1f}%)")

        # Print class distribution if this looks like a dataset
        if stats['class_distribution']:
            print("\nClass distribution by split:")
            for split, classes in sorted(stats['class_distribution'].items()):
                total_split_files = sum(classes.values())
                print(f"\n  {split.upper()} split: {total_split_files} total samples")

                # Calculate class imbalance
                if classes:
                    min_class = min(classes.values())
                    max_class = max(classes.values())
                    imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
                    print(f"  Imbalance ratio (max/min): {imbalance_ratio:.2f}")

                for class_name, count in sorted(classes.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_split_files) * 100 if total_split_files > 0 else 0
                    print(f"    {class_name}: {count} samples ({percentage:.1f}%)")

        # Create visualizations
        if visualize:
            try:
                plot_distribution(stats)
            except Exception as e:
                print(f"\nWarning: Could not generate visualization. Error: {str(e)}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Run the analysis
def run_analysis(dataset_path=None):
    path_to_analyze = dataset_path or DATASET_PATH

    if path_to_analyze == "/content/drive/MyDrive/your_dataset_path":
        print("Please set your dataset path either in the DATASET_PATH variable or pass it to run_analysis()")
        return

    analyze_folder_structure(
        path_to_analyze,
        max_depth=MAX_DEPTH,
        visualize=VISUALIZE
    )

# This doesn't run automatically so you can set your own path first
print("To analyze your dataset, run:")
print("run_analysis()  # Uses path from DATASET_PATH")
print("# or")
print("run_analysis('/path/to/your/dataset')  # Uses custom path")
