# Script to analyze the dataset statistics and generate distribution plots
# Reference: data loading script from train.py
# Output: Distribution histograms for all data (train+val) from 0.0-3.4,
#         number of samples per class for all data, and separate distributions
#         for train and val sets

import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

def collect_data_statistics(root_dir):
    """Collect label statistics from dataset, rounding labels to one decimal place"""
    train_data = defaultdict(int)
    val_data = defaultdict(int)
    all_data = defaultdict(int)
    
    # Collect training set data
    train_dir = os.path.join(root_dir, 'train')
    for label_folder in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, label_folder)):
            continue
        # Round label to one decimal place
        label = round(float(label_folder), 1)
        num_images = len([f for f in os.listdir(os.path.join(train_dir, label_folder)) if f.endswith('.jpg')])
        train_data[label] += num_images  # Accumulate count for same labels
        all_data[label] += num_images

    # Collect validation set data
    val_dir = os.path.join(root_dir, 'val')
    for label_folder in os.listdir(val_dir):
        if not os.path.isdir(os.path.join(val_dir, label_folder)):
            continue
        # Round label to one decimal place
        label = round(float(label_folder), 1)
        num_images = len([f for f in os.listdir(os.path.join(val_dir, label_folder)) if f.endswith('.jpg')])
        val_data[label] += num_images  # Accumulate count for same labels
        all_data[label] += num_images
    
    return train_data, val_data, all_data

def plot_distribution(data_dict, title, save_path):
    """Plot distribution histogram for the data"""
    # Set global font size and weight
    plt.rcParams.update({
        'font.size': 16,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold'
    })
    
    plt.figure(figsize=(15, 8))
    
    # Prepare data
    labels = sorted(data_dict.keys())
    values = [data_dict[label] for label in labels]
    
    # Plot bar chart
    plt.bar(labels, values, width=0.1)
    
    # Set chart properties with larger fonts
    plt.title(title, fontsize=32, pad=20, weight='bold')
    plt.xlabel('Fibrosis Score', fontsize=28, labelpad=15, weight='bold')
    plt.ylabel('Number of Images', fontsize=28, labelpad=15, weight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set tick font size and weight
    plt.xticks(fontsize=24, weight='bold')
    plt.yticks(fontsize=24, weight='bold')
    
    # Add value labels with larger font
    for i, v in enumerate(values):
        plt.text(labels[i], v, str(v), 
                ha='center', va='bottom', 
                fontsize=16,
                fontweight='bold')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_statistics(data_dict, name):
    """Print dataset statistics"""
    total = sum(data_dict.values())
    print(f"\n{name} Statistics:")
    print(f"Total images: {total}")
    print("\nDetailed distribution:")
    for label in sorted(data_dict.keys()):
        count = data_dict[label]
        percentage = (count / total) * 100
        print(f"Label {label:.1f}: {count} images ({percentage:.1f}%)")

def save_statistics_to_file(data_dict, filename):
    """Save statistics to a txt file"""
    with open(filename, 'w') as f:
        for label in sorted(data_dict.keys()):
            f.write(f"{label:.1f},{data_dict[label]}\n")

def load_statistics_from_file(filename):
    """Load statistics from a txt file"""
    data_dict = defaultdict(int)
    if not os.path.exists(filename):
        return None
    with open(filename, 'r') as f:
        for line in f:
            label, count = line.strip().split(',')
            data_dict[float(label)] = int(count)
    return data_dict

def main():
    # Dataset path (relative to project root)
    dataset_path = '../../dataset/cls_raw'
    
    # Define base path for saving results
    base_path = '.'
    
    # Create directory for saving figures and data
    save_dir = os.path.join(base_path, 'figures')
    os.makedirs(save_dir, exist_ok=True)
    
    # Define paths for statistics files
    train_file = os.path.join(base_path, 'train_statistics.txt')
    val_file = os.path.join(base_path, 'val_statistics.txt')
    all_file = os.path.join(base_path, 'all_statistics.txt')
    
    # Check if statistics files exist
    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(all_file):
        print("Loading statistics from existing files...")
        train_data = load_statistics_from_file(train_file)
        val_data = load_statistics_from_file(val_file)
        all_data = load_statistics_from_file(all_file)
    else:
        print("Collecting statistics from dataset...")
        # Collect data statistics
        train_data, val_data, all_data = collect_data_statistics(dataset_path)
        
        # Save statistics to files
        save_statistics_to_file(train_data, train_file)
        save_statistics_to_file(val_data, val_file)
        save_statistics_to_file(all_data, all_file)
    
    # Plot distributions
    plot_distribution(train_data, 'Training Set Distribution', 
                     os.path.join(save_dir, 'train_distribution.pdf'))
    plot_distribution(val_data, 'Validation Set Distribution', 
                     os.path.join(save_dir, 'val_distribution.pdf'))
    plot_distribution(all_data, 'Complete Dataset Distribution', 
                     os.path.join(save_dir, 'all_distribution.pdf'))
    
    # Print statistics
    print_statistics(train_data, "Training Set")
    print_statistics(val_data, "Validation Set")
    print_statistics(all_data, "Complete Dataset")

if __name__ == '__main__':
    main()