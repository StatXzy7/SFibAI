from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import os


def collect_data_statistics(root_dir):
    """Collect label statistics from a dataset directory."""
    train_data = defaultdict(int)
    val_data = defaultdict(int)
    all_data = defaultdict(int)

    train_dir = os.path.join(root_dir, 'train')
    for label_folder in os.listdir(train_dir):
        folder_path = os.path.join(train_dir, label_folder)
        if not os.path.isdir(folder_path):
            continue
        label = round(float(label_folder), 1)
        num_images = len([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])
        train_data[label] += num_images
        all_data[label] += num_images

    val_dir = os.path.join(root_dir, 'val')
    for label_folder in os.listdir(val_dir):
        folder_path = os.path.join(val_dir, label_folder)
        if not os.path.isdir(folder_path):
            continue
        label = round(float(label_folder), 1)
        num_images = len([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])
        val_data[label] += num_images
        all_data[label] += num_images

    return train_data, val_data, all_data


def plot_distribution(data_dict, title, save_path):
    """Plot a label distribution histogram."""
    plt.rcParams.update({
        'font.size': 16,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
    })

    plt.figure(figsize=(15, 8))
    labels = sorted(data_dict.keys())
    values = [data_dict[label] for label in labels]

    plt.bar(labels, values, width=0.1)
    plt.title(title, fontsize=32, pad=20, weight='bold')
    plt.xlabel('Fibrosis score', fontsize=28, labelpad=15, weight='bold')
    plt.ylabel('Number of images', fontsize=28, labelpad=15, weight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=24, weight='bold')
    plt.yticks(fontsize=24, weight='bold')

    for i, value in enumerate(values):
        plt.text(labels[i], value, str(value), ha='center', va='bottom', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_statistics(data_dict, name):
    """Print dataset statistics to stdout."""
    total = sum(data_dict.values())
    print(f"\n{name} statistics:")
    print(f"Total images: {total}")
    print("\nDetailed distribution:")
    for label in sorted(data_dict.keys()):
        count = data_dict[label]
        percentage = (count / total) * 100
        print(f"Label {label:.1f}: {count} images ({percentage:.1f}%)")


def save_statistics_to_file(data_dict, filename):
    """Save statistics to a text file."""
    with open(filename, 'w') as handle:
        for label in sorted(data_dict.keys()):
            handle.write(f"{label:.1f},{data_dict[label]}\n")


def load_statistics_from_file(filename):
    """Load statistics from a text file."""
    data_dict = defaultdict(int)
    if not os.path.exists(filename):
        return None
    with open(filename, 'r') as handle:
        for line in handle:
            label, count = line.strip().split(',')
            data_dict[float(label)] = int(count)
    return data_dict


def main():
    repo_root = Path(__file__).resolve().parents[2]
    dataset_path = repo_root / 'data' / 'seg_samples_500'

    save_dir = repo_root / 'artifacts' / 'figures'
    stats_dir = repo_root / 'artifacts' / 'statistics'
    save_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    train_file = stats_dir / 'train_statistics.txt'
    val_file = stats_dir / 'val_statistics.txt'
    all_file = stats_dir / 'all_statistics.txt'

    if train_file.exists() and val_file.exists() and all_file.exists():
        print('Loading statistics from existing files...')
        train_data = load_statistics_from_file(train_file)
        val_data = load_statistics_from_file(val_file)
        all_data = load_statistics_from_file(all_file)
    else:
        print('Collecting statistics from dataset...')
        train_data, val_data, all_data = collect_data_statistics(str(dataset_path))
        save_statistics_to_file(train_data, train_file)
        save_statistics_to_file(val_data, val_file)
        save_statistics_to_file(all_data, all_file)

    plot_distribution(train_data, 'Training set distribution', save_dir / 'train_distribution.pdf')
    plot_distribution(val_data, 'Validation set distribution', save_dir / 'val_distribution.pdf')
    plot_distribution(all_data, 'Complete dataset distribution', save_dir / 'all_distribution.pdf')

    print_statistics(train_data, 'Training set')
    print_statistics(val_data, 'Validation set')
    print_statistics(all_data, 'Complete dataset')


if __name__ == '__main__':
    main()
