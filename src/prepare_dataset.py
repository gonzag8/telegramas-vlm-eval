"""
Script to create a balanced, stratified benchmark dataset from handwritten_numbers_v1.
Extracts 50 representative samples:
- 10 zeros
- 15 single digits (1-9)
- 15 double digits (10-99)
- 10 triple digits (100-303)
Saves images to images/test/ and ground truth metadata to ground_truth/labels.json.
"""

import os
import shutil
import json
import pandas as pd

def prepare_benchmark_dataset(sample_size=50, random_seed=42):
    labels_path = 'handwritten_numbers_v1/labels.csv'
    with open(labels_path, 'r') as f:
        raw_labels = [int(x.strip()) for x in f.read().strip().split(',') if x.strip()]

    df = pd.DataFrame({'idx': range(len(raw_labels)), 'label': raw_labels})
    df['digits'] = df['label'].astype(str).str.len()

    # Stratified sampling
    zeros = df[df['label'] == 0].sample(n=10, random_state=random_seed)
    one_dig = df[(df['digits'] == 1) & (df['label'] != 0)].sample(n=15, random_state=random_seed)
    two_dig = df[df['digits'] == 2].sample(n=15, random_state=random_seed)
    three_dig = df[df['digits'] == 3].sample(n=10, random_state=random_seed)

    sample = pd.concat([zeros, one_dig, two_dig, three_dig]).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Directories
    os.makedirs('images/test', exist_ok=True)
    os.makedirs('ground_truth', exist_ok=True)
    os.makedirs('results/raw', exist_ok=True)

    ground_truth = {}
    for _, row in sample.iterrows():
        orig_idx = int(row['idx'])
        label = int(row['label'])
        src_path = f'handwritten_numbers_v1/{orig_idx}.png'
        dest_filename = f'sample_{orig_idx:04d}.png'
        dest_path = f'images/test/{dest_filename}'
        
        shutil.copy(src_path, dest_path)
        ground_truth[dest_filename] = {
            'original_idx': orig_idx,
            'label': label,
            'num_digits': len(str(label))
        }

    with open('ground_truth/labels.json', 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Benchmark dataset ready: {len(ground_truth)} samples in images/test/")
    print(f"Ground truth saved to ground_truth/labels.json")

if __name__ == '__main__':
    prepare_benchmark_dataset()
