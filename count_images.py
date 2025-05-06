#!/usr/bin/env python3
import os
import json
import concurrent.futures
from PIL import Image
import argparse
from collections import defaultdict
from pathlib import Path
import sys
from tqdm import tqdm

def get_image_size(image_path):
    """Get the dimensions of an image"""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        return None  # Skip invalid images

def is_valid_image_for_dataset(image_path, dataset_config):
    """Check if an image meets the requirements for a dataset"""
    # Skip non-image files (basic extension check)
    if not any(image_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']):
        return False
    
    # If the dataset is not an image dataset, skip
    if dataset_config.get("type") != "local" or dataset_config.get("dataset_type") == "text_embeds":
        return False
    
    # Check image dimensions
    size_result = get_image_size(image_path)
    if size_result is None:
        return False  # Skip corrupted images
    
    width, height = size_result
    
    # For center cropping, consider the minimum dimension
    min_dimension = min(width, height)
    
    # Apply dataset criteria
    min_size = dataset_config.get("minimum_image_size")
    max_size = dataset_config.get("maximum_image_size")
    
    # Check minimum size if specified
    if min_size and min_dimension < min_size:
        return False
    
    # Check maximum size if specified
    if max_size and min_dimension > max_size:
        return False
    
    return True

def count_images_for_dataset(dataset_config, shared_image_cache=None):
    """Count images that match the criteria for a dataset"""
    # Skip non-image datasets
    if dataset_config.get("dataset_type") == "text_embeds" or dataset_config.get("type") != "local":
        return dataset_config["id"], 0
    
    instance_dir = dataset_config.get("instance_data_dir")
    if not instance_dir or not os.path.exists(instance_dir):
        return dataset_config["id"], 0
    
    # Use the shared cache or process all files
    if shared_image_cache is not None:
        valid_images = [path for path in shared_image_cache 
                       if is_valid_image_for_dataset(path, dataset_config)]
        return dataset_config["id"], len(valid_images)
    
    # Otherwise, walk the directory
    valid_count = 0
    for root, _, files in os.walk(instance_dir):
        for file in files:
            image_path = os.path.join(root, file)
            if is_valid_image_for_dataset(image_path, dataset_config):
                valid_count += 1
    
    return dataset_config["id"], valid_count

def get_all_image_paths(base_dir):
    """Get all image file paths in the directory"""
    image_paths = []
    
    print(f"Finding all images in {base_dir}...")
    for root, _, files in os.walk(base_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']):
                image_paths.append(os.path.join(root, file))
    
    return image_paths

def main():
    parser = argparse.ArgumentParser(description='Count images in datasets based on configuration')
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to the dataset configuration JSON file')
    parser.add_argument('--parallel', action='store_true', 
                        help='Use parallel processing for faster counting')
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            datasets = json.load(f)
    else:
        # If no config file provided, try to read from stdin
        print("No config file provided, expecting JSON input from stdin...")
        try:
            datasets = json.loads(sys.stdin.read())
        except json.JSONDecodeError:
            print("Error: Please provide a valid JSON configuration")
            return
    
    # Check if we have a list of datasets
    if not isinstance(datasets, list):
        print("Error: Expected a list of dataset configurations")
        return
    
    # Find all unique base directories to optimize processing
    base_directories = set()
    for dataset in datasets:
        if dataset.get("type") == "local" and dataset.get("dataset_type") != "text_embeds":
            instance_dir = dataset.get("instance_data_dir")
            if instance_dir:
                base_directories.add(instance_dir)
    
    # Collect and cache all image paths for better performance
    directory_image_cache = {}
    for base_dir in base_directories:
        directory_image_cache[base_dir] = get_all_image_paths(base_dir)
    
    # Count images per dataset
    results = []
    image_datasets = [d for d in datasets if d.get("type") == "local" and d.get("dataset_type") != "text_embeds"]
    
    print(f"Counting images for {len(image_datasets)} datasets...")
    
    if args.parallel:
        # Parallel processing 
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for dataset in image_datasets:
                instance_dir = dataset.get("instance_data_dir")
                image_cache = directory_image_cache.get(instance_dir, [])
                futures.append(executor.submit(count_images_for_dataset, dataset, image_cache))
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                results.append(future.result())
    else:
        # Sequential processing
        for dataset in tqdm(image_datasets):
            instance_dir = dataset.get("instance_data_dir")
            image_cache = directory_image_cache.get(instance_dir, [])
            results.append(count_images_for_dataset(dataset, image_cache))
    
    # Print results
    print("\nImage count per dataset:")
    print("-" * 50)
    print(f"{'Dataset ID':<20} | {'Count':<10} | {'Probability':<10}")
    print("-" * 50)
    
    total_images = 0
    for dataset_id, count in sorted(results):
        dataset = next((d for d in datasets if d["id"] == dataset_id), None)
        prob = dataset.get("probability", "N/A")
        print(f"{dataset_id:<20} | {count:<10} | {prob}")
        total_images += count
    
    print("-" * 50)
    print(f"Total: {total_images} images across {len(results)} image datasets")

if __name__ == "__main__":
    main()