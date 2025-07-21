#!/usr/bin/env python3
"""
Checkpoint Analysis Script
Analyzes the contents of checkpoint files to understand size differences.
"""

import torch
import os
import sys
from pathlib import Path

def format_size(size_bytes):
    """Convert bytes to human readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f}{size_names[i]}"

def get_tensor_info(tensor):
    """Get detailed information about a tensor"""
    if torch.is_tensor(tensor):
        return {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'size_bytes': tensor.numel() * tensor.element_size(),
            'device': str(tensor.device)
        }
    else:
        return {
            'type': type(tensor).__name__,
            'size_bytes': sys.getsizeof(tensor)
        }

def analyze_nested_dict(data, prefix="", max_depth=3, current_depth=0):
    """Recursively analyze nested dictionaries"""
    results = []
    
    if current_depth > max_depth:
        return results
    
    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                results.extend(analyze_nested_dict(value, full_key, max_depth, current_depth + 1))
            elif torch.is_tensor(value):
                tensor_info = get_tensor_info(value)
                results.append({
                    'key': full_key,
                    'type': 'tensor',
                    'shape': tensor_info['shape'],
                    'dtype': tensor_info['dtype'],
                    'size_bytes': tensor_info['size_bytes'],
                    'device': tensor_info['device']
                })
            else:
                results.append({
                    'key': full_key,
                    'type': type(value).__name__,
                    'size_bytes': sys.getsizeof(value),
                    'value_preview': str(value)[:100] if not callable(value) else 'callable'
                })
    
    return results

def analyze_checkpoint(checkpoint_path):
    """Analyze a single checkpoint file"""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {checkpoint_path}")
    print(f"{'='*80}")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: File not found: {checkpoint_path}")
        return None
    
    # Get file size
    file_size = os.path.getsize(checkpoint_path)
    print(f"File size: {format_size(file_size)} ({file_size:,} bytes)")
    
    try:
        # Load checkpoint
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint loaded successfully!")
        
        # Basic information
        print(f"\nTop-level keys: {list(checkpoint.keys())}")
        print(f"Checkpoint type: {type(checkpoint)}")
        
        # Analyze each top-level key
        total_accounted_size = 0
        key_sizes = {}
        
        for key in checkpoint.keys():
            print(f"\n{'-'*60}")
            print(f"KEY: {key}")
            print(f"{'-'*60}")
            
            value = checkpoint[key]
            
            if isinstance(value, dict):
                # Recursively analyze nested dictionary
                nested_results = analyze_nested_dict(value, key)
                
                # Calculate total size for this key
                key_size = sum(item.get('size_bytes', 0) for item in nested_results)
                key_sizes[key] = key_size
                total_accounted_size += key_size
                
                print(f"Type: dict with {len(value)} items")
                print(f"Total size: {format_size(key_size)}")
                
                # Group by type and show largest items
                tensor_items = [item for item in nested_results if item.get('type') == 'tensor']
                other_items = [item for item in nested_results if item.get('type') != 'tensor']
                
                if tensor_items:
                    print(f"\nTensors ({len(tensor_items)} items):")
                    # Sort by size and show top 10
                    tensor_items.sort(key=lambda x: x.get('size_bytes', 0), reverse=True)
                    for i, item in enumerate(tensor_items[:10]):
                        print(f"  {i+1:2d}. {item['key']}: {item['shape']} ({item['dtype']}) - {format_size(item['size_bytes'])}")
                    
                    if len(tensor_items) > 10:
                        print(f"  ... and {len(tensor_items) - 10} more tensors")
                
                if other_items:
                    print(f"\nOther items ({len(other_items)} items):")
                    for item in other_items[:5]:  # Show first 5
                        print(f"  - {item['key']}: {item['type']} - {format_size(item.get('size_bytes', 0))}")
                        if 'value_preview' in item:
                            print(f"    Preview: {item['value_preview']}")
                    
                    if len(other_items) > 5:
                        print(f"  ... and {len(other_items) - 5} more items")
            
            elif torch.is_tensor(value):
                tensor_info = get_tensor_info(value)
                key_sizes[key] = tensor_info['size_bytes']
                total_accounted_size += tensor_info['size_bytes']
                
                print(f"Type: tensor")
                print(f"Shape: {tensor_info['shape']}")
                print(f"Dtype: {tensor_info['dtype']}")
                print(f"Size: {format_size(tensor_info['size_bytes'])}")
                print(f"Device: {tensor_info['device']}")
            
            else:
                item_size = sys.getsizeof(value)
                key_sizes[key] = item_size
                total_accounted_size += item_size
                
                print(f"Type: {type(value).__name__}")
                print(f"Size: {format_size(item_size)}")
                print(f"Preview: {str(value)[:200]}...")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"File size: {format_size(file_size)}")
        print(f"Accounted size: {format_size(total_accounted_size)}")
        print(f"Unaccounted: {format_size(file_size - total_accounted_size)}")
        
        print(f"\nSize breakdown by key:")
        sorted_keys = sorted(key_sizes.items(), key=lambda x: x[1], reverse=True)
        for key, size in sorted_keys:
            percentage = (size / file_size) * 100
            print(f"  {key}: {format_size(size)} ({percentage:.1f}%)")
        
        return {
            'file_size': file_size,
            'total_accounted_size': total_accounted_size,
            'key_sizes': key_sizes,
            'top_level_keys': list(checkpoint.keys())
        }
        
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main analysis function"""
    checkpoint_paths = [
        "/home/rtx4090/Desktop/tmax/johnk/mast3r/output/mast3r-demo-samplerun-wandb-t2/checkpoint-best.pth",
        "/home/rtx4090/Desktop/tmax/johnk/mast3r/output/mast3r-demo-samplerun-wandb-t2/checkpoint-final.pth",
        "/home/rtx4090/Desktop/tmax/johnk/mast3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    ]
    
    results = {}
    
    for checkpoint_path in checkpoint_paths:
        result = analyze_checkpoint(checkpoint_path)
        if result:
            results[checkpoint_path] = result
    
    # Comparative analysis
    print(f"\n{'='*80}")
    print(f"COMPARATIVE ANALYSIS")
    print(f"{'='*80}")
    
    for path, result in results.items():
        filename = os.path.basename(path)
        print(f"\n{filename}:")
        print(f"  File size: {format_size(result['file_size'])}")
        print(f"  Top-level keys: {result['top_level_keys']}")
        
        # Show largest components
        if result['key_sizes']:
            largest_key = max(result['key_sizes'].items(), key=lambda x: x[1])
            print(f"  Largest component: {largest_key[0]} ({format_size(largest_key[1])})")
    
    # Look for unique keys
    if len(results) > 1:
        all_keys = set()
        for result in results.values():
            all_keys.update(result['top_level_keys'])
        
        print(f"\nAll unique top-level keys found: {sorted(all_keys)}")
        
        # Check which keys are in which files
        for key in sorted(all_keys):
            files_with_key = []
            for path, result in results.items():
                if key in result['top_level_keys']:
                    files_with_key.append(os.path.basename(path))
            print(f"  {key}: {files_with_key}")

if __name__ == "__main__":
    main()