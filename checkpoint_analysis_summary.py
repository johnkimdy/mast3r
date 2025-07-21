#!/usr/bin/env python3
"""
Checkpoint Analysis Summary
Key findings from the checkpoint analysis.
"""

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

def main():
    print("CHECKPOINT SIZE ANALYSIS SUMMARY")
    print("=" * 60)
    
    # File sizes
    files = {
        "checkpoint-best.pth": 7.68e9,
        "checkpoint-final.pth": 2.57e9,
        "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth": 2.13e9
    }
    
    print("\nFile Sizes:")
    for filename, size in files.items():
        print(f"  {filename}: {format_size(size)}")
    
    print("\nKEY FINDINGS:")
    print("-" * 40)
    
    print("\n1. OPTIMIZER STATE IS THE CULPRIT:")
    print("   - checkpoint-best.pth contains optimizer state (5.11GB)")
    print("   - checkpoint-final.pth does NOT contain optimizer state")
    print("   - This explains the 5GB size difference!")
    
    print("\n2. COMPONENT BREAKDOWN:")
    print("   checkpoint-best.pth (7.68GB):")
    print("     - optimizer: 5.11GB (66.6%)")
    print("     - model: 2.59GB (33.7%)")
    print("     - scaler: 128B (training scaler state)")
    print("     - metadata: epoch, best_so_far, args")
    
    print("\n   checkpoint-final.pth (2.57GB):")
    print("     - model: 2.59GB (100.9%)")
    print("     - metadata: epoch, best_so_far, args")
    print("     - NO optimizer state")
    print("     - NO scaler state")
    
    print("\n   DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth (2.13GB):")
    print("     - model: 2.15GB (101.1%)")
    print("     - args: 56B")
    print("     - NO training state")
    
    print("\n3. WHAT'S IN THE OPTIMIZER STATE:")
    print("   - Contains 3,000 tensors (2x the model parameters)")
    print("   - exp_avg: exponential moving average of gradients")
    print("   - exp_avg_sq: exponential moving average of squared gradients")
    print("   - These are Adam optimizer momentum terms")
    print("   - Each parameter has 2 corresponding optimizer states")
    print("   - Total: ~1,500 model parameters → 3,000 optimizer tensors")
    
    print("\n4. WHY THE DIFFERENCE:")
    print("   - checkpoint-best.pth: TRAINING checkpoint")
    print("     * Saved during training for resuming")
    print("     * Contains full optimizer state")
    print("     * Contains gradient scaler state")
    print("     * Can resume training from this point")
    
    print("\n   - checkpoint-final.pth: INFERENCE checkpoint")
    print("     * Saved at end of training")
    print("     * Contains only model weights")
    print("     * Suitable for inference/deployment")
    print("     * Cannot resume training")
    
    print("\n   - DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth: PRETRAINED checkpoint")
    print("     * Original pretrained model")
    print("     * Contains only model weights")
    print("     * Ready for fine-tuning or inference")
    
    print("\n5. STORAGE RECOMMENDATIONS:")
    print("   - For deployment: Use checkpoint-final.pth (2.57GB)")
    print("   - For resuming training: Use checkpoint-best.pth (7.68GB)")
    print("   - For archival: Keep checkpoint-final.pth, delete checkpoint-best.pth")
    print("   - Space savings: Delete checkpoint-best.pth saves 5.11GB")
    
    print("\n" + "=" * 60)
    print("CONCLUSION: The size difference is entirely due to optimizer state")
    print("stored in checkpoint-best.pth but not in checkpoint-final.pth")
    print("=" * 60)

if __name__ == "__main__":
    main()