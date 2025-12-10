# 🚀 Google Colab - Simple Run Instructions

## Quick Start (Copy-Paste into Colab)

```python
# ===== STEP 1: Setup =====
!git clone https://github.com/Faruk982/LightGCN_pytorch_Quantization.git
%cd LightGCN_pytorch_Quantization
!pip install -q torch scikit-learn tensorboardX psutil pandas

# ===== STEP 2: Run ORIGINAL Version (Float32) =====
print("\n" + "="*60)
print("RUNNING ORIGINAL (Float32)")
print("="*60)
!git checkout original-code
!python code/main.py --dataset=gowalla --model=lgn --epochs=5 --bpr_batch=2048

# ===== STEP 3: Run QUANTIZED Version (8-bit) =====
print("\n" + "="*60)
print("RUNNING QUANTIZED (8-bit)")
print("="*60)
!git checkout quantized-version
!python code/main.py --dataset=gowalla --model=lgn --epochs=5 --bpr_batch=2048

print("\n✅ DONE! Check the output above to compare:")
print("  - Accuracy: Recall@20, Precision@20, NDCG@20")
print("  - Speed: epoch time, batch time")
print("  - Resources: memory, CPU usage")
```

## What You'll See

### During Training:
```
EPOCH[1/5] loss0.234-|Sample:2.1s|-batch:0.0123s-epoch:45.67s-mem:1024MB-cpu:67.3%-FLOAT
```

### At the End:
```
============================================================
FINAL SUMMARY - ORIGINAL MODE
============================================================
Avg Epoch Time:  45.32s
Avg Batch Time:  0.0125s
Avg Memory:      1024.5 MB
Avg CPU:         67.2%
============================================================
```

### Test Results (Every 10 Epochs):
```
[Recall@20:0.015234|Precision@20:0.012456|NDCG@20:0.023456]
```

## Datasets by Speed

| Dataset | Items | Users | Time/Epoch | Recommended Epochs |
|---------|-------|-------|------------|-------------------|
| **gowalla** | 29K | 29K | ~1-2 min | 5-10 ⚡ Best for quick test |
| **yelp2018** | 45K | 31K | ~3-5 min | 10-20 |
| **amazon-book** | 59K | 52K | ~8-10 min | 20+ |

## Advanced: Faster Test

For ultra-quick test (~5 minutes total):

```python
# Both versions with 3 epochs
!git checkout original-code
!python code/main.py --dataset=gowalla --epochs=3 --bpr_batch=4096

!git checkout quantized-version
!python code/main.py --dataset=gowalla --epochs=3 --bpr_batch=4096
```

## Compare Results

Look for these in the output:

**Speed Comparison:**
- `epoch:XX.XXs` - How long each epoch takes
- `batch:X.XXXXs` - How long each batch takes
- **Quantized should be 2-3x faster!** ⚡

**Accuracy Comparison:**
- `Recall@20` - Higher is better
- `NDCG@20` - Higher is better
- **Quantized should be ~98-99% of original** 📊

**Resource Usage:**
- `mem:XXXmb` - Memory usage
- `cpu:XX%` - CPU utilization
- **Quantized should use less memory** 💾

## Troubleshooting

**Out of Memory?**
```python
# Reduce batch size
--bpr_batch=1024
```

**Dataset not found?**
```python
# Use gowalla (most reliable)
--dataset=gowalla
```

**Want to see only final results?**
```python
# Reduce output, just run and compare the FINAL SUMMARY sections
```

---

**That's it! 🎉** Just copy-paste the code above and run in Colab. You'll see both versions train and can compare speed vs accuracy directly from the console output!


## Step 1: Setup Environment

```python
# Clone repository
!git clone https://github.com/Faruk982/LightGCN_pytorch_Quantization.git
%cd LightGCN_pytorch_Quantization

# Install dependencies
!pip install -q torch scikit-learn tensorboardX psutil

# Check GPU availability
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Step 2: Run ORIGINAL Version (Float32 Baseline)

```python
# Switch to original branch
!git checkout original-code

# Run with small epochs for quick test
!python code/main.py --dataset=gowalla --model=lgn --epochs=5 --bpr_batch=2048

# For faster results, use smaller dataset
# !python code/main.py --dataset=lastfm --model=lgn --epochs=5
```

**Expected time:** ~2-3 minutes per epoch on Gowalla

## Step 3: Run QUANTIZED Version (8-bit)

```python
# Switch to quantized branch
!git checkout quantized-version

# Enable quantization (already enabled by default)
# Verify it's on
!grep "config\['quantization'\]" code/world.py

# Run with same settings
!python code/main.py --dataset=gowalla --model=lgn --epochs=5 --bpr_batch=2048
```

**Expected time:** ~1-1.5 minutes per epoch (2x faster!)

## Step 4: Compare Results

```python
import json
import numpy as np

# Load metrics
with open('metrics_original.json') as f:
    original = json.load(f)

with open('metrics_quantized.json') as f:
    quantized = json.load(f)

# Speed comparison
orig_time = np.mean(original['metrics']['epoch_times'])
quant_time = np.mean(quantized['metrics']['epoch_times'])
speedup = ((orig_time - quant_time) / orig_time) * 100

print("=" * 60)
print("⚡ PERFORMANCE COMPARISON")
print("=" * 60)
print(f"Original (Float32):  {orig_time:.2f}s per epoch")
print(f"Quantized (8-bit):   {quant_time:.2f}s per epoch")
print(f"Speedup:             {speedup:.1f}% faster")
print(f"Memory Reduction:    {((np.mean(original['metrics']['memory_usage']) - np.mean(quantized['metrics']['memory_usage'])) / np.mean(original['metrics']['memory_usage']) * 100):.1f}%")
print()

# Accuracy comparison (last test epoch)
if original['metrics']['recall'] and quantized['metrics']['recall']:
    orig_recall = original['metrics']['recall'][-1]
    quant_recall = quantized['metrics']['recall'][-1]
    orig_ndcg = original['metrics']['ndcg'][-1]
    quant_ndcg = quantized['metrics']['ndcg'][-1]
    
    print("=" * 60)
    print("📊 ACCURACY COMPARISON")
    print("=" * 60)
    print(f"Recall@20:")
    print(f"  Original:  {orig_recall[0]:.6f}")
    print(f"  Quantized: {quant_recall[0]:.6f}")
    print(f"  Retention: {(quant_recall[0]/orig_recall[0]*100):.2f}%")
    print()
    print(f"NDCG@20:")
    print(f"  Original:  {orig_ndcg[0]:.6f}")
    print(f"  Quantized: {quant_ndcg[0]:.6f}")
    print(f"  Retention: {(quant_ndcg[0]/orig_ndcg[0]*100):.2f}%")
    print()

# Loss comparison
print("=" * 60)
print("📉 TRAINING LOSS")
print("=" * 60)
print(f"Final Loss:")
print(f"  Original:  {original['metrics']['losses'][-1]:.4f}")
print(f"  Quantized: {quantized['metrics']['losses'][-1]:.4f}")
print("=" * 60)
```

## ⚡ Quick Test (Ultra Fast)

For very quick testing (under 5 minutes total):

```python
# Test on smallest dataset with 3 epochs
!git checkout original-code
!python code/main.py --dataset=lastfm --model=lgn --epochs=3 --bpr_batch=4096

!git checkout quantized-version
!python code/main.py --dataset=lastfm --model=lgn --epochs=3 --bpr_batch=4096
```

## 📊 Visualize Results

```python
import matplotlib.pyplot as plt

# Plot training speed
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Epoch times
axes[0].plot(original['metrics']['epoch_times'], label='Original', marker='o', linewidth=2)
axes[0].plot(quantized['metrics']['epoch_times'], label='Quantized', marker='s', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Time (seconds)', fontsize=12)
axes[0].set_title('Training Speed Comparison', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Accuracy (Recall)
if original['metrics']['recall']:
    orig_recalls = [r[0] for r in original['metrics']['recall']]
    quant_recalls = [r[0] for r in quantized['metrics']['recall']]
    test_epochs = range(len(orig_recalls))
    
    axes[1].plot(test_epochs, orig_recalls, label='Original', marker='o', linewidth=2)
    axes[1].plot(test_epochs, quant_recalls, label='Quantized', marker='s', linewidth=2)
    axes[1].set_xlabel('Test Point', fontsize=12)
    axes[1].set_ylabel('Recall@20', fontsize=12)
    axes[1].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary statistics
print("\n🎯 SUMMARY")
print(f"Speed Improvement: {speedup:.1f}% faster")
if original['metrics']['recall']:
    acc_retention = (quant_recalls[-1] / orig_recalls[-1]) * 100
    print(f"Accuracy Retention: {acc_retention:.2f}%")
    print(f"\n✅ Trade-off: {speedup:.1f}% faster with {100-acc_retention:.2f}% accuracy loss")
```

## 🎯 Dataset Recommendations by Speed

| Dataset | Size | Epochs | Time per Epoch | Total Time |
|---------|------|--------|----------------|------------|
| **lastfm** | Small | 3-5 | ~30s | 2-3 min ⚡ |
| **gowalla** | Medium | 5-10 | ~90s | 8-15 min |
| **yelp2018** | Large | 10-20 | ~3-5 min | 30-100 min |
| **amazon-book** | XLarge | 20+ | ~8-10 min | 3+ hours |

**For quick testing:** Use `lastfm` with 3-5 epochs  
**For balanced results:** Use `gowalla` with 5-10 epochs  
**For best accuracy:** Use `yelp2018` with 20 epochs

## 🔧 Adjust Settings for Speed

To make training even faster:

```python
# Increase batch size (uses more memory but faster)
!python code/main.py --dataset=gowalla --model=lgn --epochs=5 --bpr_batch=4096

# Reduce layers (faster but less accurate)
!python code/main.py --dataset=gowalla --model=lgn --epochs=5 --layer=2

# Reduce embedding dimension
!python code/main.py --dataset=gowalla --model=lgn --epochs=5 --recdim=32
```

## 💾 Download Results

```python
# Download metrics files to your computer
from google.colab import files

files.download('metrics_original.json')
files.download('metrics_quantized.json')
```

## 🐛 Troubleshooting

**Out of Memory?**
```python
# Reduce batch size
--bpr_batch=1024
```

**Too slow?**
```python
# Use smaller dataset
--dataset=lastfm
# or reduce epochs
--epochs=3
```

**Quantization not working?**
```python
# Check setting
!cat code/world.py | grep quantization
# Should show: config['quantization'] = True
```

---

## 🎬 Complete Single Cell Script (Copy-Paste)

```python
# Complete automated comparison in one cell
!git clone https://github.com/Faruk982/LightGCN_pytorch_Quantization.git
%cd LightGCN_pytorch_Quantization
!pip install -q torch scikit-learn tensorboardX psutil

# Run original
!git checkout original-code
!python code/main.py --dataset=gowalla --model=lgn --epochs=5 --bpr_batch=2048

# Run quantized
!git checkout quantized-version  
!python code/main.py --dataset=gowalla --model=lgn --epochs=5 --bpr_batch=2048

# Compare
import json
import numpy as np

with open('metrics_original.json') as f:
    orig = json.load(f)
with open('metrics_quantized.json') as f:
    quant = json.load(f)

speedup = ((np.mean(orig['metrics']['epoch_times']) - np.mean(quant['metrics']['epoch_times'])) / np.mean(orig['metrics']['epoch_times'])) * 100
print(f"\n🎯 RESULT: {speedup:.1f}% FASTER with quantization!")

if orig['metrics']['recall']:
    acc = (quant['metrics']['recall'][-1][0] / orig['metrics']['recall'][-1][0]) * 100
    print(f"📊 Accuracy retention: {acc:.2f}%")
```

---

**Ready to test! 🚀**

Use the quick test with `lastfm` dataset for fastest results (~5 minutes total).
