# Quantization Performance Comparison Guide

This project has two branches for comparing training efficiency:

## Branches

### 1. `original-code` 
- Original LightGCN implementation
- Uses Float32 embeddings
- Standard uniform negative sampling

### 2. `quantized-version` 
- 8-bit quantization applied to embeddings
- Faster integer-like arithmetic
- Same uniform negative sampling
- Comprehensive performance tracking

## Quick Start

### Train Original Version
```bash
git checkout original-code
python code/main.py --dataset=gowalla --model=lgn --epochs=20
```

### Train Quantized Version
```bash
git checkout quantized-version
python code/main.py --dataset=gowalla --model=lgn --epochs=20
```

### Enable Quantization
In `code/world.py`, set:
```python
config['quantization'] = True  # Enable 8-bit quantization
config['quant_bits'] = 8       # Number of quantization bits
```

## Performance Metrics Tracked

The quantized version automatically tracks:
- ⏱️ **Epoch Time** - Total time per epoch
- 📊 **Sample Time** - Time for negative sampling
- 🔄 **Batch Time** - Average batch processing time
- 💾 **Memory Usage** - RAM consumption in MB
- 🖥️ **CPU Usage** - CPU utilization percentage

## Metrics Output

After training, two files are generated:
- `metrics_original.json` - Original version metrics
- `metrics_quantized.json` - Quantized version metrics

## Compare Performance

After training both versions:
```bash
python compare_performance.py
```

This generates:
- **Console output** with speedup statistics
- **performance_comparison.png** with visualization plots

## Expected Benefits of Quantization

✅ **2-3x faster computation** - Integer operations vs float  
✅ **~75% less memory** - 8-bit vs 32-bit storage  
✅ **Lower energy consumption** - Fewer computations  
✅ **Minimal accuracy loss** - Typically <2% difference  

## Training Output Example

```
EPOCH[1/20] loss0.234-Sample:2.1s-batch:0.0123s-epoch:15.32s-mem:450.2MB-cpu:45.3%-QUANT
```

Indicators:
- `FLOAT` - Original Float32 mode
- `QUANT` - Quantized 8-bit mode

## Notes

- Dynamic negative sampling was **removed** as it caused overfitting
- Reverted to uniform random sampling for stability
- Quantization is applied in the `computer()` method of LightGCN
- CPU/Memory tracking uses `psutil` library (install: `pip install psutil`)

## Troubleshooting

If you get import errors:
```bash
pip install psutil matplotlib
```

If metrics files are not generated, check that training completes successfully and the `finally` block executes.
