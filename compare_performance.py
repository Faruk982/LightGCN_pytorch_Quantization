"""
Performance Comparison Script
Compare training efficiency between original and quantized versions
"""
import json
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(filename='metrics.json'):
    """Load metrics from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Metrics file {filename} not found")
        return None

def compare_metrics(original_metrics, quantized_metrics):
    """Compare and visualize performance metrics"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Performance Comparison: Original vs Quantized', fontsize=16)
    
    metrics = [
        ('epoch_times', 'Epoch Time (s)', 0, 0),
        ('sample_times', 'Sample Time (s)', 0, 1),
        ('batch_times', 'Avg Batch Time (s)', 0, 2),
        ('memory_usage', 'Memory Usage (MB)', 1, 0),
        ('cpu_usage', 'CPU Usage (%)', 1, 1)
    ]
    
    for metric_name, ylabel, row, col in metrics:
        ax = axes[row, col]
        if original_metrics and metric_name in original_metrics:
            ax.plot(original_metrics[metric_name], label='Original', marker='o', alpha=0.7)
        if quantized_metrics and metric_name in quantized_metrics:
            ax.plot(quantized_metrics[metric_name], label='Quantized', marker='s', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Summary statistics in the last subplot
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "Summary Statistics\n" + "="*30 + "\n\n"
    
    if original_metrics and quantized_metrics:
        for metric_name, label, _, _ in metrics[:3]:  # Time metrics
            if metric_name in original_metrics and metric_name in quantized_metrics:
                orig_avg = np.mean(original_metrics[metric_name])
                quant_avg = np.mean(quantized_metrics[metric_name])
                speedup = (orig_avg / quant_avg - 1) * 100
                summary_text += f"{label}:\n"
                summary_text += f"  Original: {orig_avg:.4f}\n"
                summary_text += f"  Quantized: {quant_avg:.4f}\n"
                summary_text += f"  Speedup: {speedup:+.2f}%\n\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
    print("Performance comparison plot saved to 'performance_comparison.png'")
    plt.show()

if __name__ == "__main__":
    print("Loading metrics...")
    original = load_metrics('metrics_original.json')
    quantized = load_metrics('metrics_quantized.json')
    
    if original or quantized:
        compare_metrics(original, quantized)
        print("\nComparison complete!")
    else:
        print("\nNo metrics files found. Run training first to generate metrics.")
        print("Metrics will be saved automatically during training.")
