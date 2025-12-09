"""
Quick toggle script for quantization mode
"""
import sys

def toggle_quantization(enable):
    """Toggle quantization in world.py"""
    filepath = 'code/world.py'
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    if enable:
        # Enable quantization
        content = content.replace(
            "config['quantization'] = False",
            "config['quantization'] = True"
        )
        mode = "ENABLED"
    else:
        # Disable quantization
        content = content.replace(
            "config['quantization'] = True",
            "config['quantization'] = False"
        )
        mode = "DISABLED"
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✓ Quantization {mode}")
    print(f"  Mode: {'8-bit Integer Arithmetic' if enable else 'Float32 Standard'}")
    print(f"  File: {filepath}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python toggle_quantization.py on   # Enable quantization")
        print("  python toggle_quantization.py off  # Disable quantization")
        sys.exit(1)
    
    arg = sys.argv[1].lower()
    if arg in ['on', 'enable', 'true', '1']:
        toggle_quantization(True)
    elif arg in ['off', 'disable', 'false', '0']:
        toggle_quantization(False)
    else:
        print(f"Unknown argument: {arg}")
        print("Use: on/off, enable/disable, true/false, or 1/0")
