import platform
import psutil
import torch
import os

def get_system_info():
    print("=== System Information for Report ===\n")

    # OS info
    os_info = platform.platform()
    print(f"Operating System: {os_info}")

    # CPU info
    cpu_name = platform.processor() or "Unknown"
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq().max
    print(f"CPU: {cpu_name}")
    print(f"CPU Cores: {cpu_cores}")
    print(f"CPU Threads: {cpu_threads}")
    print(f"Max CPU Frequency: {cpu_freq:.2f} MHz")

    # RAM info
    ram = psutil.virtual_memory().total / (1024 ** 3)
    print(f"RAM: {ram:.2f} GB")

    # GPU info (if available)
    print("\n=== GPU Information ===")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_mem:.2f} GB")
    else:
        print("No GPU detected or CUDA not available.")

    # Python & Torch
    print("\n=== Software Versions ===")
    print(f"Python version: {platform.python_version()}")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

if __name__ == "__main__":
    get_system_info()
