import platform
import psutil
import torch
import cpuinfo  # pip install py-cpuinfo

def get_system_info():
    print("# === System Information for Report ===\n")

    # OS info
    os_info = platform.system() + " " + platform.release() + f" (Build {platform.version()})"
    print(f"🖥️ Operating System: {os_info}")

    # CPU info
    cpu_info = cpuinfo.get_cpu_info()
    cpu_name = cpu_info.get('brand_raw', 'Unknown')
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq().max
    print(f"🧠 CPU: {cpu_name}")
    print(f"🔢 Cores: {cpu_cores}")
    print(f"🔁 Threads: {cpu_threads}")
    print(f"⚡ Max Frequency: {cpu_freq:.2f} MHz")

    # RAM info
    ram = psutil.virtual_memory().total / (1024 ** 3)
    print(f"💾 RAM: {ram:.2f} GB")

    # GPU info (if available)
    print("\n# === GPU Information ===")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"🎮 GPU: {gpu_name}")
        print(f"📦 GPU Memory: {gpu_mem:.2f} GB")
    else:
        print("❌ No GPU detected or CUDA not available.")

    # Python & Torch versions
    print("\n# === Software Versions ===")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"🔥 Torch: {torch.__version__}")
    print(f"🧪 CUDA Available: {torch.cuda.is_available()}")

if __name__ == "__main__":
    get_system_info()
