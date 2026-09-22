"""
Hardware Audit Script (Part 4).
Inspects CPU, RAM, GPU, PyTorch device availability, CUDA support, and acceleration backends.
"""
import os
import sys
import platform
import psutil
import torch

def audit_hardware():
    print("========================================")
    print("      SYSTEM HARDWARE AUDIT REPORT      ")
    print("========================================")
    
    cpu_name = platform.processor() or "Unknown CPU"
    cpu_cores = os.cpu_count()
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    ram_avail_gb = round(psutil.virtual_memory().available / (1024**3), 2)
    
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else "None"
    
    pytorch_version = torch.__version__
    default_device = "cuda" if cuda_available else "cpu"

    report = {
        "os_platform": platform.system() + " " + platform.release(),
        "python_version": sys.version.split()[0],
        "pytorch_version": pytorch_version,
        "cpu_processor": cpu_name,
        "cpu_logical_cores": cpu_cores,
        "ram_total_gb": ram_gb,
        "ram_available_gb": ram_avail_gb,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,
        "recommended_pytorch_device": default_device
    }

    for k, v in report.items():
        print(f"  {k:<30}: {v}")

    print("========================================")
    
    os.makedirs("logs", exist_ok=True)
    with open("logs/hardware_audit.json", "w", encoding="utf-8") as f:
        import json
        json.dump(report, f, indent=2)

    return report

if __name__ == "__main__":
    audit_hardware()
