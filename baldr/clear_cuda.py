import torch

BYTES_IN_MB = 1024**2

def print_gpu_memory(prefix=""):
    allocated = torch.cuda.memory_allocated() / BYTES_IN_MB
    reserved = torch.cuda.memory_reserved() / BYTES_IN_MB
    print(f"{prefix}Allocated: {allocated:.2f} MB | Reserved {reserved:.2f}")

if torch.cuda.is_available():
    print_gpu_memory("Before clearing cache - ")

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    print_gpu_memory("After clearing cache - ")
else:
    print("CUDA is not available.")
