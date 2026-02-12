import torch
import os

def check_hardware():
    launch_mode = os.environ.get("LAUNCH_MODE", "interactive")
    if launch_mode == "interactive": return 1, 8
    num_gpus = int(os.getenv("NUM_GPUS", 1))
    num_cpus = int(os.getenv("NUM_CPUS", 4))
    print("Visible devices:", torch.cuda.device_count())
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"[Rank {local_rank}] Using GPU: {torch.cuda.current_device()}")
    assert torch.cuda.device_count() == num_gpus, f"Expected {num_gpus} GPUs, but found {torch.cuda.device_count()}"
    return num_gpus, num_cpus
