import torch

def get_torch_backend() -> str:
    """
    Returns the best backend for torch on the executing system.
    For instance, it should return cuda on our linux boxes,
    which all have Nvidia GPUs,
    and mps (Metal Performance Shaders) on our macbooks,
    all of which are Apple Silicon.
    """
    valid_backends = ["cuda", "cpu", "mps"]
    if torch.cuda.is_available():
        y = "cuda"
    elif torch.backends.mps.is_available():
        y = "mps"
    else:
        y = "cpu"
    
    assert y in valid_backends
    
    return y
    

if __name__ == "__main__":
    print(get_torch_backend())