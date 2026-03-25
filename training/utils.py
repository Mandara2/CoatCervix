import torch
import random
import numpy as np
import gc
import time
from ptflops import get_model_complexity_info

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_cuda():
    torch.cuda.empty_cache()
    gc.collect()


def calculate_efficiency_metrics(model, input_size=(3, 224, 224), device='cuda'):
    """Calculate GFLOPS, number of parameters, and latency for a given model."""
    
    # 1. Calculate GFLOPS and number of parameters using ptflops
    try:
        # Nota: ptflops need that the model are in the GPU, otherwise it can give errors with some layers (ej. LayerNorm)
        macs, params = get_model_complexity_info(
            model, 
            input_size, 
            as_strings=False, 
            print_per_layer_stat=False, 
            verbose=False
        )
        
        # MACs (Multiply-Accumulate Operations) are frecuently used to calculate FLOPS (Floating Point Operations Per Second).
        gflops = macs / 1e9  
        m_params = params / 1e6 # Convert parameters to millions for easier interpretation
    except Exception as e:
        print(f"⚠️ Error en ptflops: {e}")
        gflops, m_params = 0.0, 0.0

    # 2. measure times of latency (inference time) using torch.cuda.Event for GPU and time.time() for CPU
    model.eval()
    input_tensor = torch.randn(1, *input_size).to(device)
    num_runs = 100 
    
    if device.type == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(10): _ = model(input_tensor)
        starter.record()
        for _ in range(num_runs): _ = model(input_tensor)
        ender.record()
        torch.cuda.synchronize()
        latency_ms = starter.elapsed_time(ender) / num_runs
    else:
        start_time = time.time()
        for _ in range(num_runs): _ = model(input_tensor)
        latency_ms = (time.time() - start_time) * 1000 / num_runs
        

    print(f"⚡️ [Eficiencia] GFLOPS: {gflops:.2f}, Parámetros (M): {m_params:.2f}, Latencia (ms): {latency_ms:.2f}")
    return gflops, m_params, latency_ms

def get_parameter_summary(model):
    """
    Calculate the total number of parameters and the number of trainable parameters.
    """
    total_params = 0
    trainable_params = 0
    
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_params += parameter.numel()
        total_params += parameter.numel()
        
    non_trainable_params = total_params - trainable_params
    
    total_m = total_params / 1e6
    trainable_m = trainable_params / 1e6
    non_trainable_m = non_trainable_params / 1e6

    print("\n📊 Parameter Summary of the Model:")
    print(f"Total Parameters: {total_m} M")
    print(f"Trainable: {trainable_m} M")
    print(f"Non-Trainable: {non_trainable_m} M")

    return {
        "Total_M": total_m,
        "Trainable_M": trainable_m,
        "Non_Trainable_M": non_trainable_m,
        "Total_Raw": total_params 
    }

