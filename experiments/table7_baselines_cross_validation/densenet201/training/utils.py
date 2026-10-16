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
    """Calcula GFLOPS, Parámetros y Tiempo de Inferencia usando ptflops."""
    
    # 1. Calcular GFLOPS y Parámetros (Usando ptflops)
    try:
        # Nota: ptflops requiere el tamaño sin el batch size (H, W)
        macs, params = get_model_complexity_info(
            model, 
            input_size, 
            as_strings=False, 
            print_per_layer_stat=False, 
            verbose=False
        )
        
        # MACs (Multiply-Accumulate Operations) son a menudo reportados como FLOPS
        gflops = macs / 1e9  
        m_params = params / 1e6 # Convertir Parámetros a Millones
    except Exception as e:
        print(f"⚠️ Error en ptflops: {e}")
        gflops, m_params = 0.0, 0.0

    # 2. Medir Tiempo de Inferencia (Latency - Mismo código que antes)
    model.eval()
    input_tensor = torch.randn(1, *input_size).to(device)
    num_runs = 100 
    
    # [Resto del código de medición de latencia (starter, ender, etc.)]
    if device.type == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # Calentar GPU
        for _ in range(10): _ = model(input_tensor)
        # Medición
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
    Calcula el número total de parámetros y el número de parámetros entrenables.
    """
    total_params = 0
    trainable_params = 0
    
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_params += parameter.numel()
        total_params += parameter.numel()
        
    non_trainable_params = total_params - trainable_params
    
    # Formatear a millones para la salida del artículo (M)
    total_m = total_params / 1e6
    trainable_m = trainable_params / 1e6
    non_trainable_m = non_trainable_params / 1e6

    print("\n📊 Resumen de Parámetros del Modelo:")
    print(f"Total de Parámetros: {total_m} M")
    print(f"Entrenables: {trainable_m} M")
    print(f"No Entrenables: {non_trainable_m} M")
    
    return {
        "Total_M": total_m,
        "Trainable_M": trainable_m,
        "Non_Trainable_M": non_trainable_m,
        "Total_Raw": total_params # Para cálculos exactos
    }

