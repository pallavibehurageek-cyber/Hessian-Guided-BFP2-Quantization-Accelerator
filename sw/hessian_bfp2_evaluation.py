"""
unified_fp2_eval.py

Master BFP2 (FP2 x FP2) evaluation suite for both CNNs (ResNets) and LLMs (LLaMAs).
Evaluates all permutations of:
  - Encoding: E1M0 vs E0M1
  - Grouping: Spatial vs Hessian K-Means
  - Boundaries: First/Last layers quantized vs unquantized

Includes VRAM optimizations to prevent OOM errors on massive LLM layers, 
and generates accuracy comparison plots for both CNN fine-tuning and LLM zero-shot tests.
"""

import argparse
import gc
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    HAS_LLM_LIBS = True
except ImportError:
    HAS_LLM_LIBS = False

# ==========================================================================
# 0. Logging & Constants
# ==========================================================================

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger()

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

BLOCK_SIZE = 32          
PAIRS_PER_BLOCK = BLOCK_SIZE // 2   
EXP_WIDTH = 8             
HW_EXP_BIAS = 128         

# ==========================================================================
# 1. Memory-Safe Grouping Algorithms
# ==========================================================================

def spatial_grouping(tensor: torch.Tensor, block_size: int = BLOCK_SIZE) -> List[List[int]]:
    n = tensor.numel()
    indices = np.arange(n)
    return [indices[i : i + block_size].tolist() for i in range(0, n, block_size)]

@torch.no_grad()
def hessian_kmeans_grouping(
    weights: torch.Tensor, hessian: torch.Tensor, block_size: int = BLOCK_SIZE, max_iters: int = 20,
) -> List[List[int]]:
    device = weights.device
    n = weights.numel()
    
    # [VRAM FIX 1] Recursive Chunking for Massive Tensors
    CHUNK_SIZE = 16777216  # 16M elements
    if n > CHUNK_SIZE:
        all_groups = []
        for i in range(0, n, CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, n)
            w_chunk = weights.view(-1)[i:end]
            h_chunk = hessian.view(-1)[i:end]
            chunk_groups = hessian_kmeans_grouping(w_chunk, h_chunk, block_size, max_iters)
            for g in chunk_groups:
                all_groups.append([idx + i for idx in g])
        return all_groups

    flat_w = weights.view(-1).to(torch.float32)
    flat_h = hessian.view(-1).to(torch.float32)

    w_std = torch.std(flat_w) + 1e-8
    h_std = torch.std(flat_h) + 1e-8
    flat_h = flat_h * (w_std / h_std)
    flat_h = torch.log1p(flat_h * 100)

    # [VRAM FIX 2] Quantile Subsampling
    MAX_QUANTILE_ELEMENTS = 16000000
    if flat_h.numel() > MAX_QUANTILE_ELEMENTS:
        sample_indices = torch.randint(0, flat_h.numel(), (MAX_QUANTILE_ELEMENTS,), device=device)
        sampled_h = flat_h[sample_indices]
        h_min_q, h_max_q = torch.quantile(sampled_h, torch.tensor([0.05, 0.95], device=device))
    else:
        h_min_q, h_max_q = torch.quantile(flat_h, torch.tensor([0.05, 0.95], device=device))

    flat_h = torch.clamp(flat_h, h_min_q, h_max_q)
    h_min, h_max = torch.min(flat_h), torch.max(flat_h)
    
    if h_max > h_min:
        flat_h = (flat_h - h_min) / (h_max - h_min) + 0.01
    else:
        flat_h = torch.ones_like(flat_h)

    pad = (block_size - n % block_size) % block_size
    if pad > 0:
        flat_w = F.pad(flat_w, (0, pad), value=0.0)
        flat_h = F.pad(flat_h, (0, pad), value=0.0)
        
    sorted_w, order = torch.sort(flat_w)
    sorted_h = flat_h[order]

    num_blocks = flat_w.numel() // block_size
    w_blocks = sorted_w.view(num_blocks, block_size).clone()
    h_blocks = sorted_h.view(num_blocks, block_size).clone()
    idx_blocks = order.view(num_blocks, block_size).clone()

    def compute_cost(w, h):
        sum_h = h.sum(dim=1, keepdim=True)
        sum_h = torch.where(sum_h == 0, torch.tensor(1e-9, device=device), sum_h)
        centroid = (h * w).sum(dim=1, keepdim=True) / sum_h
        return (h * (w - centroid) ** 2).sum(dim=1)

    for _ in range(max_iters):
        changed = False
        def attempt_swap(swap_size):
            nonlocal changed
            for offset in [0, 1]:
                left_idx = torch.arange(offset, num_blocks - 1, 2, device=device)
                if len(left_idx) == 0: continue
                right_idx = left_idx + 1

                w_left, h_left = w_blocks[left_idx], h_blocks[left_idx]
                w_right, h_right = w_blocks[right_idx], h_blocks[right_idx]

                old_cost = compute_cost(w_left, h_left) + compute_cost(w_right, h_right)

                w_left_test, h_left_test = w_left.clone(), h_left.clone()
                w_right_test, h_right_test = w_right.clone(), h_right.clone()

                w_left_test[:, -swap_size:] = w_right[:, :swap_size]
                h_left_test[:, -swap_size:] = h_right[:, :swap_size]
                w_right_test[:, :swap_size] = w_left[:, -swap_size:]
                h_right_test[:, :swap_size] = h_left[:, -swap_size:]

                new_cost = compute_cost(w_left_test, h_left_test) + compute_cost(w_right_test, h_right_test)
                improve_mask = new_cost < old_cost

                if improve_mask.any():
                    changed = True
                    w_blocks[left_idx[improve_mask]] = w_left_test[improve_mask]
                    h_blocks[left_idx[improve_mask]] = h_left_test[improve_mask]
                    w_blocks[right_idx[improve_mask]] = w_right_test[improve_mask]
                    h_blocks[right_idx[improve_mask]] = h_right_test[improve_mask]
                    
                    idx_left, idx_right = idx_blocks[left_idx], idx_blocks[right_idx]
                    idx_left_test, idx_right_test = idx_left.clone(), idx_right.clone()
                    idx_left_test[:, -swap_size:] = idx_right[:, :swap_size]
                    idx_right_test[:, :swap_size] = idx_left[:, -swap_size:]
                    idx_blocks[left_idx[improve_mask]] = idx_left_test[improve_mask]
                    idx_blocks[right_idx[improve_mask]] = idx_right_test[improve_mask]

        attempt_swap(1)
        attempt_swap(2)
        if not changed: break

    valid_mask = idx_blocks < n
    grouping = []
    idx_blocks_cpu = idx_blocks.cpu().numpy()
    valid_mask_cpu = valid_mask.cpu().numpy()
    
    for i in range(num_blocks):
        block_indices = idx_blocks_cpu[i][valid_mask_cpu[i]].tolist()
        if block_indices:
            grouping.append(block_indices)

    return grouping

# ==========================================================================
# 2. Hardware-Exact Quantization Core (E0M1/E1M0)
# ==========================================================================

@torch.no_grad()
def _build_padded_grouping(grouping: List[List[int]], block_size: int, device: torch.device):
    padded, masks = [], []
    for b in grouping:
        pad_len = block_size - len(b)
        padded.append(b + [b[0]] * pad_len)
        masks.append([1] * len(b) + [0] * pad_len)
    return (
        torch.tensor(padded, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.bool, device=device),
    )

@torch.no_grad()
def quantize_fp2_pairwise(flat_values: torch.Tensor, grouping: List[List[int]], enc_type: str, block_size: int = BLOCK_SIZE):
    device = flat_values.device
    idx_tensor, mask_tensor = _build_padded_grouping(grouping, block_size, device)
    grouped = flat_values[idx_tensor] 

    max_abs = torch.max(torch.abs(grouped * mask_tensor), dim=1, keepdim=True)[0].clamp(min=1e-9)
    X = torch.floor(torch.log2(max_abs))
    L = torch.pow(2.0, X)

    if enc_type.upper() == "E0M1":
        V = torch.cat([1.0 * L, 1.5 * L, -1.0 * L, -1.5 * L], dim=1)
    else: 
        V = torch.cat([0.5 * L, 1.0 * L, -0.5 * L, -1.0 * L], dim=1)

    zero = torch.zeros_like(L)
    states = [[zero, zero]]
    for i in range(4):
        v_i = V[:, i : i + 1]
        states.extend([[v_i, zero], [zero, v_i], [v_i, v_i]])
    
    candidates = torch.stack([torch.cat(s, dim=1) for s in states], dim=1)
    pairs_w = grouped.view(grouped.shape[0], block_size // 2, 2)
    
    # [VRAM FIX 3] Chunked Distance Calculation
    CHUNK_SIZE = 65536  
    best_pairs = torch.zeros_like(pairs_w)
    for i in range(0, grouped.shape[0], CHUNK_SIZE):
        pw_chunk = pairs_w[i : i + CHUNK_SIZE]
        cand_chunk = candidates[i : i + CHUNK_SIZE]
        dist_chunk = torch.sum((pw_chunk.unsqueeze(2) - cand_chunk.unsqueeze(1)) ** 2, dim=-1)
        best_idx_chunk = torch.argmin(dist_chunk, dim=-1)
        exp_cand_chunk = cand_chunk.unsqueeze(1).expand(-1, block_size // 2, -1, -1)
        gather_idx_chunk = best_idx_chunk.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        best_pairs[i : i + CHUNK_SIZE] = torch.gather(exp_cand_chunk, 2, gather_idx_chunk).squeeze(2)

    quantized_grouped_w = best_pairs.view(grouped.shape[0], block_size)
    quant_flat = flat_values.clone()

    # [VRAM FIX 4] Chunked Scatter Operation
    SCATTER_CHUNK = 262144
    for i in range(0, grouped.shape[0], SCATTER_CHUNK):
        end = i + SCATTER_CHUNK
        idx_chunk = idx_tensor[i:end]
        mask_chunk = mask_tensor[i:end]
        qw_chunk = quantized_grouped_w[i:end]
        valid_flat_idx = idx_chunk[mask_chunk]
        valid_quant_w = qw_chunk[mask_chunk]
        quant_flat.scatter_(0, valid_flat_idx, valid_quant_w.to(quant_flat.dtype))

    return quant_flat

class FP2PairwiseSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, grouping: Optional[List[List[int]]], block_size: int, enc_type: str):
        if grouping is None:
            return tensor
        shape = tensor.shape
        flat = tensor.reshape(-1)
        q_flat = quantize_fp2_pairwise(flat, grouping, enc_type=enc_type, block_size=block_size)
        return q_flat.view(shape)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

# ==========================================================================
# 3. Quantized Layers Integration
# ==========================================================================

class _QuantizedLayerMixin:
    def _init_quant_state(self, quantize: bool, weight_grouping_mode: str = 'hessian', enc_type: str = 'E0M1'):
        self.quantize = quantize          
        self.weight_grouping_mode = weight_grouping_mode
        self.enc_type = enc_type
        self.cached_weight_groups = None

    def _quantize_weight(self) -> torch.Tensor:
        if not self.quantize: return self.weight
        if self.cached_weight_groups is None:
            if self.weight_grouping_mode == 'hessian':
                self.cached_weight_groups = hessian_kmeans_grouping(self.weight, self.hessian_map, block_size=BLOCK_SIZE)
            else:
                self.cached_weight_groups = spatial_grouping(self.weight, block_size=BLOCK_SIZE)
        return FP2PairwiseSTE.apply(self.weight, self.cached_weight_groups, BLOCK_SIZE, self.enc_type)

    def _quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantize: return x
        act_groups = spatial_grouping(x, block_size=BLOCK_SIZE)
        return FP2PairwiseSTE.apply(x, act_groups, BLOCK_SIZE, self.enc_type)

class QConv2d(nn.Conv2d, _QuantizedLayerMixin):
    def __init__(self, *args, quantize: bool=True, weight_grouping_mode: str='hessian', enc_type: str='E0M1', **kwargs):
        super().__init__(*args, **kwargs)
        self._init_quant_state(quantize, weight_grouping_mode, enc_type)
        self.register_buffer("hessian_map", torch.ones_like(self.weight))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        q_weight = self._quantize_weight()
        q_input = self._quantize_activation(input)
        return self._conv_forward(q_input, q_weight, self.bias)

class QLinear(nn.Linear, _QuantizedLayerMixin):
    def __init__(self, *args, quantize: bool=True, weight_grouping_mode: str='hessian', enc_type: str='E0M1', **kwargs):
        super().__init__(*args, **kwargs)
        self._init_quant_state(quantize, weight_grouping_mode, enc_type)
        self.register_buffer("hessian_map", torch.ones_like(self.weight))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        q_weight = self._quantize_weight()
        q_input = self._quantize_activation(input)
        return F.linear(q_input, q_weight, self.bias)

def replace_with_quantized_layers(
    module: nn.Module, skip_first_last: bool, enc_type: str = "E0M1", 
    name_prefix: str = "", weight_grouping_mode: str = 'hessian'
) -> nn.Module:
    for name, child in module.named_children():
        full_name = f"{name_prefix}.{name}" if name_prefix else name
        # Exclude standard boundaries for both CNNs and LLMs
        is_boundary_layer = full_name in ("conv1", "fc", "lm_head")
        quantize_this = not (skip_first_last and is_boundary_layer)

        if isinstance(child, nn.Conv2d):
            new_layer = QConv2d(
                child.in_channels, child.out_channels, child.kernel_size,
                child.stride, child.padding, child.dilation, child.groups,
                child.bias is not None, child.padding_mode,
                quantize=quantize_this, weight_grouping_mode=weight_grouping_mode, enc_type=enc_type
            )
            new_layer.weight.data = child.weight.data.clone()
            if child.bias is not None: new_layer.bias.data = child.bias.data.clone()
            setattr(module, name, new_layer)

        elif isinstance(child, nn.Linear):
            new_layer = QLinear(
                child.in_features, child.out_features, child.bias is not None,
                quantize=quantize_this, weight_grouping_mode=weight_grouping_mode, enc_type=enc_type
            )
            new_layer.weight.data = child.weight.data.clone()
            if child.bias is not None: new_layer.bias.data = child.bias.data.clone()
            setattr(module, name, new_layer)
        else:
            replace_with_quantized_layers(child, skip_first_last, enc_type, full_name, weight_grouping_mode)
    return module

# ==========================================================================
# 4. CNN Evaluation Logic
# ==========================================================================

def get_pretrained_cifar_resnet(arch: str) -> nn.Module:
    if arch == "resnet18": model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif arch == "resnet50": model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def build_dataloaders(data_root: str, num_workers: int):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=num_workers)
    calib_loader = DataLoader(Subset(train_set, torch.randperm(len(train_set))[:200]), batch_size=200, shuffle=False)
    return train_loader, test_loader, calib_loader

def update_cnn_hessian(model: nn.Module, calib_loader: DataLoader, device: torch.device):
    model.eval()
    model.zero_grad()
    data, target = next(iter(calib_loader))
    output = model(data.to(device))
    F.cross_entropy(output, target.to(device)).backward()
    for m in model.modules():
        if isinstance(m, (QConv2d, QLinear)):
            m.hessian_map.data = (m.weight.grad ** 2).detach().clone() if m.weight.grad is not None else torch.ones_like(m.weight.data)
            m.cached_weight_groups = None  
    model.zero_grad()

def fine_tune_cnn(model: nn.Module, train_loader, test_loader, calib_loader, epochs: int, lr: float, device: torch.device, tag: str) -> List[float]:
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    
    acc_history = []
    for epoch in range(epochs):
        update_cnn_hessian(model, calib_loader, device)
        model.train()
        for data, target in tqdm(train_loader, desc=f"[{tag}] Epoch {epoch+1}/{epochs}", file=sys.stderr):
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = F.cross_entropy(model(data.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Evaluate at the end of each epoch to build the plot history
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in test_loader:
                pred = torch.max(model(data.to(device)), 1)[1]
                correct += (pred == target.to(device)).sum().item()
                total += target.size(0)
        
        acc = (correct / total) * 100.0
        acc_history.append(acc)

    return acc_history

# ==========================================================================
# 5. LLM Evaluation & Plotting Logic
# ==========================================================================

def update_llama_hessian(model: nn.Module, tokenizer, device: torch.device):
    model.eval()
    model.zero_grad()
    calib_text = "This is a calibration string used to compute the dynamic hessian map for LLaMA weights." * 10
    calib_tokens = tokenizer(calib_text, return_tensors="pt").input_ids.to(device)
    
    model(input_ids=calib_tokens, labels=calib_tokens).loss.backward()
    for m in model.modules():
        if isinstance(m, (QConv2d, QLinear)):
            m.hessian_map.data = (m.weight.grad ** 2).detach().clone() if m.weight.grad is not None else torch.ones_like(m.weight.data)
            m.cached_weight_groups = None  
    model.zero_grad()

def eval_llama(model_id: str, enc_type: str, skip_first_last: bool, grouping: str, device: torch.device):
    if not HAS_LLM_LIBS: return 0.0
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model = replace_with_quantized_layers(model, skip_first_last, enc_type, weight_grouping_mode=grouping)
    model.to(device)

    if grouping == 'hessian':
        update_llama_hessian(model, tokenizer, device)

    # Force quantization before passing to LM_Eval to save memory
    for name, module in tqdm(model.named_modules(), desc=f"Quantizing {model_id} ({enc_type}_{grouping})"):
        if isinstance(module, (QConv2d, QLinear)) and module.quantize:
            module.weight.data = module._quantize_weight().to(torch.bfloat16)
            
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, device=str(device))
    results = lm_eval.simple_evaluate(model=lm_eval_model, tasks=["winogrande", "hellaswag"], num_fewshot=0, batch_size=4)
    
    acc = sum(results['results'][t].get('acc,none', 0.0) for t in ["winogrande", "hellaswag"]) / 2.0 * 100
    
    del model, lm_eval_model
    gc.collect()
    torch.cuda.empty_cache()
    return acc

def generate_llama_plot(metrics_dict: dict, filename: str = "llama_evaluation_plot.png"):
    """Generates the categorical comparison plot for LLaMA zero-shot evaluations."""
    plt.figure(figsize=(12, 6))
    x_labels = []
    
    # Filter only LLM results (exclude Baseline)
    llm_metrics = {k: v for k, v in metrics_dict.items() if "Llama" in k and "Baseline" not in k}
    if not llm_metrics:
        return

    for k, v in llm_metrics.items():
        parts = k.split(" | ")
        model_name = parts[0]
        enc = parts[1]
        bounds = "Skip Bounds" if "True" in parts[2] else "Keep Bounds"
        
        base_label = f"{model_name}\n{enc}\n{bounds}"
        if base_label not in x_labels:
            x_labels.append(base_label)
            
    spatial_acc = [0] * len(x_labels)
    hessian_acc = [0] * len(x_labels)
    
    for k, v in llm_metrics.items():
        parts = k.split(" | ")
        base_label = f"{parts[0]}\n{parts[1]}\n{'Skip Bounds' if 'True' in parts[2] else 'Keep Bounds'}"
        idx = x_labels.index(base_label)
        
        if parts[3] == "spatial":
            spatial_acc[idx] = v
        elif parts[3] == "hessian":
            hessian_acc[idx] = v

    plt.plot(x_labels, hessian_acc, marker='s', linestyle='-', color='tab:orange', label='FP2 Hessian K-Means')
    plt.plot(x_labels, spatial_acc, marker='^', linestyle='-', color='tab:green', label='FP2 Spatial Grouping')
    
    plt.title("LLaMA Evaluation - Fully Quantized", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xlabel("Configurations", fontsize=12)
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename)
    logger.info(f"Saved LLaMA evaluation plot to {filename}")
    plt.close()

# ==========================================================================
# 6. Master Loop
# ==========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}\n" + "=" * 70)

    train_loader, test_loader, calib_loader = build_dataloaders(args.data_root, args.num_workers)
    
    final_results = {}

    # --- PHASE 1: CNNs ---
    for arch in ["resnet18", "resnet50"]:
        logger.info(f"\nEvaluating FP32 Baseline for {arch}")
        fp32_model = get_pretrained_cifar_resnet(arch).to(device)
        fp32_acc_history = fine_tune_cnn(fp32_model, train_loader, test_loader, calib_loader, args.epochs, args.lr, device, f"{arch}_FP32")
        final_results[f"{arch} | FP32 | Baseline"] = fp32_acc_history[-1]
        del fp32_model; torch.cuda.empty_cache()

        for enc in ["E1M0", "E0M1"]:
            for skip_bounds in [True, False]:
                
                # Dictionary to hold epoch histories for plotting
                histories = {}
                
                for grouping in ["spatial", "hessian"]:
                    tag = f"{arch} | {enc} | Skip Bnd: {skip_bounds} | {grouping}"
                    logger.info(f"\nEvaluating CNN: {tag}")
                    model = get_pretrained_cifar_resnet(arch)
                    model = replace_with_quantized_layers(model, skip_bounds, enc, weight_grouping_mode=grouping)
                    
                    acc_hist = fine_tune_cnn(model, train_loader, test_loader, calib_loader, args.epochs, args.lr, device, tag)
                    
                    histories[grouping] = acc_hist
                    final_results[tag] = acc_hist[-1]
                    
                    del model; torch.cuda.empty_cache()

                # Generate Plot for CNN configuration
                plt.figure(figsize=(10, 6))
                epochs_range = range(1, args.epochs + 1)
                plt.plot(epochs_range, fp32_acc_history, label="FP32 Baseline", marker='o', linestyle='--')
                plt.plot(epochs_range, histories['hessian'], label="FP2 Hessian K-Means", marker='s')
                plt.plot(epochs_range, histories['spatial'], label="FP2 Spatial Grouping", marker='^')
                
                title_tag = "First & Last Layer Unquantized" if skip_bounds else "Fully Quantized"
                plt.title(f"{arch} Evaluation ({enc}) - {title_tag}")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy (%)")
                plt.legend()
                plt.grid(True)
                plot_filename = f"{enc}_{arch}_{'first_last_unquantized' if skip_bounds else 'fully_quantized'}_comparison_plot.png"
                plt.savefig(plot_filename)
                logger.info(f"\nSaved CNN accuracy comparison plot to: {plot_filename}")
                plt.close()

    # --- PHASE 2: LLMs ---
    if HAS_LLM_LIBS:
        for model_id in ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"]:
            arch = model_id.split("/")[-1]
            for enc in ["E1M0", "E0M1"]:
                for skip_bounds in [True, False]:
                    for grouping in ["spatial", "hessian"]:
                        tag = f"{arch} | {enc} | Skip Bnd: {skip_bounds} | {grouping}"
                        logger.info(f"\nEvaluating LLM: {tag}")
                        acc = eval_llama(model_id, enc, skip_bounds, grouping, device)
                        final_results[tag] = acc

        # Generate LLaMA Plot
        generate_llama_plot(final_results)

    # --- Output ---
    logger.info("\n" + "="*80)
    logger.info(f"{'CONFIGURATION':<55} | {'ACCURACY'}")
    logger.info("="*80)
    with open("bfp2_evaluation_results.txt", "w") as f:
        f.write(f"{'Models':<20} | {'Methods':<20} | {'Wbits':<5} | {'Param':<5} | {'Accuracy':<10}\n")
        f.write("-" * 80 + "\n")
        for k, v in final_results.items():
            logger.info(f"{k:<55} | {v:.2f}%")
            parts = k.split(" | ")
            model_name = parts[0]
            if len(parts) == 3: # FP32 Baseline
                method = "FP32"
            else:
                method = f"FP2({parts[1]}_{parts[3]})"
            f.write(f"{model_name:<20} | {method:<20} | 2     | 8x    | {v:.1f}\n")

if __name__ == "__main__":
    main()