# Hessian-Guided BFP2 Quantization and Channel-Scatter Accelerator

A hardware/software co-design for 2-bit block-floating-point (BFP2) neural-network inference, comprising a
synthesizable SystemVerilog accelerator and a PyTorch evaluation suite.

The work has two coupled contributions:

* **Software.** *Hessian K-Means weight grouping* forms block-floating-point blocks by a curvature-weighted
  clustering objective rather than by memory order, so that weights sharing a block also share a suitable block
  exponent. Under E0M1 this recovers up to 9.9 points of CIFAR-10 accuracy over spatial grouping on ResNet-50 and
  up to ~22 points on Llama-3.2-3B, at identical bit-width. The gain is consistent under E0M1 and on LLMs; under
  E1M0 it is small on ResNet-18 and **negative on ResNet-50** ([§6.2](#62-cnn--e1m0-encoding),
  [§7.3](#73-the-e1m0--resnet-50-collapse)).
* **Hardware.** That grouping destroys the correspondence between storage order and output channel, which a
  systolic PE grid structurally cannot represent — it does not stall or degrade, it *silently computes wrong sums*.
  The **channel-scatter compute engine** replaces the grid's fixed PE-to-channel binding with a comparator-based
  routing stage, reusing the FP2 encode/multiply/accumulate datapath unchanged, and sustains one weight block per
  cycle.

---

## Table of Contents

1. [The Locality Problem](#1-the-locality-problem)
2. [Numeric Format](#2-numeric-format)
3. [Hessian K-Means Weight Grouping](#3-hessian-k-means-weight-grouping)
4. [Hardware Architecture](#4-hardware-architecture)
5. [Evaluation Methodology](#5-evaluation-methodology)
6. [Accuracy Results](#6-accuracy-results)
7. [Analysis](#7-analysis)
8. [Repository Layout](#8-repository-layout)
9. [Running the Project](#9-running the project)
10. [Verification Status](#10-verification-status)
11. [Limitations](#11-limitations)
12. [References](#12-references)

---

## 1. The Locality Problem

### 1.1 Spatial grouping is channel-pure by accident

Block floating point amortizes one shared exponent across a block of mantissas, recovering much of the dynamic
range of floating point at close to the cost of fixed point. Conventionally the partition is *spatial*: walk the
weight tensor in its stored `(out_ch, in_ch, kh, kw)` order and cut it into consecutive 32-element chunks.

For a typical layer this partition happens to be **channel-pure**: `256 × 3 × 3 = 2304` is an exact multiple of 32,
so every block lies entirely within one output channel. A systolic grid depends on this. One PE, one channel, one
weight stream — the binding is a synthesis-time wire, not a convenience.

### 1.2 Hessian K-Means optimizes a different objective

Values of similar magnitude quantize more accurately when grouped together, wherever they originally sat in the
tensor. Hessian K-Means grouping flattens the whole tensor, sorts by magnitude with a curvature weighting, and
re-groups by proximity in that sorted order. It has no mechanism to respect channel boundaries, because channel
identity was never part of its objective.

### 1.3 Measured severity

The exported permutation for `layer3_1.conv2` — a 256→256, 3×3 ResNet-18 convolution, 18,432 weight blocks — was
decoded and the distinct output channels per 32-element block counted directly:

| Metric | Value |
|---|---|
| Blocks with exactly 1 distinct channel ("pure") | **0 / 18,432** |
| Mean distinct channels per block | **30.0** |
| Min / max distinct channels per block | 15 / 32 |
| Pairs (2 elements sharing one FP2 code) spanning 2 channels | **99.6%** |

Every block is mixed. The grid's core assumption is false for essentially all of the data — not an occasional edge
case to special-case around.

### 1.4 Why the grid cannot absorb this

Over the ~72-cycle reduction that sums one channel's `in_ch × kH × kW` contributions, a PE expects 72 *consecutive*
blocks that all belong to it. Feed Hessian-grouped data through it and there are exactly two options:

* **Route the whole block to one PE.** Roughly 29 of its 32 elements belong to other channels, and the grid has no
  wire that lets a PE forward those contributions elsewhere — at any wiring.
* **Broadcast to all 256 PEs and have each compare incoming channel tags against its own identity.** This is a
  comparator-based routing network — genuinely necessary — but built 256 times over, with 4,096 multiply units
  standing by each cycle to service a stream that only ever carries 16 new values.

The second option, built **once** rather than 256 times, is the architecture in [§4](#4-hardware-architecture).

---

## 2. Numeric Format

| Parameter | Value |
|---|---|
| `BLOCK_SIZE` | 32 elements |
| `PAIRS_PER_BLOCK` | 16 |
| Element width | 2 bits |
| `EXP_WIDTH` | 8 bits |
| `HW_EXP_BIAS` | 128 |
| Effective precision | 2 bits (≈2.25 b/weight including the shared exponent) |

### 2.1 Block exponent

For a block *B* with element mask *m*:

```
max_abs = max |w_i · m_i|     for i ∈ B
X       = floor(log2(max_abs))
L       = 2^X                 (the block scale)
```

### 2.2 Encodings

Two 2-bit encodings are swept throughout, each allocating four signed levels relative to `L`:

| Encoding | Level set | Character |
|---|---|---|
| **E0M1** (mantissa-only) | `{ +1.0L, +1.5L, −1.0L, −1.5L }` | Clustered near the block maximum; fine relative resolution, but no representation for a small weight other than zero. |
| **E1M0** (exponent-only) | `{ +0.5L, +1.0L, −0.5L, −1.0L }` | One octave of range inside the block; tolerates a wider magnitude spread. |

### 2.3 Pairwise codebook

Quantization operates on **pairs** of elements, mirroring the accelerator's shared-K-factor multiplier in which a
pair of weights shares one product path. A pair `(w₀, w₁)` is snapped to the nearest of 13 candidate states under
squared Euclidean distance:

```
(0, 0)
(v, 0), (0, v), (v, v)   for each v in the 4-level set
```

giving `1 + 3×4 = 13` reachable pair states. The RTL encoder targets the **same 13-state set**, so the two share a
representable range.

> **Decision-rule difference.** The software model performs a *joint* nearest-neighbour search over all 13 pair
> states; the RTL encoder derives the level from each element's leading-bit position and reconciles the two. These
> agree in the common case but are not the same function, so the hardware does not run the software
> accuracy figures exactly. The gap is not yet quantified — see [§11](#11-limitations). Earlier revisions of this
> document described the software model as bit-exact with respect to the RTL; that claim was too strong and has
> been withdrawn.

This pair-level encoding is preserved unmodified from the baseline, and is the direct cause of the routing
approximation documented in [§4.2](#42-metadata-path-weight_lane_channel_map).

### 2.4 Gradient path

Quantization is wrapped in a straight-through estimator (`FP2PairwiseSTE`) — forward quantizes, backward passes
the gradient through unchanged — making the format usable for quantization-aware fine-tuning.

> **Activations** are always grouped spatially in blocks of 32 and quantized with the same pairwise codebook. Only
> *weights* are Hessian-grouped, since only weights are static enough for an offline grouping to be computed,
> exported, and stored alongside the weight data.

---

## 3. Hessian K-Means Weight Grouping

Implemented in `hessian_kmeans_grouping()`. The routine returns a permutation of the tensor's flat indices
partitioned into blocks of 32, cached per layer and invalidated when the Hessian map is refreshed.

### 3.1 Sensitivity map

The per-element sensitivity is the squared gradient of the task loss — a diagonal Fisher / Gauss-Newton
approximation of the Hessian rather than a true second derivative:

```
h_i = (∂L/∂w_i)²
```

For CNNs this comes from a single 200-image calibration batch; for LLMs, from a fixed calibration string. The map
is refreshed **at the start of every fine-tuning epoch**, invalidating the cached grouping with it.

### 3.2 Conditioning

Raw squared gradients are long-tailed across many orders of magnitude, which would let a handful of elements
dominate the clustering cost:

```
h ← h · (std(w) / std(h))            # rescale to the weight distribution
h ← log1p(100 · h)                   # compress the tail
h ← clamp(h, Q₀.₀₅, Q₀.₉₅)           # trim extremes
h ← (h − min) / (max − min) + 0.01   # normalize to [0.01, 1.01]
```

The additive `0.01` floor guarantees no element is assigned exactly zero weight.

### 3.3 Initialization

The tensor is zero-padded to a multiple of 32, then **sorted by weight value**. Reshaping the sorted array into
rows of 32 yields an initial partition that is magnitude-contiguous by construction — already a strong solution,
and the reason the refinement below converges in a handful of sweeps.

Note that this initialization is *sensitivity-blind*: the sort uses weight value only. The Hessian map enters the
algorithm exclusively through the boundary refinement in [§3.5](#35-boundary-refinement), which is what
[§7.6](#76-how-much-does-the-sensitivity-weighting-contribute) measures.

### 3.4 Cost function

Hessian-weighted within-block variance about the Hessian-weighted centroid:

```
centroid = Σ(hᵢ·wᵢ) / Σ(hᵢ)
cost     = Σ hᵢ · (wᵢ − centroid)²
```

### 3.5 Boundary refinement

Because blocks are sorted, any beneficial reassignment is a **boundary** exchange between adjacent blocks. The
solver exploits this to stay fully vectorized:

* Adjacent blocks are paired at **two offsets** (`0`, `1`) so every boundary is considered across the sub-passes.
* Swap widths of **1 and 2** elements are attempted — the last *k* of the left block exchanged with the first *k*
  of the right.
* All candidate pairs are evaluated in one batched tensor op; a swap commits only where the combined cost strictly
  decreases (`improve_mask`).
* Weight, sensitivity, and *index* blocks update together, so the result remains a valid permutation.
* Up to `max_iters = 20` sweeps, terminating on the first sweep with no accepted swap.

Because the objective is one-dimensional, contiguous clusters in sorted order are the right structure — the
equal-size (32) constraint comes from the format, not from an approximation.

### 3.6 Memory-safety measures

Four guards bound peak VRAM on multi-billion-parameter layers:

| Guard | Threshold | Purpose |
|---|---|---|
| Recursive chunking | 16,777,216 elements | Split and group chunk-wise, rebasing indices on return. |
| Quantile subsampling | 16,000,000 elements | Estimate percentile bounds from a random subsample. |
| Chunked distance search | 65,536 blocks | Materialize the `pairs × candidates` distance tensor in slices. |
| Chunked scatter | 262,144 blocks | Write quantized values back incrementally. |

---

## 4. Hardware Architecture

The design keeps the entire FP2 encode/multiply/accumulate datapath — the actual arithmetic — **unmodified**. What
changes is only the topology deciding which physical accumulator a result feeds: a fixed wire in the grid becomes
a runtime routing decision here.

### 4.1 Pipeline

```
Weight SRAM + weight-exponent SRAM  ──┐
Weight-lane channel map (NEW)         ├──►  Channel-scatter compute engine (NEW)  ──►  Output pipeline
Activation gather (index map,             16 FP2 decode/multiply lanes                 (align, compact,
 gather FSM, reorder buffer)  ────────►    routed into 256 independent accumulators      FP2 re-encode)
```

| Stage | Status |
|---|---|
| 1. Weight/exponent SRAMs | Unchanged |
| 1. `weight_lane_channel_map` | **New** — records the owning output channel of each pair |
| 2. Activation gather (index map, gather FSM, ping-pong reorder buffer) | Unchanged |
| 3. `channel_scatter_accumulator` | **New** — the core contribution |
| 4. Output pipeline (per-pair alignment, block-exponent compaction, FP2 re-encode) | Unchanged |

### 4.2 Metadata path: `weight_lane_channel_map`

Before compute begins, the exported K-Means permutation — one line of 32 raw tensor indices per block — is decoded
**once, at load time**, never in the per-cycle hot loop:

```
channel_id = original_flat_index / (in_channels × kH × kW)
```

a constant-divisor integer division. Because it runs only during the ~18,432-cycle load phase it costs nothing
during compute. The result (one channel ID per pair, 16 per block) is stored in a flat, BRAM-inference-safe SRAM
addressed identically to the weight data SRAM, so the two stay in lockstep.

> **Documented approximation — read this before quoting end-to-end accuracy.** Since 99.6% of pairs span two
> channels and the pair-level FP2 encoding is kept deliberately unmodified, each pair is routed using **element
> 0's** channel. Element 1's product is still computed correctly, but is accumulated into element 0's channel.
> For a mixed pair that contribution lands in the wrong output channel, so the engine as configured does not
> compute a numerically correct convolution on Hessian-grouped data — it demonstrates the routing mechanism at
> pair granularity. Element-level routing (the `im2col/` variant, ~2–3× hardware cost) is what closes this. This
> is the stated cost of preserving the proven pair-level encoding; see [§11](#11-limitations).

### 4.3 Per-cycle datapath

Each cycle, one weight block (16 pairs, 32 raw elements) streams in alongside its gathered activation block:

1. **Decode** *(unchanged)* — 16 instances of `fp2_block_decode_wrapper` → `fp2_processing_element` perform the
   exact FP2 pair decode and K-factor multiply, producing sign, magnitude code, and zero flag per lane.
2. **Convert** *(unchanged)* — each lane's result becomes a signed two's-complement term.
3. **Route** *(new)* — for each of the 256 destination channels, a small comparator bank checks all 16 lane channel
   IDs this cycle. Same-cycle collisions (two pairs of one block landing on the same channel) are **summed before
   accumulation**, not lost or overwritten.
4. **Accumulate** *(unchanged)* — the per-channel sum feeds that channel's accumulator, addressed by channel
   number instead of wired to a fixed grid position.

Two structural decisions keep the routing stage from re-introducing the replication it was meant to remove:

* **Merge at the source.** Collisions are resolved once across the 16 lanes (a comparator network plus a fold)
  rather than 256 times at the accumulators. After this step at most one value targets any channel per cycle, so
  each accumulator needs a register and a plain adder rather than a 16-input adder tree.
* **Align once.** Every term is referenced to a single layer-global exponent in the 16 lanes, before routing, so
  no accumulator carries exponent state or a barrel shifter. A side effect is slightly better numerics: the
  running sum is never re-truncated mid-accumulation.

### 4.4 Throughput

Exactly one weight block is read, decoded, routed, and accumulated per cycle — the same cadence as the weight
memory read itself. Hessian K-Means produces a single globally magnitude-sorted stream, not 256 independent
per-channel streams, so widening the *lane* count does not by itself buy throughput: it would require a
proportionally wider routing network, which scales quadratically in comparators.

**One block per cycle is line rate for this datapath.** It is not an upper bound on the architecture: processing
several output *pixels* in parallel reuses the same weights, the same channel labels and the same routing
decisions, and so multiplies throughput while leaving the routing network and metadata SRAM untouched. That
direction is unexplored here and is the most promising future work ([§11](#11-limitations)).

For scale, absolute throughput is 32 MACs/cycle against the reference grid's 8,192 — per-block cadence is
preserved, not throughput at equal hardware.

### 4.5 Comparison against the systolic baseline

| | Systolic grid (baseline) | Channel-scatter engine (this work) |
|---|---|---|
| Compute topology | 16×16 grid, 256 fixed PE-to-channel bindings | 16 decode/multiply lanes + 256-way comparator-routed accumulator pool |
| Multiply units active per cycle | 256 × 16 = 4,096 | 16 |
| Weight stream assumption | 256 independent, per-channel | 1 shared, magnitude-sorted |
| Behaviour on Hessian K-Means data | **Silently incorrect** (mixed-channel sums) | Correct at pair granularity ([§4.2](#42-metadata-path-weight_lane_channel_map)) |
| Blocks per cycle | Up to 256 in parallel (spatial-mode data only) | 1 |
| Per-channel barrel shifters / exponent registers | 256 | 0 (one layer-global exponent) |
| Per-channel adder trees | Required | 0 (merged at source) |
| Measured stall cycles | — | 0 |
| FP2 encode/multiply/accumulate | Original | Identical, unmodified |
| Verified on real ResNet-18 layer data | Yes (spatial mode), 0 assertion failures | Yes (K-Means mode), 0 failures, 1.0000 cycles/block |

The comparison is not "this design is faster or smaller at equal work" — on spatially grouped data the grid wins,
which is what it is built for. The claim is narrower and stronger: on magnitude-sorted data the grid is
*inapplicable*, and this is the cheapest structure found that restores correct routing at unchanged per-block
cadence.

### 4.6 Why not keep the grid and add routing on top?

The grid's parallelism and a general routing network are mutually exclusive uses of the same silicon. Its 4,096
multiply units exist to service 256 *independent* weight streams; that parallelism has no work to do when the data
is one shared stream.

The two possible orderings both dead-end:

* **Route after the multiply.** Each PE receives an already-multiplied, already-addressed value, so its multiplier
  has nothing to do and its neighbour wiring is unused. Strip those and what remains — 256 registers with adders
  plus a routing network — is exactly this design.
* **Route before the multiply.** The PEs' multipliers become useful again, but un-multiplied operand pairs cannot
  be folded (collisions must be resolved at the destination) and their product exponents are not yet known
  (alignment must happen at the destination). That restores 256 wide adders and 256 shifters.

Nor is this damage control: the grid's headline throughput was never achievable for K-Means-organized data at any
implementation. Removing it does not sacrifice performance the design could otherwise have had.

**Optionality is preserved.** `fp2_accelerator_top.sv` — the original 16×16 grid — is retained unmodified for
spatial-mode weight data, where its 256×-parallel throughput genuinely is achievable. The two engines are
complementary, each matched to the data organization it was built for.

### 4.7 Engine variants

| Engine | Routing | Activation addressing | Status |
|---|---|---|---|
| `fp2_accelerator_top.sv` | Fixed PE-to-channel | Block | Spatial-mode baseline, unmodified |
| `hessian_bfp2_accelerator_top.sv` | Pair-level channel scatter | Block-granularity | **Recommended for primary synthesis and reported hardware metrics** |

---

## 5. Evaluation Methodology

`hessian_bfp2_evaluation.py` enumerates the full cross-product of **encoding × grouping × boundary policy** for both
CNNs and LLMs.

### 5.1 Sweep axes

| Axis | Values |
|---|---|
| Encoding | `E1M0`, `E0M1` |
| Grouping | `spatial`, `hessian` |
| Boundary policy | *Skip Bounds* (first/last in FP32) / *Keep Bounds* (fully quantized) |
| CNN models | ResNet-18, ResNet-50 |
| LLM models | Llama-3.2-1B, Llama-3.2-3B |

Boundary layers are identified by name: `conv1`, `fc`, `lm_head`.

### 5.2 CNN protocol

| Setting | Value |
|---|---|
| Dataset | CIFAR-10 |
| Initialization | ImageNet-pretrained torchvision weights, `fc` replaced with a 10-way head |
| Fine-tuning | 10 epochs, quantization-aware, STE backward |
| Optimizer | Adam, `lr = 1e-3` |
| Schedule | StepLR, `step_size = 4`, `gamma = 0.5` |
| Batch size | 256 |
| Compute | `bfloat16` autocast |
| Calibration | 200-image subset, re-drawn per epoch for the Hessian map |
| Metric | Top-1 test accuracy (%) |
| Seed | 42 |

> The backbone is ImageNet-pretrained and evaluated on 32×32 CIFAR-10 inputs without input-stem adaptation, which
> is why the FP32 reference sits near 86–91% rather than the ~95% typical of CIFAR-native ResNets. All three arms
> share the same reference, so the comparison is unaffected.

### 5.3 LLM protocol

| Setting | Value |
|---|---|
| Harness | `lm-eval-harness` via `HFLM` |
| Tasks | WinoGrande, HellaSwag |
| Shots | 0 |
| Batch size | 4 |
| Weights | `bfloat16`, quantized in place before evaluation to bound peak memory |
| Metric | Mean of the two task accuracies (%) |

LLMs are evaluated post-training-quantization only — no fine-tuning — making them the more discriminating test of
grouping quality.

---

## 6. Accuracy Results

Values are read at the final epoch (epoch 10). **Δ** is the Hessian K-Means margin over spatial grouping; **Gap**
is the shortfall against FP32.

### 6.1 CNN — E0M1 encoding

| Model | Boundary policy | FP32 | Spatial | Hessian K-Means | Δ | Gap |
|---|---|---:|---:|---:|---:|---:|
| ResNet-18 | Fully quantized | 86.1 | 68.1 | **72.5** | **+4.4** | −13.6 |
| ResNet-18 | First/last unquantized | 86.0 | 75.1 | **78.5** | **+3.4** | −7.5 |
| ResNet-50 | Fully quantized | 90.6 | 58.2 | **68.1** | **+9.9** | −22.5 |
| ResNet-50 | First/last unquantized | 90.6 | 70.7 | **75.3** | **+4.6** | −15.3 |

### 6.2 CNN — E1M0 encoding

| Model | Boundary policy | FP32 | Spatial | Hessian K-Means | Δ | Gap |
|---|---|---:|---:|---:|---:|---:|
| ResNet-18 | Fully quantized | 85.8 | 73.0 | **73.9** | +0.9 | −11.9 |
| ResNet-18 | First/last unquantized | 85.8 | 78.3 | **79.9** | +1.6 | −5.9 |
| ResNet-50 | Fully quantized | 90.6 | **58.8** | 48.3 | −10.5 | −31.8 |
| ResNet-50 | First/last unquantized | 90.6 | **66.9** | 64.6 | −2.3 | −23.7 |

### 6.3 LLM — post-training quantization, no fine-tuning

Mean of WinoGrande and HellaSwag 0-shot accuracy.

| Model | Encoding | Boundary policy | Spatial | Hessian K-Means | Δ |
|---|---|---|---:|---:|---:|
| Llama-3.2-1B | E1M0 | Skip Bounds | 37.0 | **40.9** | +3.9 |
| Llama-3.2-1B | E1M0 | Keep Bounds | 38.0 | **39.0** | +1.0 |
| Llama-3.2-1B | E0M1 | Skip Bounds | 36.9 | **50.0** | **+13.1** |
| Llama-3.2-1B | E0M1 | Keep Bounds | 37.4 | **49.3** | **+11.9** |
| Llama-3.2-3B | E1M0 | Skip Bounds | 38.8 | **42.9** | +4.1 |
| Llama-3.2-3B | E1M0 | Keep Bounds | 37.7 | **42.7** | +5.0 |
| Llama-3.2-3B | E0M1 | Skip Bounds | ≈38.5 | **60.8** | **≈+22.3** |
| Llama-3.2-3B | E0M1 | Keep Bounds | 39.4 | **60.6** | **+21.2** |

> The spatial value for Llama-3.2-3B / E0M1 / Skip Bounds is occluded by the plot legend and is interpolated;
> substitute the exact figure from `bfp2_evaluation_results.txt`.

### 6.4 Best configuration per model

| Model | Best configuration | Accuracy |
|---|---|---:|
| ResNet-18 | E1M0 · Hessian · first/last unquantized | **79.9** |
| ResNet-50 | E0M1 · Hessian · first/last unquantized | **75.3** |
| Llama-3.2-1B | E0M1 · Hessian · Skip Bounds | **50.0** |
| Llama-3.2-3B | E0M1 · Hessian · Skip Bounds | **60.8** |

### 6.5 Accuracy curves — E0M1

**ResNet-18, fully quantized.** Hessian grouping leads from the first epoch (57.2 vs 44.2, a 13.0-point margin)
and holds a 4.4-point lead through convergence.

<img width="1000" height="600" alt="resnet18_fully_quantized_comparison_plot" src="https://github.com/user-attachments/assets/028963fa-8b81-4943-b553-7d66b638f8fa" />


**ResNet-18, first and last layers unquantized.** Exempting `conv1` and `fc` recovers roughly 6 points for both
schemes and halves the gap to FP32.

<img width="1000" height="600" alt="resnet18_first_last_unquantized_comparison_plot" src="https://github.com/user-attachments/assets/3d606aa3-964b-4984-acb3-4f338a5801c7" />


**ResNet-50, fully quantized.** The largest margin observed for the method: 68.1 vs 58.2, visible at every epoch.

<img width="1000" height="600" alt="resnet50_fully_quantized_comparison_plot" src="https://github.com/user-attachments/assets/8913445b-560c-43fd-a90d-d7b58756b750" />


**ResNet-50, first and last layers unquantized.**
<img width="1000" height="600" alt="resnet50_first_last_unquantized_comparison_plot" src="https://github.com/user-attachments/assets/67585bbc-b2f4-4299-80ca-137d4f70b37c" />


### 6.6 Accuracy curves — E1M0

**ResNet-18, fully quantized.** The schemes converge to within a point; the Hessian advantage under E1M0 is
confined to the early epochs.

<img width="1000" height="600" alt="E1M0_resnet18_fully_quantized_comparison_plot" src="https://github.com/user-attachments/assets/ee550a8e-2469-441c-9770-dd9573157910" />


**ResNet-18, first and last layers unquantized.** The strongest ResNet-18 result overall, at 79.9.

<img width="1000" height="600" alt="E1M0_resnet18_first_last_unquantized_comparison_plot" src="https://github.com/user-attachments/assets/10b9cc59-0793-4707-8ed2-1c07e6a7a84d" />


**ResNet-50, fully quantized.** The one configuration where Hessian grouping loses: the curve peaks at 56.4
(epoch 8) then collapses to 48.3, ending below spatial. See [§7.3](#73-the-e1m0--resnet-50-collapse).

<img width="1000" height="600" alt="E1M0_resnet50_fully_quantized_comparison_plot" src="https://github.com/user-attachments/assets/58b22ebc-ff3b-4f19-a768-6981660da781" />


**ResNet-50, first and last layers unquantized.** The curves track closely, with spatial ending marginally ahead.

<img width="1000" height="600" alt="E1M0_resnet50_first_last_unquantized_comparison_plot" src="https://github.com/user-attachments/assets/af1e3936-f102-474e-a0fb-9fae08a30232" />


### 6.7 LLM evaluation

All eight Llama configurations. The two E0M1 Hessian points on Llama-3.2-3B are the outlying results of the study,
near 60% against a spatial-grouping floor around 38%.

<img width="1000" height="600" alt="llama_evaluation_plot" src="https://github.com/user-attachments/assets/d297f8c0-6789-4957-a9b6-ba2a49ed21fd" />


---

## 7. Analysis

### 7.1 The grouping matters more than the encoding

Under E0M1, Hessian K-Means recovers 3.4 to 9.9 points over spatial grouping at identical bit-width, area, and
cycle cost. The only change is *which* weights share an exponent. On LLMs the effect reaches ~22 points, because
there is no fine-tuning to compensate for a poor partition — grouping quality is measured directly rather than
through the optimizer's recovery capacity.

### 7.2 E0M1 and Hessian grouping are complementary

The interaction is consistent across all four LLM pairings and both ResNet-50 rows: the Hessian margin under E0M1
is several times its margin under E1M0.

This follows from the level sets. E0M1's `{±1.0L, ±1.5L}` cluster near the block maximum and offer **no
representation for a small weight** other than the pairwise zero state — so E0M1 is only viable when a block's
members are of comparable magnitude, which is exactly what Hessian K-Means enforces. E1M0's `{±0.5L, ±1.0L}` span
an octave and tolerate wider spread, making it more robust under spatial grouping but leaving it less to gain from
a better partition. The two mechanisms are complementary rather than additive, and E0M1 should not be evaluated
with spatial grouping.

### 7.3 The E1M0 / ResNet-50 collapse

E1M0 with Hessian grouping on a fully quantized ResNet-50 is the sole configuration where the method loses. The
curve rises normally to 56.4 at epoch 8, then falls to 48.3 — a late-training collapse, not a failure to learn,
and the shape implicates the grouping–fine-tuning interaction rather than the format.

Because the Hessian map and cached grouping are recomputed every epoch, the partition is a moving target. Once the
learning-rate schedule has decayed (`gamma = 0.5` at epochs 4 and 8) the optimizer can no longer re-adapt to a
partition that has shifted underneath it. ResNet-50's 1×1 bottleneck convolutions, with few weights per output
channel, are the most exposed to a re-partition. Candidate mitigations, in order of expected effect:

1. **Freeze the grouping** after a warm-up period (e.g. epoch 4) and fine-tune against a fixed partition.
2. **Reduce refresh cadence** to every *N* epochs.
3. **Accumulate the Hessian** over several calibration batches instead of one, reducing the sampling noise that
   drives partition churn.

Read each table entry alongside the per-epoch peak; epoch-10 values are reported throughout for consistency.

### 7.4 Boundary layers remain expensive

Leaving `conv1` and `fc` in FP32 is worth 5–12 points in every CNN configuration, at a small parameter cost. The
*Skip Bounds* configuration is the realistic deployment target; *Keep Bounds* is retained as the strict upper
bound on quantization coverage.

### 7.5 Depth interacts with quantization pressure

ResNet-50 has a higher FP32 ceiling than ResNet-18 (90.6 vs 86.1) but a worse quantized floor in most
configurations. Its bottleneck blocks concentrate more parameters in 1×1 convolutions whose per-channel weight
count is small, so a poorly matched block exponent affects proportionally more of the network. This is why
ResNet-50 shows both the largest gain (+9.9, E0M1) and the largest regression (−10.5, E1M0) — it is the more
sensitive of the two networks.

### 7.6 How much does the sensitivity weighting contribute?

The initialization is a pure magnitude sort ([§3.3](#33-initialization)); the Hessian map enters only through
boundary refinement. A structural ablation isolates the two by grouping each layer twice — once with the real
`grad²` map, once with uniform weights — and counting how many weights land in a different block:

| Layer | Weights | Relocated |
|---|---:|---:|
| `layer1.0.conv1` | 36,864 | 2.22% |
| `layer1.1.conv2` | 36,864 | 2.74% |
| `layer2.0.conv1` | 73,728 | 2.74% |
| `layer2.1.conv2` | 147,456 | 2.87% |
| `layer3.0.conv1` | 294,912 | 2.91% |
| `layer3.1.conv2` | 589,824 | 2.64% |
| `layer4.0.conv1` | 1,179,648 | 2.05% |
| `layer4.1.conv2` | 2,359,296 | 1.16% |
| **Mean** | | **2.42%** |

With a uniform sensitivity map the refinement accepts **zero** swaps — for a one-dimensional equal-size objective
the sorted partition is already optimal, so only the sensitivity weighting can make an exchange profitable. Every
relocated weight is therefore attributable to the Hessian term.

Two readings follow. The magnitude sort determines ~97.6% of the grouping structure; the sensitivity weighting is
a targeted refinement on top. And the effect declines with layer size — `layer4.1.conv2` at 1.16% is half the peak
— consistent with larger layers having more blocks over the same value range, so boundary exchanges change
within-block variance less.

> This is a **structural** measure. It does not establish that the relocated weights drive the accuracy gain; 2.4%
> of weights could carry disproportionate weight precisely because they are the high-sensitivity ones. The
> accuracy ablation — a full run with the sensitivity map forced to uniform — is in progress and is what settles
> whether the Hessian term belongs in the headline.

---

## 8. Repository Layout

```text
.
├── rtl/
│   ├── activation_reorder_buffer.sv
│   ├── channel_scatter_accumulator.sv
│   ├── fp2_block_decode_wrapper.sv
│   ├── hessian_bfp2_accelerator_top.sv
│   ├── fp2_processing_element.sv
│   ├── fp2_sram.sv
│   ├── memory_address_router.sv
│   ├── output_block_compactor.sv
│   ├── output_compression_pairwise.sv
│   ├── post_processing_unit.sv
│   ├── temporal_accumulator.sv
│   ├── weight_gather_ctrl.sv
│   ├── weight_index_map_sram.sv
│   └── weight_lane_channel_map.sv
├── verification/
│   └── tb_hessian_bfp2_resnet18.sv
├── sw/
│   └── hessian_bfp2_evaluation.py
├── .gitignore
└── README.md
```

> This layout lists the files transferred from the author's public repository. The available source snapshot does not contain the spatial baseline RTL, optional im2col RTL, architecture document, result plots, simulation input vectors, or a Makefile; those files are not included here.

---

## 9. Running the Project

### 9.1 Software evaluation

```bash
pip install torch torchvision numpy matplotlib tqdm
# Optional, for the LLM phase:
pip install transformers lm-eval

python sw/hessian_bfp2_evaluation.py \
    --epochs 10 \
    --lr 1e-3 \
    --data-root ./data
```

CIFAR-10 downloads on first run. The LLM phase is skipped gracefully if `transformers` and `lm-eval` are absent
(`HAS_LLM_LIBS`), so the CNN sweep runs without Hugging Face access. Results are written to
`bfp2_evaluation_results.txt` as `Models | Methods | Wbits | Param | Accuracy`, and to stdout as the run proceeds.

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 10 | Fine-tuning epochs per CNN configuration |
| `--lr` | 1e-3 | Adam learning rate |
| `--data-root` | `./data` | CIFAR-10 cache directory |
| `--num-workers` | `min(8, cpu_count)` | DataLoader workers |

The sweep runs 2 baselines plus 16 CNN and 16 LLM configurations; gate individual axes in the `main()` loops for
shorter runs.

### 9.2 RTL simulation

The testbench is `verification/tb_hessian_bfp2_resnet18.sv`. It expects the external input-vector files used by the original project. Those vector files and a Makefile are not present in this repository snapshot, so the testbench cannot be run from this checkout alone. Provide the matching vectors and compile the RTL source files with your simulator before treating simulation results as locally verified.

### 9.3 FPGA synthesis

Target: **Xilinx Artix-7 `xc7a200tfbg676-2`**, 10 ns target period, synthesized out-of-context so the wide
load-time interface does not make the run I/O-bound.

```tcl
synth_design -top hessian_bfp2_accelerator_top -part xc7a200tfbg676-2 -mode out_of_context
```

Two structural properties matter for running the reported numbers:

* The 256 accumulators must infer **DSP48E1** slices. This requires a *synchronous* reset on the accumulator
  register — an asynchronous reset blocks inference entirely and the design falls back to 256 LUT carry chains,
  which alone pushes LUT utilization past the device.
* Wide combinational reductions must stay pipelined. Two were found by reading post-synthesis critical-path
  reports rather than by inspection: a flat 256-input OR on the overflow-sticky flag, and a **serial** 16-deep
  max-exponent comparator chain in the output compactor (a `for`-loop max, which infers a dependency chain rather
  than a tree). Both are now balanced trees with a register stage.

Timing closure at 10 ns is **in progress**; utilization is comfortable on the device, and worst negative slack has
been reduced by an order of magnitude from the first synthesis attempt.

### 9.4 ASIC synthesis

* **Synthesis / P&R:** Yosys with OpenROAD-flow-scripts, Nangate45 / FreePDK45.
* **Memory macros:** OpenRAM against the FreePDK45 setup; where OpenRAM generation is unavailable, hand-built
  LEF/LIB views derived from CACTI estimates may be substituted.
* Macros are integrated via `ADDITIONAL_LEFS`, `ADDITIONAL_LIBS`, and `ADDITIONAL_GDS`.

Note that DSP inference is Xilinx-specific: in the ASIC flow the 256 accumulators map to standard-cell adders, so
FPGA and ASIC area are not comparable on that axis.

`hessian_bfp2_accelerator_top.sv` is the recommended target for reported hardware metrics.

---

## 10. Verification Status

All results below are measured on real weight and permutation data, not asserted from construction.

| Item | Result |
|---|---|
| Full 18,432-block sweep, `hessian_bfp2_accelerator_top` | **0 assertion failures** |
| Output channel coverage | All **256** channels received differentiated, non-degenerate values |
| Output pair coverage | All **128** pairs fired and settled |
| Throughput | **1.0000 cycles/block**, from instrumented `$time` delta ÷ block count |
| Stall cycles | 0 |
| `fp2_accelerator_top` (spatial mode) against real hex vectors | 0 assertion failures |
| Post-restructure equivalence | Output words **bit-identical** to the pre-restructure design across multiple configurations, confirming the timing work is behaviour-preserving |

### Defects found and fixed before integration

Each of these was found by differential comparison against a reference model or by reading tool reports — none by
inspection, which is the point worth recording.

* A **load-time latency mismatch** between the new channel-map memory and the weight SRAM it must stay
  synchronized with — caught by a dedicated hand-checkable testbench before it could hide inside a larger
  integration.
* A **2D array-of-arrays memory declaration** in `weight_lane_channel_map` — a real BRAM-inference risk at this
  module's ~288 KB size, the largest single memory in the engine. Flattened to the packed-word pattern used by
  every other memory in the design, with re-verification confirming zero behavioural change.
* An **output collector/compactor race**: the compactor was fired one cycle after the collector filled, and
  because the drain streams continuously that extra cycle is exactly when the next group overwrites slot 0. One
  channel pair in every output word carried the wrong data. It self-masked on the final group and on any group
  whose neighbour encoded identically, so only a minority of words looked wrong.
* A **racy simulation assertion** that reported failures on values never presented to the accumulators — noise
  that would have masked a real failure. Re-homed as a clocked check on settled values.
* A **testbench defect** in which every output position after the first read a weight bank that was never loaded,
  making any throughput or switching-activity measurement taken from it meaningless.

### im2col variant

| Item | Result |
|---|---|
| `fp2_element_product.sv` | Exhaustively verified over all 128 input combinations |
| `im2col_addr_gen.sv` | Verified against hand-computed convolution cases, including zero-padding |
| `weight_lane_element_map.sv` | Verified; a bit-packing-order defect found and fixed |
| `channel_router_im2col.sv` | Verified on a hand-computed two-exponent collision case |
| `activation_tile_snapshot.sv` | Verified standalone; an off-by-one found and fixed |
| 4-pass sweep, `hessian_bfp2_im2col_top` | Verified at reduced scale (100 blocks), position-differentiated results |

---

## 11. Limitations

Stated directly rather than left for a reader to discover.

1. **Pair-level, not element-level, routing.** Element 1's product in a mixed pair is accumulated into element 0's
   channel ([§4.2](#42-metadata-path-weight_lane_channel_map)), and 99.6% of pairs are mixed. The recommended
   engine therefore demonstrates the routing mechanism at pair granularity rather than computing a numerically
   correct convolution on Hessian-grouped data; hardware metrics should be scoped to the routing mechanism
   accordingly. The `im2col/` variant implements element-level routing at ~2–3× hardware cost.
2. **The paper's pairwise MAC collapse does not apply here.** The reference design's `Psum = (F₁ ± F₂)·W`
   optimization assumes a pair's two weights contribute to the *same* partial sum. Under global magnitude sorting
   they share an output channel with probability ≈ 1/C_out, so that identity is unavailable. This is a structural
   consequence of the grouping, not an implementation gap: magnitude-sorted grouping and the pairwise MAC collapse
   cannot coexist. The weight is still decoded once per pair; only the activation-side work doubles.
3. **Encoder decision rule differs from the software model.** [§2.3](#23-pairwise-codebook). Same 13-state
   representable set, different selection rule; the resulting accuracy difference is not yet measured.
4. **Block-granularity activation gather.** True im2col addressing — per-element scalar fetches with correct
   zero-padding at image borders — is not implemented in the recommended configuration.
5. **Per-layer exponent spread is bounded.** The alignment stage supports a shift range of `MAX_SHIFT`; a layer
   whose combined activation-plus-weight exponent spread exceeds it will have small terms silently dropped below
   the accumulator LSB, with no health flag raised. Layer statistics should be checked against this bound before
   trusting a run.
6. **Accuracy ablation for the sensitivity term is in progress.** [§7.6](#76-how-much-does-the-sensitivity-weighting-contribute).
7. **K-blocks-per-cycle scaling.** Processing *K* blocks per cycle was explored and deferred; the multi-exponent
   alignment required across simultaneously issued blocks is unresolved. Pixel-parallel scaling
   ([§4.4](#44-throughput)) is the more promising direction and is unexplored.

Limitations (1), (3) and (4) affect end-to-end numerical agreement with a full convolution reference; none affects
the correctness or efficiency of the channel-routing mechanism itself, which is verified in
[§10](#10-verification-status).

---

## 12. References

1. Dang et al., *IEEE Transactions on Circuits and Systems I*, 2026 — the BFP2 format and reference systolic
   accelerator on which the spatial-mode datapath is based.
2. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016 — ResNet-18 / ResNet-50.
3. Meta AI, *Llama 3.2* model family.
4. Gao et al., *A Framework for Few-Shot Language Model Evaluation* (`lm-evaluation-harness`).

---

---

## Source Attribution

The RTL and Python evaluation files were transferred and renamed from [RohanCoderiiitb/Hessian-Aware-BFP2-Accelerator](https://github.com/RohanCoderiiitb/Hessian-Aware-BFP2-Accelerator) with the author's permission, as communicated by the repository owner. The top-level RTL module, testbench module, and evaluation output filename were updated to match this repository's naming.
