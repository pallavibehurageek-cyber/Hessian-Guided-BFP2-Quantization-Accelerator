# Software implementation 

This folder contains the software-side HAWQ quantization reference and supporting utilities. It is kept separate from the BFP2 SystemVerilog accelerator because here the original implementation provides PyTorch uniform and mixed-precision quantization workflows; it is **not** a BFP2 RTL implementation and is not wired into the accelerator testbench.

## Contents

- `quant_train.py` — quantization-aware training entry point.
- `bit_config.py` — model bit-width configuration dictionaries.
- `utils/quantization_utils/` — quantized layer implementations and quantization functions.
- `utils/models/` — quantized model definitions.
- `utils/export/` — model export helpers.
- `model_zoo.md` — supported model/configuration examples.
- `requirements.txt` — Python package dependencies from the HAWQ project.
- `LICENSE` — original HAWQ MIT license.

## Environment

Use an isolated Python environment for this reference code. Install the dependencies listed in `requirements.txt`; model training also requires the appropriate dataset and may require GPU/distributed-training support. The code is from an older PyTorch ecosystem, so current package versions may require compatibility adjustments.

## Running

Run commands from this directory so the original imports such as `from bit_config import *` and `from utils import *` resolve correctly. Consult `model_zoo.md` for the original training configurations and dataset requirements.

