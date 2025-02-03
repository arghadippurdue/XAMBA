# XAMBA: Enabling Efficient State Space Models on Resource-Constrained Neural Processing Units

This repository contains scripts for converting pretrained Hugging Face models to OpenVINO Intermediate Representation (IR), applying various XAMBA techniques on the SSM (Mamba, Mamba-2, etc.) models, and benchmarking the execution latency using OpenVINO.

## Files

- **convert.py**: Converts pretrained Hugging Face models to OpenVINO IR format.
- **xamba.py**: Implements CumBA, ReduBA & ActiBA techniques on the Mamba-2 model.
- **benchmark.py**: Evaluates execution latency using OpenVINO's `benchmark_app`.

## Prerequisites

To use this repository, you need to install the following dependencies:

- Python 3.6 or higher
- PyTorch
- OpenVINO
- Hugging Face Transformers library

## Installation

```bash
pip install torch openvino transformers
```

## Usage

### 1. Converting Hugging Face Models to OpenVINO IR

Run the `convert.py` script to convert a pretrained Hugging Face model into OpenVINO Intermediate Representation (IR). Make sure to locally update the Mamba-2 model inside the `transformers` library with the `xamba.py`.

Example:

```bash
python convert.py
```

### 2. Benchmarking Execution Latency

To benchmark the execution latency of a model with OpenVINO, run the `benchmark.py` script:

```bash
python benchmark.py
```

This will evaluate the inference latency on the specified device.

## Directory Structure

```plaintext
├── convery.py              # Convert HF models to OpenVINO IR
├── xamba.py                # XAMBA techniques on Mamba-2 model
├── benchmark.py            # Evaluate execution latency using OpenVINO benchmark_app
└── README.md               # This README file
```
