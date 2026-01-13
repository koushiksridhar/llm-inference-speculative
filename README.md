# Speculative Decoding for LLM Inference Optimization

## Overview

This project implements **speculative decoding**, an inference optimization technique for Large Language Models (LLMs) that can significantly accelerate text generation without compromising output quality. The implementation is based on DeepMind's research paper on speculative sampling (https://arxiv.org/pdf/2302.01318.pdf).

The project provides a complete benchmarking framework to compare the performance of standard autoregressive decoding against speculative decoding, measuring key metrics such as tokens per second (TPS) and token acceptance rates.

## Motivation

Traditional autoregressive decoding generates tokens sequentially, one at a time, which creates a bottleneck in LLM inference. Speculative decoding addresses this by:

- **Parallel Token Generation**: Uses a smaller, faster "draft" model to generate multiple candidate tokens in "parallel"
- **Efficient Verification**: Verifies these candidates in a single forward pass through the larger "target" model
- **Quality Preservation**: Uses rejection sampling to ensure the output distribution matches the target model exactly.
- **Speed Improvements**: Achieves ~2x speedup (in this implementation on Apple Silicon) in tokens per second when the draft model predictions align well with the target model

## Key Features

### 1. Speculative Decoding Engine

The core implementation (`src/decoders.py`) includes:

- **Speculative Sampling Algorithm**: Implements the full speculative decoding pipeline with:
  - Draft token generation using a smaller model
  - Parallel verification of multiple tokens via the target model
  - Rejection sampling with proper distribution correction
  - KV cache management for efficient state tracking

- **Baseline Decoding**: Standard autoregressive decoding for performance comparison

- **Advanced Features**:
  - Temperature and top-p sampling support
  - KV cache truncation on token rejection
  - Proper handling of acceptance/rejection paths
  - Token acceptance rate tracking

### 2. Benchmarking System

The benchmarking framework (`src/benchmarking.py`) provides:

- **Automated Performance Testing**: Runs both baseline and speculative decoding on configurable prompts
- **Comprehensive Metrics**: Tracks:
  - Generation time
  - Tokens per second (TPS): Throughput metric - Our results show that speculative decoding has ~2x higher throughput
  - Token acceptance rate (for speculative decoding): Percentage of draft tokens accepted - Typical acceptance rate was ~75%
  - Output quality comparison

- **Configurable Parameters**: 
  - Multiple draft lengths (k values)
  - Custom prompts
  - Adjustable generation parameters

- **Results Export**: Saves benchmark results to CSV for analysis

## Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.1.0+
- CUDA-capable GPU (optional, but recommended) or Apple Silicon (MPS support)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/koushiksridhar/llm-inference-speculative.git
cd llm-inference
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure models (optional):
Edit `configs/models.yaml` to specify your draft and target models. Our results are for Qwen-2.5 0.5B and 7B.

## Usage

### Running Benchmarks

The primary way to use this project is through the benchmarking script:

```bash
python src/benchmarking.py
```

This will:
1. Load models and tokenizer from `configs/models.yaml`
2. Run warmup inference to initialize models
3. Execute benchmarks on prompts from `configs/benchmarks.yaml`
4. Compare baseline vs. speculative decoding with different draft lengths (k)
5. Save results to `results/benchmarks.csv`

## Technical Details

### How Speculative Decoding Works

1. **Draft Phase**: The smaller draft model generates k candidate tokens sequentially
2. **Verification Phase**: The target model verifies all k tokens in parallel using a single forward pass
3. **Acceptance/Rejection**: Each draft token is accepted or rejected using rejection sampling
4. **Correction**: If a token is rejected, a corrected token is sampled from a modified distribution
5. **Cache Management**: KV caches are updated or truncated based on acceptance/rejection

### Key Implementation Details

- Uses KV cache for efficient incremental generation
- Implements proper rejection sampling to maintain distribution correctness
- Handles KV cache truncation when tokens are rejected
- Supports temperature and top-p sampling parameters

## Future Work

Several extensions could further enhance this project:

1. **MLX Implementation**: Port the speculative decoding engine to MLX (Apple's machine learning framework) to leverage optimized inference on Apple Silicon hardware, potentially achieving even better performance than the current PyTorch/MPS implementation.

2. **Scratch-Built Baseline Decoder**: Rewrite the baseline autoregressive decoder from scratch (without using HuggingFace's `generate()` method) to ensure a more 1:1 comparison with the speculative decoder implementation, eliminating potential differences in optimization paths and overhead.

3. **Batch Inference Support**: Extend the implementation to support batch processing, allowing multiple prompts to be processed in parallel. This would enable more efficient utilization of GPU resources and better throughput in production scenarios.


4. **Performance Analysis and Visualization Tools**: Develop visualization tools to analyze acceptance patterns, rejection rates by position, and performance characteristics across different draft lengths. This would help identify optimal k values and understand where speculative decoding provides the most benefit.
