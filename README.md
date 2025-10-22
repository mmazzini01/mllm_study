On the Use of Eye-Tracking During Preference Annotation for the Alignment of Vision-Language Models

## Overview

This project investigates how humans read and process Vision Large Language Models (VLLMs) responses using eye-tracking technology. The study combines:

- **Eye-tracking data** from 14 participants reading VLLMs responses
- **Attention analysis** of  LLAVA VLLM between the human gaze
- **Saliency comparison** between human gaze patterns and synthetic model generation

The research aims to understand alignment between human reading patterns and model attention mechanisms, providing insights for improving VLLMs-Human alignment.

## 📊 Dataset Description

### Eye-Tracking Data
- **Participants**: 14 participants with eye-tracking recordings
- **Stimuli**: 30 image prompts with corresponding VLLM responses
- **Metrics**: Fixation duration, TRT, nFIX, FFD, GPT

### Model Attention Data
- **LLaVA-1.5-13/7B**: Vision-language model attention patterns

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for model inference)
- 16GB+ RAM (for large model processing)

## 📁 Project Structure

```
mllm_study_2/
├── attention/                    # Attention analysis modules
│   ├── main_compute_attention_*.py
│   ├── main_plot_attention_*.py
│   └── models/
├── attention_saliency/           # Attention-saliency comparison
│   ├── llava_attention_extractor_new.py
│
├── data/                         # Dataset and processed data
│   ├── raw/                      # Raw eye-tracking data
│   ├── processed/               # Processed datasets
│   └── fixation_visualizer/      # Visualization tools
├── et_analyse/                  # Eye-tracking analysis
├── et_process/                  # Eye-tracking processing
├── matteo_scripts/             # Analysis scripts
├── saliency/                   # Saliency analysis
├── results/                    # Analysis results
└── utils/                      # Utility functions
```

## 🔬 Analysis Modules

### 1. Reading Measures Analysis (`matteo_scripts/`)
- **Preference Analysis**: Chosen vs rejected response analysis
- **Alignment Signals**: Human preference correlation with model attention
- **Statistical Analysis**: Comprehensive metrics and significance testing

### 3. Saliency Analysis (`saliency/`)
- **Human Saliency Maps**: Generate saliency maps from eye-tracking data
- **Model Saliency**: Extract attention-based saliency from synthtic models
- **Comparison Metrics**: CC, KL divergence, SIM similarity
- **Leave-One-Subject-Out (LOSO)**: Cross-validation evaluation

### 2. Attention Analysis (`attention/`)
- **Model Attention Extraction**: Extract attention weights from vision-language models
- **Attention Rollout**: Layer-by-layer attention flow computation
- **Visualization**: Attention heatmap generation and overlay


