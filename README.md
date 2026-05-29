# Cross-Lingual Composed Image Retrieval via Hybrid Contrastive Learning

## Overview

This repository contains the official implementation of a cross-lingual Zero-Shot Composed Image Retrieval (ZS-CIR) system, specifically optimized for the Vietnamese language. The project localizes the standard IBM Fashion-IQ dataset and introduces an advanced feature fusion architecture to address the semantic manifold collapse commonly observed in traditional vector addition methods.

By leveraging a Gating Mechanism-based Combiner Network and a Hybrid Contrastive Loss (InfoNCE + Hard Negative Mining), the proposed system demonstrates substantial performance improvements in mapping natural language modification intents to visual feature spaces.

## Repository Structure

The codebase is modularized into distinct operational stages within the experimental pipeline:

`config.py`: The centralized configuration module. It defines global paths, hyperparameters (batch size, learning rates, temperatures), and handles SSL certificate configurations for secure enterprise environments.

`core_models`.py: The core neural network architecture. Contains the PyTorch implementation of the `CombinerNetwork`, which utilizes a Multi-Layer Perceptron (MLP) and a Sigmoid Gating Mechanism to dynamically regulate the fusion of visual and textual embeddings.

`translate.py`: Data localization script. Employs `deep_translator` with an exponential backoff and retry mechanism to automatically translate the English Fashion-IQ captions into Vietnamese while maintaining data integrity and logging failures.

`extract.py`: Offline feature extraction module. Utilizes a pre-trained Vision-Language Model (e.g., CLIP) to extract and L2-normalize visual embeddings for the entire image corpus, accelerating downstream training.

`extract_baseline.py`: Baseline evaluation script implementing a Hybrid Spherical-Late Fusion (HSLF) approach using Spherical Linear Interpolation (SLERP) to evaluate zero-shot modality gap bridging without explicit combiner training.

`train_combiner.py`: The primary training script. Optimizes the Combiner Network using AdamW and Cosine Annealing. It minimizes a Hybrid Loss function consisting of Multi-modal InfoNCE and an In-batch Hardest Negative Triplet Loss to refine micro-level fashion attributes.

`evaluate_combiner.py`: The comparative evaluation pipeline. Assesses the trained Combiner Network against standard baselines (Image-Only, Text-Only, Vector Addition) across standard retrieval metrics (Recall@1, 5, 10, 50, and Mean Rank).

`generate_charts.py`: Data visualization script. Generates publication-ready, high-resolution comparative bar charts and rank plots using `matplotlib`.

`app.py`: An interactive web application built with Streamlit, providing a real-time Graphical User Interface (GUI) for end-users to test the multimodal retrieval system.

## Prerequisites & Installation

It is highly recommended to run this project in a dedicated virtual environment (e.g., Conda or venv) with a CUDA-enabled GPU (NVIDIA RTX 4000 series or equivalent is recommended).

**1. Environment Setup**
```
conda create -n cir_env python=3.10
conda activate cir_env
```

**2. Requirements Installation**

Create a `requirements.txt` file in the root directory with the following dependencies:
```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
Pillow>=9.5.0
tqdm>=4.66.0
transformers>=4.35.0
sentence-transformers>=2.2.2
streamlit>=1.28.0
matplotlib>=3.8.0
deep-translator>=1.11.4
certifi
```

Install the dependencies via `pip`:
```
pip install -r requirements.txt
```

*Note: For optimal performance, ensure that the PyTorch installation matches your specific CUDA toolkit version.*

**3. Data & Model Preparation**

To ensure execution stability in offline and enterprise environments, the foundational dataset and pre-trained models must be explicitly downloaded and placed in the appropriate directories prior to execution.

**A. The Fashion-IQ Dataset**

- **Source**: [IBM Fashion-IQ Official Repository](https://github.com/XiaoxiaoGuo/fashion-iq)

- **Action**: Download the raw image corpus and the associated natural language JSON caption files.

- Placement Structure:

  - Extract the image files to: `data/images/`
  - Place the JSON caption files to: `data/captions/`

**B. Pre-Trained Foundation Models**
Download the following models directly from the Hugging Face Model Hub and store them in the data/models/ directory.

- **Vision Model**: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)

  - Placement: `data/models/clip-vit-base-patch32/`

- **Multilingual Text Model**: [sentence-transformers/clip-ViT-B-32-multilingual-v1](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1)

  - Placement: `data/models/clip-ViT-B-32-multilingual-v1/`

## Execution Pipeline

To reproduce the experimental results, execute the scripts in the following sequential order:

1. **Data Localization**: Translate the English Fashion-IQ dataset to Vietnamese.
```
python translate.py

```

2. **Feature Extraction**: Pre-compute visual embeddings for the image corpus.
```
python extract.py

```

3. **Evaluation Baseline**: Baseline evaluation script
```
evaluate_baseline.py

```

4. **Model Training**: Train the Combiner Network on the localized dataset.
```
python train_combiner.py

```

5. **Quantitative Evaluation**: Evaluate the trained weights against theoretical baselines.
```
python evaluate_combiner.py

```

6. **Visualization**: Render academic charts from the JSON evaluation logs.
```
python generate_charts.py

```

7. **Interactive Demo**: Launch the Streamlit web interface.
```
streamlit run app.py

```

## Performance Metrics

Based on internal evaluations of 6,016 queries, the proposed Gating Mechanism Combiner Network substantially outperforms linear vector addition baselines. Key improvements are noted in Recall@10 (7.15%) and a significant reduction in the Mean Rank (3266.1), demonstrating superior intent disentanglement and cross-lingual robustness.

## Acknowledgments

**- Fashion-IQ Dataset**: We acknowledge IBM Research for providing the foundational dataset.

Wu, H., Gao, Y., Guo, X., Al-Halah, Z., Rennie, S., Grauman, K., & Feris, R. (2021). *The Fashion IQ Dataset: Retrieving Images by Combining Side Information and Relative Natural Language Feedback*. CVPR.

**- OpenAI CLIP**: For the foundational multi-modal semantic space.
