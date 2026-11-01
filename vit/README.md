# Vision Transformer (ViT-B/16)

**Contributor:** Anjney Jeswal  
**Registration Number:** 230953236  
**Course:** ICT 4442 – Deep Learning

## Overview
Implementation of an ImageNet-pretrained Vision Transformer (ViT-B/16) for detecting AI-generated face images.

## Architecture
- Input resolution: 224 × 224
- Patch size: 16 × 16
- Number of patches: 196
- Embedding dimension: 768
- Transformer encoder blocks: 12
- Attention heads: 12
- MLP dimension: 3072
- Binary classification head

## Training Configuration
- Loss: BCEWithLogitsLoss
- Optimizer: AdamW
- Batch size: 32
- Maximum epochs: 12
- Head warm-up: 2 epochs
- Fine-tuning: Final 2 encoder blocks
- Seed: 42

## Preliminary Subset Test Results

| Metric | Result |
|---|---:|
| Accuracy | 93.49% |
| Precision (Fake) | 92.81% |
| Recall (Fake) | 94.28% |
| F1-score (Fake) | 93.54% |
| ROC-AUC | 0.9836 |

Test set: 10,000 images (5,000 real and 5,000 fake).

**Note:** These are balanced-subset results, not full-dataset results.

## Evidence
The accompanying experiment files contain the model configuration, training history, evaluation metrics, confusion matrix, ROC curve, and attention visualizations.
