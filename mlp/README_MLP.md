# MLP Baseline for AI-Generated Face Detection

**Contributor:** Aditya Mohan Jha (230911428)  
**Course:** ICT 4442 Deep Learning Mini Project  
**Task:** Distinguish real FFHQ face images from StyleGAN-generated face images using a non-convolutional multilayer perceptron (MLP).

This is the MLP baseline in our team's four-architecture comparison (MLP, transfer-learned ResNet-18, patch BiLSTM, and ViT-B/16). It uses global colour and frequency summaries instead of spatial feature maps. Its purpose is to establish how much discrimination is possible without a CNN or attention-based image encoder.

## Dataset and labels

The primary dataset is [140k Real and Fake Faces by xhlulu](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces). It contains 70,000 real FFHQ faces and 70,000 StyleGAN-generated faces in supplied `train`, `valid`, and `test` directories. The notebook preserves these partitions.

| Folder name | Label |
| --- | ---: |
| `real` | 0 |
| `fake` | 1 |

The completed full run used **100,000 training images** and **20,000 test images**. Fake is the positive class for precision, recall, and F1. The primary test set was evaluated after checkpoint selection on validation data. The independent CIPLAB Photoshop dataset discussed in the project report was **not evaluated** in this MLP run.

## Method

Each face is EXIF-corrected, converted to RGB, and resized to **128 × 128** pixels. Feature extraction produces a **145-dimensional** vector:

| Feature group | Dimensions | Description |
| --- | ---: | --- |
| RGB statistics and quantiles | 27 | Per-channel mean, standard deviation, minimum, maximum, and five quantiles |
| RGB histograms | 48 | 16 bins for each of the three colour channels |
| Grayscale histogram | 32 | Global luminance distribution |
| Radial Fourier profile | 32 | Frequency bands from the centered log-magnitude 2D FFT |
| Spectral summaries | 6 | Low/mid/high power fractions, high-to-low ratio, centroid, and entropy |
| **Total** | **145** | One fixed-length vector per image |

A `StandardScaler` is fitted **only on training features** and then applied unchanged to validation and test features. This prevents preprocessing leakage.

The network is `145 → 256 → 128 → 32 → 1 logit`. The first two dense blocks use batch normalization, ReLU, and dropout (0.35 and 0.30). The third uses ReLU and dropout 0.20. The single logit is converted to a fake probability with a sigmoid at evaluation; a fixed **0.5** threshold produces class predictions.

| Training setting | Value |
| --- | --- |
| Loss | `BCEWithLogitsLoss` |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 512 |
| Maximum epochs | 30 |
| Early-stopping patience | 5 non-improving epochs |
| Checkpoint criterion | Highest validation fake-class F1 |
| Random seed | 42 |

## Reproduce on Kaggle

1. Create a Kaggle notebook and attach the [primary dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) using **Add Input**.
2. Open `aditya_mlp_baseline.ipynb`. Kaggle's standard Python environment should supply NumPy, Pillow, PyTorch, scikit-learn, Matplotlib, and tqdm.
3. For a fast pipeline check, set the first code cell to:

   ```python
   RUN_NAME = "preliminary"
   LIMIT_PER_CLASS = 5000
   EVALUATE_TEST = False
   ```

4. For the reported full experiment, set:

   ```python
   RUN_NAME = "full"
   LIMIT_PER_CLASS = None
   EVALUATE_TEST = True
   ```

5. Run all cells **from the beginning** with the dataset attached. The complete output is written to `/kaggle/working/mlp_full/`. Feature extraction is primarily CPU work and may take longer than training the small MLP. Cached features can speed reruns in the same working environment, but a fresh Kaggle session may require extraction again.
6. Check that `config.json` records `run_name: full` and `train_count: 100000`, and that `test_metrics.json` records `test_count: 20000`. Keep the notebook, configuration, and result files together; do not mix them with preliminary-run artifacts.

The preliminary experiment used 5,000 real and 5,000 fake images from each of training and validation. Its validation results are **not** the final held-out test results.

## Verified full-run test results

The best full-run checkpoint was selected at **epoch 29**. On the **20,000-image primary test set**, the saved metrics are:

| Metric | Value |
| --- | ---: |
| Accuracy | **0.7757** (77.57%) |
| Precision, fake | **0.7621** (76.21%) |
| Recall, fake | **0.8016** (80.16%) |
| F1, fake | **0.7814** (78.14%) |
| ROC-AUC | **0.8572** |
| Binary cross-entropy test loss | **0.4699** |
| Decision threshold | **0.5** |

Confusion matrix (rows are true labels; columns are predicted labels):

|  | Predicted real | Predicted fake |
| --- | ---: | ---: |
| **True real** | 7,498 | 2,502 |
| **True fake** | 1,984 | 8,016 |

Thus, the model correctly classified 15,514 of 20,000 test images. It missed 1,984 generated faces and falsely flagged 2,502 real faces. These counts and metrics describe the **primary FFHQ-versus-StyleGAN test partition**, not out-of-distribution performance.

## Repository files

Suggested layout inside the team's shared repository:

```text
mlp/
├── README.md
├── aditya_mlp_baseline.ipynb
├── config.json
├── standard_scaler.npz
├── best_mlp.pt
└── results/
    ├── test_metrics.json
    ├── validation_metrics.json
    ├── training_history.csv
    ├── test_predictions.csv
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── training_curve.png
```

The model checkpoint and scaler are useful for inference **only with the same feature-extraction code and feature order**. Do not commit the raw face dataset or the large `train_features.npz`, `valid_features.npz`, and `test_features.npz` caches. If repository size is a concern, store the checkpoint separately and document its location; retain the notebook, configuration, metrics, and plots in Git.

## Interpretation and limitations

The MLP shows that colour distributions and frequency statistics contain a useful signal for this dataset. Because these are global summaries, the network cannot directly localize a seam, eye artefact, or inconsistent facial region. The result may partly reflect differences between the FFHQ and StyleGAN image-generation pipelines; it should not be presented as universal deepfake detection. No cross-dataset result, repeated-seed uncertainty estimate, or feature-group ablation is claimed here.

The preliminary training curve showed training loss continuing to decrease while validation loss flattened, suggesting a generalization gap on that subset. That curve should not be labelled as the full-run training curve. For fair comparison with the other team models, evaluate on the same test images, use fake as the positive class, and report the same metrics.

## Attribution and academic integrity

Dataset: xhlulu, *140k Real and Fake Faces*, Kaggle. The MLP's frequency-feature motivation follows J. Frank et al., *Leveraging Frequency Analysis for Deep Fake Image Recognition* (ICML 2020), and Y. Qian et al., *Thinking in Frequency: Face Forgery Detection by Mining Frequency-Aware Clues* (ECCV 2020). The team report contains the fuller literature review.

AI assistance was used to help draft and refine the notebook and explanatory text. The contributor should review the code, verify each reported result against the generated files, and disclose this assistance as required by the course guidelines. No unmeasured result is asserted here.
