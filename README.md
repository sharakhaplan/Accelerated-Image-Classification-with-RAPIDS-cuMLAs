# GPU-Accelerated Image Classification with RAPIDS cuML Random Forest

A comparative machine learning experiment using the MNIST handwritten digit dataset to evaluate CPU-based and GPU-based Random Forest implementations in terms of training time and classification accuracy.

## Overview

This project was completed as part of **Module 14 of the DLI Accelerated Data Science Teaching Kit**.

The objective is to compare:

- `scikit-learn` Random Forest running on the CPU
- `RAPIDS cuML` Random Forest running on the GPU

Both implementations are evaluated on the MNIST dataset using the same train/test split and comparable model settings.

## Objectives

- Prepare the MNIST dataset for image classification.
- Train a CPU-based Random Forest classifier.
- Train a GPU-accelerated Random Forest classifier using RAPIDS cuML.
- Compare model-fitting time and test accuracy.
- Examine class-level classification metrics and confusion matrices.

## Dataset

The experiment uses the `mnist_784` dataset retrieved through OpenML.

- **Total samples:** 70,000
- **Image dimensions:** 28 × 28 pixels
- **Features per image:** 784
- **Classes:** Digits 0–9
- **Training samples:** Approximately 60,000
- **Test samples:** Approximately 10,000

Each image is represented as a flattened vector of 784 pixel values. Pixel intensities are scaled from the original 0–255 range to 0–1.

## Methodology

1. Load MNIST using `fetch_openml`.
2. Convert image features to `float32` and labels to `int32`.
3. Normalize pixel values.
4. Split the dataset using `train_test_split`, with `test_size=1/7`, `random_state=42`, and stratification by class.
5. Train a scikit-learn Random Forest classifier on the CPU.
6. Transfer the training and test arrays to GPU memory using CuPy.
7. Train a RAPIDS cuML Random Forest classifier on the GPU.
8. Measure model-fitting time and evaluate both models on the test set.
9. Compare accuracy, classification reports, and confusion matrices.

## Model Configuration

| Parameter | scikit-learn (CPU) | RAPIDS cuML (GPU) |
|---|---|---|
| Estimator | RandomForestClassifier | RandomForestClassifier |
| Number of trees | 100 | 100 |
| Maximum depth | 20 | 20 |
| Random seed | 42 | 42 |
| Additional settings | `n_jobs=-1` | `n_bins=128`, `n_streams=1` |

The GPU model uses `output_type="numpy"` for NumPy-compatible prediction output.

## Experimental Results

| Metric | CPU | GPU |
|---|---:|---:|
| Training time | 29.8554 s | 4.7358 s |
| Test accuracy | 96.59% | 96.59% |

**Training speedup:** approximately 6.31×  
**Training-time reduction:** approximately 84.1%

The GPU implementation achieved the same reported test accuracy while requiring less model-fitting time in this experiment.

*Timing scope:* GPU training data was transferred to GPU memory before timing began. GPU synchronization was used around model fitting to ensure the measured training operation had completed. Therefore, these timings represent model-fitting time rather than total pipeline execution time.

## Evaluation

The experiment also generates:

- A training-time comparison chart
- A test-accuracy comparison chart
- Classification reports for both implementations
- Confusion matrices for class-level error analysis

Accuracy provides an overall measure of correct predictions, while precision, recall, F1-score, and confusion matrices help identify differences in class-level performance.

## Technologies

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn
- RAPIDS cuML
- CuPy
- OpenML
- Google Colab with GPU runtime

## Environment and Reproducibility

Run the notebook in an environment with a supported NVIDIA GPU and compatible RAPIDS, CUDA, Python, and library versions.

The original experiment was developed in Google Colab. RAPIDS installation instructions depend on the runtime and software versions; use the official RAPIDS installation guidance for your environment.

The dataset is retrieved through OpenML when the notebook runs. The reported results are from the recorded experiment and may vary across hardware, software versions, and execution environments.

## Limitations

- Results represent one dataset, one train/test split, and one model configuration.
- CPU and GPU implementations may differ in implementation details and tree-building behavior.
- GPU data transfer is excluded from the reported GPU model-fitting time.
- Equal overall accuracy does not necessarily mean both models made identical predictions.
- Results should not be generalized to all GPUs, datasets, or Random Forest configurations.

## References

- [MNIST Database of Handwritten Digits](http://yann.lecun.com/exdb/mnist/)
- [OpenML](https://www.openml.org/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [RAPIDS Documentation](https://docs.rapids.ai/)
- [CuPy Documentation](https://docs.cupy.dev/)

## Author

Şara Kaplan

---

*Educational project focused on GPU-accelerated machine learning and performance evaluation.*
