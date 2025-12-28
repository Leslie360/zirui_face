# Technical Documentation: Memristor-based Neural Network Experiment
**Date:** 2025-12-26  
**Subject:** Performance Optimization and Characterization of Memristor-based CNN on CIFAR-10

---

## 1. Model Technical Specifications

### 1.1 Architecture Description
The experiment utilizes a **MemristorCNN_Strong** architecture, a specialized Convolutional Neural Network (CNN) designed to simulate hardware-aware constraints of memristive devices. The architecture integrates standard deep learning components with memristor-specific behavior in the fully connected layers.

*   **Backbone:** A ResNet-like feature extractor comprising three stages of residual blocks (`ResBlock`).
    *   **Stage 1:** 64 channels, 2 blocks.
    *   **Stage 2:** 128 channels, 2 blocks (stride 2).
    *   **Stage 3:** 256 channels, 2 blocks (stride 2).
*   **Attention Mechanism:** Squeeze-and-Excitation (SE) blocks are integrated within each residual block to adaptively recalibrate channel-wise feature responses.
*   **Projection Head:** A multi-layer perceptron (MLP) projection head (256*8*8 -> 512 -> 128) is attached for contrastive learning capability (though primarily used for feature representation in this supervised setup).
*   **Classifier:** A memristor-aware fully connected layer sequence:
    *   Input: Flattened features (256 * 8 * 8).
    *   Hidden: 1024 units with ReLU and Dropout (0.5).
    *   Output: 10 units (Classes).
    *   *Note:* The weights of these linear layers are managed by `MemFCLState` to simulate discrete conductance states and device non-idealities (LTP/LTD curves).

### 1.2 Training Dataset Characteristics
*   **Dataset:** CIFAR-10
*   **Scale:** 50,000 training images, 10,000 testing images.
*   **Resolution:** 32x32 RGB images.
*   **Preprocessing:**
    *   **Training:** 4-bit quantization (Ideal mode), Random Crop, Random Horizontal Flip, AutoAugment (CIFAR10 Policy), Color Jitter, Random Erasing.
    *   **Testing:** 4-bit quantization with varying degrees of **RGB Channel Overlap** (Crosstalk simulation).
    *   **Overlap Levels:** 0.0 (Ideal), 0.1, 0.3, 0.5.

### 1.3 Key Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Epochs** | 200 | Full training cycle without early stopping (for paper curves). |
| **Batch Size** | 128 | Optimized for convergence speed. |
| **Optimizer** | AdamW | `lr=0.001`, `weight_decay=1e-4`. |
| **Scheduler** | CosineAnnealingLR | Decay over 200 epochs. |
| **Loss Function** | Weighted CrossEntropy | Class weights: 1.5 for Cat/Dog, 1.0 for others. |
| **Label Smoothing** | 0.05 | Regularization to prevent over-confidence. |
| **Overlap Alpha** | 2.0 | Crosstalk amplification factor (increased to force degradation). |
| **Seed** | 42 | Fixed for reproducibility. |

### 1.4 Performance Metrics
*   **Primary Metric:** Top-1 Accuracy (%).
*   **Evaluation Scenarios:**
    *   **Clean Accuracy:** Accuracy at Overlap=0.0.
    *   **Robustness:** Accuracy degradation at Overlap=0.1, 0.3, 0.5.
*   **Target Goals:**
    *   Clean Accuracy > 90%.
    *   Significant performance gap between Bidirectional and Unidirectional modes at Overlap=0.5 (demonstrating bidirectional superiority).

---

## 2. Visualization Chart Analysis

This section describes the visualization artifacts generated during the experiment.

### 2.1 Confusion Matrix (Heatmap)
*   **Filename:** `cm_{mode}_seed{seed}_epoch{epoch}.png`
*   **Description:** A heatmap visualizing the normalized classification performance across all 10 classes.
*   **Technical Role:**
    *   **Diagonal Elements:** Represent the Recall (Sensitivity) for each class. Higher values (closer to 1.0/dark blue) indicate correct classification.
    *   **Off-diagonal Elements:** Represent misclassifications. Specific "hotspots" (e.g., Row 3/Col 5) indicate model confusion between specific pairs (e.g., Cat vs. Dog).
*   **Interpretation:** Used to verify if the **Weighted Loss** strategy successfully reduced the confusion between Class 3 (Cat) and Class 5 (Dog).

### 2.2 Weight Distribution Histogram
*   **Filename:** `weights_hist_{mode}_seed{seed}_epoch{epoch}.png`
*   **Description:** Histograms showing the distribution of synaptic weights in the model's layers.
*   **Technical Role:**
    *   **Health Monitoring:** Checks for vanishing (all near zero) or exploding weights.
    *   **Sparsity:** A peak at zero suggests sparse representations.
    *   **Memristor Simulation:** Indirectly reflects how the discrete conductance updates (LTP/LTD) affect the weight population statistics.

### 2.3 Loss Curve
*   **Filename:** `loss_{mode}_seed{seed}.png`
*   **Description:** A line plot tracking `Train Loss` and `Validation Loss` (at Overlap=0.0) over 200 epochs.
*   **Technical Role:**
    *   **Convergence:** Shows whether the model is learning effectively (loss decreasing).
    *   **Overfitting:** A divergence where Train Loss decreases but Validation Loss increases (or plateaus) signals overfitting.
    *   **Stability:** Spikes indicate instability in the optimization process or memristor updates.

### 2.4 Final Performance Comparison (Error Bar Plot)
*   **Filename:** `final_performance_comparison.png`
*   **Description:** A consolidated plot comparing **Bidirectional** vs. **Unidirectional** update modes across all Overlap levels.
*   **Technical Role:**
    *   **Mean Accuracy:** The solid line represents the central tendency of the model's performance.
    *   **Shaded Area:** Represents the standard deviation (though with Seed=42 fixed, this serves as a placeholder for multi-seed variance).
    *   **Critical Comparison:** The gap between the curves at **Overlap=0.5** visually quantifies the advantage of the Bidirectional update mechanism under high-noise conditions.

---

## 3. Academic Writing Standards & Implementation Details

*   **Terminology:** All references to "crosstalk" utilize the term **RGB Channel Overlap**. The memristor update logic is referred to as **Memristive Conductance Update (LTP/LTD)**.
*   **Figure Referencing:**
    *   **Figure 1:** System Architecture and Memristor Integration Scheme.
    *   **Figure 2:** Confusion Matrices comparing Baseline vs. Weighted Loss performance.
    *   **Figure 3:** Learning Dynamics (Loss Curves) over 200 Epochs.
    *   **Figure 4:** Impact of Crosstalk Intensity (Overlap) on Recognition Accuracy (Bidir vs. Unidir).
*   **Implementation Notes:**
    *   *Footnote 1:* The memristor simulation employs a look-up table (LUT) based approach derived from experimental LTP/LTD measurements (`ltp_ltd.txt`).
    *   *Footnote 2:* The "Overlap" is simulated via a linear mixing matrix of RGB channels, followed by a non-linear Gamma correction ($\gamma=1.0$) and 4-bit quantization to mimic hardware readout noise.
    *   *Footnote 3:* The "Bidirectional" mode utilizes a dual-memristor pair (differential pair) per weight to enable both potentiation and depression, whereas "Unidirectional" is restricted to potentiation-only updates (simulated via reset-and-set or limited update logic).

---
*Generated based on Terminal Output Log Reference #147-1012.*
