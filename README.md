# Neural Network Jailbreak: Adversarial Attacks on Image Classifiers

## Project Overview
This project implements and evaluates various adversarial attack methods against state-of-the-art deep learning image classifiers. The study focuses on three primary attack strategies: Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), and localized patch attacks. We demonstrate how these attacks can dramatically reduce the accuracy of ResNet-34 and DenseNet-121 models on ImageNet data, achieving up to 99.74% relative accuracy drop while maintaining imperceptible perturbations.

## Results Highlights

- **FGSM Attack**: Simple one-step attack achieving 99.21% relative drop in accuracy (76.00% → 0.60%)
- **PGD Attack**: Iterative attack improving upon FGSM with 99.74% relative drop (76.00% → 0.20%)
- **Patch Attack**: Localized attack achieving 90.79% relative drop (76.00% → 7.00%) while modifying only 5.34% of pixels
- **Transferability**: High cross-architecture vulnerability with transfer rates exceeding 90%
- **Model Comparison**: ResNet-34 slightly more vulnerable than DenseNet-121 across all attack types

## Key Results Summary

| Attack Method | Parameters | ResNet-34 Accuracy |  | DenseNet-121 Accuracy |  | Modified Pixels | Transfer Rate |
|---------------|:----------:|:------------------:|:------------------:|:---------------------:|:---------------------:|:---------------:|:-------------:|
|               |            | Top-1 | Top-5 | Top-1 | Top-5 | L0 Norm (%) | (%) |
| Original (Clean) | - | 76.00% | 94.00% | 75.60% | 93.60% | - | - |
| FGSM | ε = 0.02 | 0.60% | 3.80% | 6.80% | 11.40% | 293.16% | 91.25% |
| PGD | ε = 0.02, steps = 10 | 0.20% | 1.00% | 7.00% | 12.00% | 288.37% | 90.50% |
| Patch | ε = 0.3, 32×32 px | 7.00% | 10.20% | 8.00% | 12.00% | 5.34% | 97.97% |

## Setup Instructions

### Required Packages

| Package | Version | Purpose |
|---------|:-------:|---------|
| Python | ≥ 3.8 | Programming language |
| PyTorch | ≥ 1.9.0 | Deep learning framework |
| torchvision | ≥ 0.10.0 | Computer vision utilities |
| NumPy | ≥ 1.20.0 | Numerical computing |
| Pandas | ≥ 1.3.0 | Data manipulation |
| matplotlib | ≥ 3.4.0 | Visualization |
| seaborn | ≥ 0.11.0 | Enhanced visualization |
| scikit-learn | ≥ 0.24.0 | Machine learning utilities |
| tqdm | ≥ 4.60.0 | Progress bars |
| Jupyter | ≥ 1.0.0 | Notebook environment |

### Installation
```bash
# Clone the repository
git clone https://github.com/[username]/neural-network-jailbreak.git
cd neural-network-jailbreak

# Install required packages
pip install torch torchvision numpy pandas matplotlib seaborn tqdm scikit-learn jupyter
```

## Usage

### Running the Jupyter Notebook
```bash
# Run the notebook interactively
jupyter notebook script.ipynb

# Or view the pre-executed notebook
jupyter notebook script_executed.ipynb
```

### Dataset
The project uses a subset of the ImageNet dataset with 500 test images across 100 classes:
```
neural-network-jailbreak/
└── TestDataSet/
    ├── class_0/
    │   ├── image_0.jpg
    │   ├── image_1.jpg
    │   └── ...
    ├── class_1/
    │   ├── image_0.jpg
    │   └── ...
    └── ...
```

## Results in Detail

### Baseline Model Performance

**Model Accuracy on Clean Test Images:**
```
ResNet-34 Results:
Original Test Set - Top-1 Accuracy: 76.00%, Top-5 Accuracy: 94.00%

DenseNet-121 Results:
Original Test Set - Top-1 Accuracy: 75.60%, Top-5 Accuracy: 93.60%
```

### Baseline Model Performance

![Confusion Matrix](https://github.com/fzinnah17/neural-network-jailbreak/blob/main/images/clean_confusion_matrix.png)

*Confusion matrix showing strong diagonal performance on clean images*


### FGSM Attack Results (ε = 0.02)

This simple one-step attack produced dramatic accuracy reduction while maintaining imperceptible perturbations:

```
ResNet-34 Results:
FGSM Attack - Top-1 Accuracy: 0.60%, Top-5 Accuracy: 3.80%
Absolute drop in Top-1 Accuracy: 75.40%
Relative drop in Top-1 Accuracy: 99.21%

DenseNet-121 Results:
FGSM Attack - Top-1 Accuracy: 6.80%, Top-5 Accuracy: 11.40%
```
### FGSM Attack Results (ε = 0.02)

![FGSM Attack](https://github.com/fzinnah17/neural-network-jailbreak/blob/main/images/fgsm_triplet_visualization.png)

*FGSM attack visualization: Original image (left), adversarial image (center), and perturbation amplified 10× (right)*

**Key Statistics:**
- L∞ norm (maximum change): 0.0200
- L0 norm (% pixels changed): 293.16%
- Most images completely misclassified despite no visible changes

### PGD Attack Results (ε = 0.02)

Our iterative attack achieved even better results than FGSM:

```
ResNet-34 Results:
PGD Attack - Top-1 Accuracy: 0.20%, Top-5 Accuracy: 1.00%
Absolute drop in Top-1 Accuracy: 75.80%
Relative drop in Top-1 Accuracy: 99.74%

DenseNet-121 Results:
PGD Attack - Top-1 Accuracy: 7.00%, Top-5 Accuracy: 12.00%
```

### PGD Attack Results (ε = 0.02)

![PGD Attack](https://github.com/fzinnah17/neural-network-jailbreak/blob/main/images/pgd_triplet_visualization.png)

*PGD attack visualization showing imperceptible perturbations that create misclassification*

![PGD Probability Decay](https://github.com/fzinnah17/neural-network-jailbreak/blob/main/images/pgd_probability_graph.png)

*Evolution of class probabilities during PGD attack iterations*

**Key Statistics:**
- L∞ norm (maximum change): 0.0200
- L0 norm (% pixels changed): 288.37%
- Improved effectiveness through 10 iterations of gradient descent

### Patch Attack Results (ε = 0.3)

The localized attack demonstrated impressive effectiveness despite modifying only a small region:

```
ResNet-34 Results:
Patch Attack - Top-1 Accuracy: 7.00%, Top-5 Accuracy: 10.20%
Absolute drop in Top-1 Accuracy: 69.00%
Relative drop in Top-1 Accuracy: 90.79%

DenseNet-121 Results:
Patch Attack - Top-1 Accuracy: 8.00%, Top-5 Accuracy: 12.00%
```

### Patch Attack Results (ε = 0.3)

![Patch Attack](https://github.com/fzinnah17/neural-network-jailbreak/blob/main/images/patch_triplet_visualization.png)

*Patch attack affecting only 5.34% of pixels yet causing misclassification*

**Key Statistics:**
- L∞ norm (maximum change): 0.3000
- L0 norm (% pixels changed): 5.34%
- Patch size: 32×32 pixels
- Effectiveness varies significantly based on patch location

### Attack Transferability

Adversarial examples crafted for ResNet-34 remained highly effective against DenseNet-121:

```
Transferability Results:
FGSM Transferability Rate: 91.25%
PGD Transferability Rate: 90.50%
Patch Transferability Rate: 97.97%
```

## Cross-Model Transferability Matrix

| Attack Generated For | Tested On | Top-1 Accuracy | Top-5 Accuracy | Transfer Success Rate |
|----------------------|-----------|:--------------:|:--------------:|:---------------------:|
| ResNet-34 (Original) | ResNet-34 | 76.00% | 94.00% | - |
| ResNet-34 (FGSM) | ResNet-34 | 0.60% | 3.80% | 99.21% |
| ResNet-34 (PGD) | ResNet-34 | 0.20% | 1.00% | 99.74% |
| ResNet-34 (Patch) | ResNet-34 | 7.00% | 10.20% | 90.79% |
| ResNet-34 (Original) | DenseNet-121 | 75.60% | 93.60% | - |
| ResNet-34 (FGSM) | DenseNet-121 | 6.80% | 11.40% | 91.25% |
| ResNet-34 (PGD) | DenseNet-121 | 7.00% | 12.00% | 90.50% |
| ResNet-34 (Patch) | DenseNet-121 | 8.00% | 12.00% | 97.97% |


**Notable Findings:**
- Patch attacks showed the highest transferability (97.97%)
- All attack types achieved >90% transfer rates
- Suggests shared vulnerabilities across architectures

## Project Structure
```
neural-network-jailbreak/
├── images
├── script.ipynb                  # Main implementation notebook
├── script_executed.ipynb         # Executed notebook with outputs
├── accuracy_results.txt                # Accuracy results for all experiments
├── report.pdf                    # Research paper summarizing findings
└── README.md                     # Project documentation
```

## Key Visualizations

The project includes several important visualizations:

1. **Model Prediction**: Original image with top-5 prediction confidences
2. **FGSM Attack**: Original image, adversarial image, and 10× amplified perturbation
3. **PGD Evolution**: Graph showing probability decay over PGD iterations
4. **Confusion Matrix**: Classification performance on clean test set
5. **PGD Attack**: Comparison of original, adversarial, and perturbation visualizations
6. **Patch Attack**: Visualization showing localized perturbation with 32×32 patch

## Running on NYU HPC

To run this project on NYU's High-Performance Computing (HPC) cluster:

### 1. Connect to HPC
```bash
# Install Cisco AnyConnect VPN for your OS
# Connect to NYU VPN, then SSH to Greene
ssh netid@greene.hpc.nyu.edu
# Enter your password

# Connect to the compute node
ssh burst
```

### 2. Request GPU Resources
```bash
# For V100 GPU:
srun --account=ece_gy_7123-2025sp --partition=n1s8-v100-1 --gres=gpu:v100:1 --time=04:00:00 --pty /bin/bash
```

### 3. Setup Container Environment
```bash
# Start Singularity container
singularity exec --bind /scratch --nv --overlay /scratch/netid/overlay-25GB-500K.ext3:rw /scratch/netid/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash

# Inside the container
Singularity> source /ext3/env.sh
Singularity> conda activate base
(base) Singularity> cd /scratch/netid/neural-network-jailbreak
```

### 4. Run the Notebook on HPC
```bash
# Install required packages
(base) Singularity> pip install torch torchvision numpy pandas matplotlib seaborn tqdm scikit-learn jupyter

# Execute the notebook non-interactively
(base) Singularity> jupyter nbconvert --to notebook --execute script.ipynb --output script_executed.ipynb
```

## Conclusion

This project demonstrates the alarming vulnerability of state-of-the-art deep learning models to adversarial attacks. Even simple one-step attacks can reduce classification accuracy from over 75% to less than 1% while maintaining imperceptible perturbations. More concerning, patch attacks achieve similar effectiveness while modifying only a tiny portion of the image, and attacks transfer across different model architectures with high success rates.

These findings highlight significant security concerns for deployment of deep learning systems in critical applications, where adversaries might exploit these vulnerabilities to cause misclassification. The results emphasize the urgent need for robust defenses against adversarial manipulation.
