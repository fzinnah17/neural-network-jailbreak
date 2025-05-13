# Neural Network Jailbreak: Adversarial Attacks on Image Classifiers

## Project Overview
This project implements and evaluates various adversarial attack methods against state-of-the-art deep learning image classifiers. The study focuses on three primary attack strategies: Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), and localized patch attacks. We demonstrate how these attacks can dramatically reduce the accuracy of ResNet-34 and DenseNet-121 models on ImageNet data, achieving up to 99.74% relative accuracy drop while maintaining imperceptible perturbations.

## Results Highlights

- **FGSM Attack**: Simple one-step attack achieving 99.21% relative drop in accuracy (76.00% → 0.60%)
- **PGD Attack**: Iterative attack improving upon FGSM with 99.74% relative drop (76.00% → 0.20%)
- **Patch Attack**: Localized attack achieving 90.79% relative drop (76.00% → 7.00%) while modifying only 5.34% of pixels
- **Transferability**: High cross-architecture vulnerability with transfer rates exceeding 90%
- **Model Comparison**: ResNet-34 slightly more vulnerable than DenseNet-121 across all attack types

## Setup Instructions

### Requirements
- Python 3.8+
- PyTorch 1.9+
- torchvision
- NumPy
- Pandas
- matplotlib
- seaborn
- tqdm
- scikit-learn
- Jupyter

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

**Notable Findings:**
- Patch attacks showed the highest transferability (97.97%)
- All attack types achieved >90% transfer rates
- Suggests shared vulnerabilities across architectures

## Project Structure
```
neural-network-jailbreak/
├── TestDataSet/                  # Original test images
├── AdversarialTestSet1/          # FGSM adversarial examples
├── AdversarialTestSet2/          # PGD adversarial examples
├── AdversarialTestSet3/          # Patch attack adversarial examples
├── script.ipynb                  # Main implementation notebook
├── script_executed.ipynb         # Executed notebook with outputs
├── accuracies.txt                # Accuracy results for all experiments
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
