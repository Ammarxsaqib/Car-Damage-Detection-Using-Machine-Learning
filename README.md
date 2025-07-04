The system uses two primary models:

CNN for Car Damage Detection (binary classification: damaged/not damaged).

Fine-Tuned VGG16 for Damage Intensity Classification (multi-class: minor/severe).

1. Car Damage Detection Model (CNN)
Architecture
Type: Convolutional Neural Network (CNN).

Layers:

Convolutional + Pooling layers (extract features).

Dense layers (classification).

Dropout layers (prevent overfitting).

Output: Binary (Damaged / Not Damaged).

Training
Dataset: Labeled car images (damaged vs. undamaged).

Preprocessing:

Resizing (e.g., 224x224 pixels).

Normalization (pixel values scaled to [0, 1]).

Data Augmentation (rotation, flipping, etc.).

Optimization:

Loss Function: Binary Cross-Entropy.

Optimizer: Adam.

Techniques: Early stopping, dropout.

Performance Metrics
Target: High accuracy (>90%) and low validation loss.

2. Damage Intensity Classification Model (VGG16)
Architecture
Base Model: Pre-trained VGG16 (ImageNet weights).

Modifications:

Added dense layers (custom head) for fine-tuning.

Output layer: 2 classes (Minor/Severe damage).

Transfer Learning: Frozen initial layers, trained only added layers.

Training
Dataset: Subset of damaged car images labeled for intensity.

Preprocessing:

Same as CNN (resizing, normalization).

Possible grayscale conversion (if applicable).

Fine-Tuning:

Loss Function: Categorical Cross-Entropy.

Optimizer: SGD or Adam with low LR (e.g., 0.001).

Epochs: Limited to avoid overfitting.

Performance Metrics
Target: High precision/recall for severity classes.

Integration Pipeline
Input: User uploads image via Kotlin app.

Step 1: CNN model detects if damage exists.

Step 2: If damaged, VGG16 classifies intensity.

Output: Results sent back to app (e.g., "Severe Damage Detected").

Challenges & Solutions
Challenge	Solution
Limited labeled data	Data augmentation, synthetic data generation.
Model overfitting	Dropout, early stopping, regularization.
Slow inference	Optimized model architecture, cloud deployment.
Future Model Improvements
Dataset Expansion: More diverse damage scenarios.

Multi-class Intensity: Add granularity (e.g., "moderate" damage).

Real-time Processing: Optimize models for video frames.

Key Technologies
Frameworks: TensorFlow/Keras (Python).

Libraries: OpenCV (image preprocessing).

Deployment: Cloud (AWS/Google Cloud) for scalable inference.
