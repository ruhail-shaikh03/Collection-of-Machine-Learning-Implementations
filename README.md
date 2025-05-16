# Collection of Machine Learning Implementations

This repository contains implementations for several machine learning tasks, ranging from fundamental algorithms from scratch to more complex deep learning models for image classification, image generation, and machine translation.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [General Setup](#general-setup)
3.  [Running the Notebooks](#running-the-notebooks)
4.  [Project Details](#project-details)
    *   [4.1 Question 1: Implementing Rosenblatt’s Perceptron from Scratch](#41-question-1-implementing-rosenblatts-perceptron-from-scratch)
    *   [4.2 Question 2: Implementing Convolution from Scratch](#42-question-2-implementing-convolution-from-scratch)
    *   [4.3 Question 3: Implementing a CNN for CIFAR-10 (from `gen ai q3.ipynb`)](#43-question-3-implementing-a-cnn-for-cifar-10-from-gen-ai-q3ipynb)
    *   [4.4 Question 4: Vision Transformer vs CNN on CIFAR-10 (from `genai-a2-q4.ipynb`)](#44-question-4-vision-transformer-vs-cnn-on-cifar-10-from-genai-a2-q4ipynb)
    *   [4.5 Question 1 (Part 2): Image Generation using GAN and VAE on CIFAR-10](#45-question-1-part-2-image-generation-using-gan-and-vae-on-cifar-10)
    *   [4.6 Question 3 (Part 2): Machine Translation using Transformers for English-to-Urdu (from `gen ai q3.ipynb`)](#46-question-3-part-2-machine-translation-using-transformers-for-english-to-urdu-from-gen-ai-q3ipynb)
    *   [4.7 Question 4 (Part 2): Implementing a Vanilla RNN for Next-Word Prediction](#47-question-4-part-2-implementing-a-vanilla-rnn-for-next-word-prediction)
    *   [4.8 Question 5: Hyperparameter Search for CNN and RNN](#48-question-5-hyperparameter-search-for-cnn-and-rnn)
5.  [Overall Summary & Learnings](#overall-summary--learnings)
6.  [Future Work](#future-work)
7.  [Acknowledgements](#acknowledgements)

## 1. Project Overview

This collection of projects aims to provide a hands-on understanding of various machine learning and deep learning concepts. Implementations include:
*   **Rosenblatt's Perceptron:** Manual implementation for binary classification.
*   **2D Convolution:** Manual implementation to understand image filtering.
*   **CNN for Image Classification:** Building and evaluating a CNN on CIFAR-10.
*   **Vision Transformer vs. CNN:** Comparing ViT and CNN performance on a CIFAR-10 subset.
*   **Generative Models (GAN & VAE):** Image generation on a CIFAR-10 subset.
*   **Sequence-to-Sequence (Transformer):** English-to-Urdu machine translation.
*   **Vanilla RNN:** Next-word prediction using a custom RNN cell.
*   **Hyperparameter Optimization:** Using random search for CNN and RNN models.

The primary implementations are expected to be within Jupyter Notebooks (`gen ai q3.ipynb` and `genai-a2-q4.ipynb` cover specific tasks).

## 2. General Setup

1.  **Python Environment:**
    It is recommended to use a Python virtual environment (e.g., `venv` or `conda`).
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2.  **Dependencies:**
    Install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
    A `requirements.txt` file should include (but may not be limited to):
    ```
    numpy
    matplotlib
    pandas
    tensorflow  # Or pytorch, torchvision
    scikit-learn
    nltk        # For BLEU score, tokenization in RNN
    sentencepiece # For Transformer NMT tokenization
    seaborn     # For confusion matrices, plotting
    Pillow      # For image processing in convolution
    # Add other libraries like Hugging Face's datasets, transformers, timm if used
    ```
3.  **Datasets:**
    Specific dataset instructions are provided within each project section. Many, like CIFAR-10 or Shakespeare from Hugging Face, can be downloaded automatically by the respective libraries.

## 3. Running the Notebooks

*   All projects are primarily implemented and demonstrated within Jupyter Notebooks. The key notebooks identified are `gen ai q3.ipynb` and `genai-a2-q4.ipynb`. It's assumed other tasks might be in a combined notebook or separate ones.
*   Open the relevant notebook(s) in a Jupyter environment (Jupyter Lab, Jupyter Notebook, Google Colab, Kaggle Notebooks).
*   Run the cells sequentially.
*   Ensure that dataset paths (if manually downloaded) and model-saving paths are correctly configured within the notebooks.
*   Using a **GPU-enabled environment is highly recommended** for deep learning tasks (CNN, ViT, GAN, VAE, Transformer NMT, RNN) for feasible training times.

---

## 4. Project Details

### 4.1 Question 1: Implementing Rosenblatt’s Perceptron from Scratch

*   **Objective:** Implement a single-layer perceptron manually for binary classification, including forward pass, backward pass (weight updates using perceptron learning rule), dataset generation, visualization, and evaluation.
*   **Dataset:**
    *   Synthetic dataset: 500 samples, 2 features, binary labels (+1, -1).
    *   Class +1 centered at (s,s), class -1 at (-s,-s), sampled from a normal distribution.
    *   Normalized and split: 80% training, 20% testing.
*   **Implementation:**
    *   **Manual Core Functions:** Forward pass (weighted sum + step activation), weight update rule (`w ← w + ηyX`, `b ← b + ηy`).
    *   Weights and biases stored in NumPy matrices.
    *   Libraries like NumPy and Matplotlib used for data handling and visualization.
*   **Training & Evaluation:**
    *   Trained for 50 epochs.
    *   Error displayed per iteration.
    *   Decision boundary plotted after training.
    *   Test data points visualized on the decision boundary.
    *   Achieved **89% accuracy** on the test set.
*   **Discussion:** Successful for linearly separable data. Future work: MLP for complex datasets.
*   **How to Run:** Refer to the Perceptron section/cells in the main project notebook.

### 4.2 Question 2: Implementing Convolution from Scratch

*   **Objective:** Manually implement 2D convolution to understand its operations, kernel effects, and comparison with correlation.
*   **Methodology:**
    *   **Manual 2D Convolution Function:**
        *   Input: Grayscale image, kernel (user-defined or random default).
        *   Parameters: Kernel size, stride, padding ('valid' or 'same'), mode ('convolution' or 'correlation').
        *   Manual loop implementation for core operation.
    *   **Kernels Applied:** Blur, Sharpen, Edge detection (Sobel-like), Symmetric, Non-symmetric.
    *   **Comparison:**
        *   Manual vs. NumPy-based convolution.
        *   Convolution vs. Correlation, especially with symmetric and non-symmetric kernels.
*   **Analysis & Results:**
    *   Visualizations of original and convolved images.
    *   Effect of kernels (blur softens, sharpen enhances, edge detection highlights).
    *   Impact of kernel size, stride, padding (larger kernels smooth more, higher stride reduces resolution, 'same' padding preserves dimensions).
    *   Convolution vs. Correlation: Differences more evident with asymmetric kernels.
    *   Insights: Sobel for edges, uniform averaging for blur, sharpening can add noise, sequential kernels combine effects.
*   **How to Run:** Refer to the Convolution section/cells in the main project notebook.

### 4.3 Question 3: Implementing a CNN for CIFAR-10 (from `gen ai q3.ipynb`)

*(This appears to be the CNN part of the English-Urdu NMT notebook, or a separate CNN task. Assuming it's focused on CIFAR-10 classification as per the title in the PDF.)*
*   **Objective:** Build, train, and evaluate a CNN for image classification on CIFAR-10, including ablation studies.
*   **Dataset:**
    *   CIFAR-10 (60,000 images, 32x32 RGB, 10 classes). Downloaded from Hugging Face.
    *   Preprocessing: Normalization, one-hot encoding, 80/20 train/test split.
    *   Data Augmentation: Flipping, rotation, shifting (for one model variant).
*   **CNN Architecture (Baseline):**
    *   3 Convolutional Layers (ReLU activation).
    *   Max-Pooling Layers.
    *   Dropout.
    *   Fully Connected Layers.
    *   Softmax Output Layer.
    *   Variants: Deeper CNN, Batch Normalization, Dropout variations.
*   **Training & Evaluation:**
    *   Optimizer: Adam, SGD.
    *   Performance Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
    *   Loss and accuracy curves visualized.
    *   Feature map visualization from different layers.
*   **Ablation Study:**
    *   Impact of: Learning Rate, Batch Size, Number of Filters, Number of Layers.
    *   Results compared in terms of accuracy, parameters, inference time (e.g., Baseline CNN: 85.2% Acc).
*   **Results:**
    *   Comparison table for models with/without data augmentation.
    *   Confusion matrices and discussion of misclassifications.
*   **How to Run:** Implemented within `gen ai q3.ipynb`. Run cells related to CNN classification.

### 4.4 Question 4: Vision Transformer vs. CNN on CIFAR-10 (from `genai-a2-q4.ipynb`)

*   **Objective:** Implement and compare a CNN and a ViT for image classification on a subset of CIFAR-10 (cats vs. deer).
*   **Dataset:**
    *   CIFAR-10, filtered for "cat" and "deer" classes (10k training, 2k testing).
*   **CNN Implementation:**
    *   3 convolutional blocks (32, 64, 128 filters), each with 2 Conv2D (ReLU) + BatchNorm + MaxPooling + Dropout.
    *   Dense layers with L2 regularization.
    *   Optimizer: Adam. Loss: Binary Crossentropy.
    *   Data Augmentation: Random flip, brightness.
*   **ViT Implementation:**
    *   Patch size: 4x4 (64 patches). Projection dim: 64.
    *   PatchExtractor, PatchEncoder (with positional embeddings).
    *   6 Transformer encoder layers (4 heads each).
    *   MLP head for classification.
    *   Optimizer: AdamW. Loss: Binary Crossentropy.
    *   Data Augmentation: Random rotation, zoom, contrast. ViT-specific normalization (ImageNet stats).
*   **Training & Evaluation:**
    *   Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
    *   Metrics: Accuracy, Precision, Recall, F1-Score, Loss curves, Confusion Matrix.
*   **Comparison & Results:**
    *   CNN achieved ~93.6% accuracy.
    *   ViT (trained from scratch) achieved ~77.0% accuracy.
    *   Analysis of performance, training dynamics, model size, and modeling effectiveness.
*   **How to Run:** Implemented within `genai-a2-q4.ipynb`.

### 4.5 Question 1 (Part 2): Image Generation using GAN and VAE on CIFAR-10

*   **Objective:** Train and compare a custom GAN (with similarity discriminator) and a VAE for generating "cat" and "dog" images from CIFAR-10.
*   **Dataset:** CIFAR-10, filtered for "cat" and "dog" classes.
*   **GAN Implementation:**
    *   **Generator:** DCGAN-like architecture.
    *   **Custom Similarity Discriminator:**
        *   Takes a pair of images (real, generated).
        *   Outputs a similarity score.
        *   Potentially Siamese Network architecture.
        *   Aims to maximize dissimilarity for (real, fake) pairs.
    *   **Diversity-Promoting Technique:** Mini-Batch Discrimination, Feature Matching, or other.
*   **VAE Implementation:**
    *   Standard Encoder (outputs μ, log(σ^2)) and Decoder architecture.
    *   Reparameterization trick.
    *   Loss: Reconstruction Loss + KL Divergence.
*   **Training:** Both models trained on cat/dog images.
*   **Evaluation & Comparison:**
    *   Visual quality of generated images.
    *   Convergence behavior.
    *   Quantitative metrics (reconstruction loss for VAE, similarity score for GAN, FID/IS if feasible).
*   **How to Run:** Refer to the GAN/VAE section/cells in the main project notebook.

### 4.6 Question 3 (Part 2): Machine Translation using Transformers for English-to-Urdu (from `gen ai q3.ipynb`)

*   **Objective:** Develop an English-to-Urdu NMT model using Transformer architecture.
*   **Dataset:**
    *   Parallel Corpus for English-Urdu Language (Kaggle, ~24k sentence pairs after cleaning).
    *   Preprocessing: Cleaning (punctuation, extra spaces, language-specific character removal), train/validation split (85/15).
*   **Implementation:**
    *   **Subword Tokenization:** SentencePiece (BPE) for English and Urdu (Vocab size ~7000 each). BOS/EOS tokens added to Urdu.
    *   **Transformer Architecture:**
        *   D_MODEL=512, NUM_LAYERS=6, NUM_HEADS=8, DFF=2048, DROPOUT_RATE=0.1.
        *   Positional Encoding, Multi-Head Attention, Encoder layers, Decoder layers.
        *   Padding and Look-Ahead masks.
*   **Training:**
    *   Optimizer: AdamW with `CustomSchedule` (warmup steps).
    *   Loss: Custom sparse categorical crossentropy with Label Smoothing.
    *   Metrics: `masked_accuracy`.
    *   Callbacks: EarlyStopping, ModelCheckpoint, TensorBoard.
*   **Evaluation:**
    *   Corpus BLEU score on validation set (achieved ~3.55, indicating room for improvement).
    *   Qualitative examples of translations.
    *   Training/validation loss and accuracy curves.
*   **How to Run:** Implemented within `gen ai q3.ipynb`. Run cells related to NMT.
    *   *Note: Notebook output indicated a `Lambda` layer loading issue, consider `safe_mode=False` or refactoring lambdas if re-loading saved models.*

### 4.7 Question 4 (Part 2): Implementing a Vanilla RNN for Next-Word Prediction

*   **Objective:** Train a Vanilla RNN on Shakespeare text for next-word prediction, comparing randomly initialized vs. pre-trained embeddings.
*   **Dataset:** Shakespeare text dataset (from Hugging Face).
    *   Preprocessing: Tokenization, vocab creation, 80/20 train/test split, fixed-length sequences.
*   **Vanilla RNN Implementation:**
    *   **Custom RNN Cell** (manual implementation, no LSTM/GRU).
    *   Trainable Embedding Layer.
    *   SimpleRNN layer (using the custom cell or framework's if allowed for wrapper).
    *   Dense output layer with softmax.
*   **Training:**
    *   Backpropagation Through Time (BPTT).
    *   Loss: Cross-Entropy. Optimizer: Adam (LR 0.001). Early stopping.
*   **Evaluation:**
    *   Text Generation from a seed phrase (e.g., "To be or not to").
    *   Metrics: Perplexity, Word-level accuracy (achieved ~81.32% with random init, ~85.67% with Word2Vec).
    *   Loss curve visualization.
    *   Confusion matrix for misclassified words.
*   **Ablation Study (Embeddings):**
    *   Randomly initialized embeddings.
    *   Pre-trained Word2Vec embeddings.
    *   Results: Word2Vec improved accuracy and reduced perplexity.
*   **How to Run:** Refer to the RNN section/cells in the main project notebook.

### 4.8 Question 5: Hyperparameter Search for CNN and RNN

*   **Objective:** Implement random search for hyperparameter optimization for CNN (image classification) and RNN (text classification/next-word prediction) models.
*   **Methodology:**
    *   Random Search technique (RandomizedSearchCV from Scikit-Learn or custom sampling).
*   **Hyperparameters Searched (examples):**
    *   Learning rate, batch size.
    *   Number of layers, number of neurons/filters.
    *   Optimizer (Adam, SGD, RMSprop).
    *   Activation functions (ReLU, Tanh, Sigmoid).
    *   Dropout rate.
    *   Kernel size, stride (for CNN).
    *   Weight initialization.
*   **Process:**
    *   Train multiple models with different hyperparameter combinations.
    *   Select best configuration based on validation accuracy.
    *   Test best CNN and RNN configurations on their respective test datasets.
*   **Comparison:** Compare optimized CNN and RNN performance using evaluation metrics.
*   **How to Run:** Refer to the Hyperparameter Search section/cells in the main project notebook.

---

## 5. Overall Summary & Learnings

This collection of projects provides practical experience in:
*   Implementing fundamental ML algorithms from scratch.
*   Understanding the mechanics of core deep learning operations like convolution.
*   Building and training various neural network architectures (CNN, ViT, GAN, VAE, Transformer, RNN) for different tasks.
*   Data preprocessing, tokenization, and augmentation techniques.
*   Model evaluation using appropriate metrics and ablation studies.
*   The importance of hyperparameter tuning and its impact on model performance.
*   Comparing different architectural approaches (e.g., CNN vs. ViT, GAN vs. VAE).

## 6. Future Work

*   **Perceptron/Convolution:** Extend to multi-layer/multi-channel versions.
*   **CNN/ViT:** Experiment with larger datasets, more advanced pre-trained models, and different attention mechanisms.
*   **GAN/VAE:** Explore more advanced architectures (StyleGAN, β-VAE), improved loss functions, and more rigorous evaluation metrics (FID, IS for GANs).
*   **Transformer NMT:** Use larger corpora, fine-tune larger pre-trained NMT models (mBART, NLLB), implement beam search decoding.
*   **RNN:** Explore LSTM/GRU cells, attention mechanisms for text generation, larger datasets.
*   **Hyperparameter Search:** Utilize more advanced techniques like Bayesian Optimization or automated HPO tools.

## 7. Acknowledgements

*   Respective authors of the original papers for Perceptron, Convolution, CNNs, ViT, GAN, VAE, Transformer, RNNs.
*   Creators of datasets used (CIFAR-10, Shakespeare, English-Urdu Parallel Corpus).
*   Developers of libraries like NumPy, Matplotlib, TensorFlow, PyTorch, Scikit-learn, NLTK, SentencePiece.
*   [Your Institution/Course Name, if applicable]

