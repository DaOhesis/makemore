# Makemore Portfolio: Building Language Models from Scratch

This repository documents my implementation of the **Makemore** series by **Andrej Karpathy**, exploring the foundations of generative AI and language modeling from simple statistical methods to modern neural architectures.

The project trains character-level language models on a dataset of **32,000+ unique names**, gradually building models capable of generating entirely new, phonetically plausible human names.

## 📚 Project Roadmap

This repository is being updated progressively as I work through each architecture in the series.

### Part 1: Bigram Language Model ✅

A simple statistical language model built using character transition counts and a single-layer neural network.

### Part 2: MLP Language Model 🚧

Implementing a Multi-Layer Perceptron based on the **Bengio et al. (2003)** paper, introducing embeddings and nonlinear representations.

### Part 3: Batch Normalization & Activations 🚧

Exploring deep network initialization, activation functions, manual backpropagation, and training diagnostics.

### Part 4: WaveNet 🚧

Building a hierarchical convolutional architecture capable of modeling longer contexts efficiently.

### Part 5: GPT / Transformer 🚧

Implementing the core self-attention mechanisms that power modern Large Language Models.

---

##  Current Status: Part 1 — Character Bigram Model

The first phase focuses on building a character-level bigram language model trained on a dataset of names.

### Key Concepts Implemented

#### Character Pair Extraction

Using Python's `zip()` function to create overlapping character pairs while preserving word boundaries through special start and end tokens.

#### Transition Matrix Construction

Building a **27 × 27** tensor in PyTorch that stores the frequency of character-to-character transitions.

#### Probability Normalization

Converting raw frequency counts into valid probability distributions through row-wise normalization.

#### Stochastic Sampling

Using `torch.multinomial()` to sample characters according to learned probabilities, enabling the generation of entirely new names.

#### Neural Network Formulation

Re-implementing the bigram model as a single-layer neural network and optimizing it using gradient descent.

---

##  Technologies Used

* Python
* PyTorch
* NumPy
* Matplotlib
* Jupyter Notebook

---

##  Running the Project

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Dependencies

```bash
pip install torch numpy matplotlib jupyter
```

### 3. Launch the Notebook

```bash
jupyter notebook 01_bigram_language_model/bigrams.ipynb
```

---

##  Learning Objectives

Through this project, I aim to develop a first-principles understanding of:

* Statistical language modeling
* Neural network fundamentals
* Gradient-based optimization
* Character embeddings
* Sequence modeling
* Attention mechanisms and Transformers

Each notebook focuses on implementing concepts from scratch rather than relying on high-level abstractions, providing a deeper understanding of how modern language models work under the hood.
