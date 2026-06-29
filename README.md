# Makemore: Building Language Models from Scratch

> A from-scratch implementation of **Andrej Karpathy's Makemore** series using **PyTorch**, documenting the evolution of language models from simple statistical methods to modern Transformer architectures.

This repository follows the progression of the Makemore series, implementing each model from first principles to build an intuitive understanding of how language models work internally.

The models are trained on a dataset of **32,000+ unique human names**, learning to generate entirely new, phonetically plausible names one character at a time.

---

## Project Roadmap

| Part | Architecture                      |     Status     |
| :--- | :-------------------------------- | :------------: |
| 1    | Bigram Language Model             |   ✅ Completed  |
| 2    | Multi-Layer Perceptron (MLP)      | 🚧 In Progress |
| 3    | Batch Normalization & Activations |     Planned    |
| 4    | WaveNet                           |     Planned    |
| 5    | GPT / Transformer                 |     Planned    |

---

## Implemented Models

### Part 1 : Bigram Language Model

A character-level language model implemented using both statistical methods and a neural network.

**Concepts Covered**

* Character transition probabilities
* Maximum Likelihood Estimation (MLE)
* One-hot encoding
* Negative Log-Likelihood (NLL)
* Matrix multiplication
* Gradient descent
* Character-level text generation

---

### Part 2 : Multi-Layer Perceptron (Current)

Currently implementing the neural probabilistic language model introduced in **Bengio et al. (2003)**.

**Implemented**

* Character embedding lookup table
* Context window construction
* Dynamic tensor reshaping using `.view()`
* Hidden layers with `tanh` activation
* Output logits
* Numerically stable cross-entropy loss using `torch.nn.functional.cross_entropy`

**Next**

* Manual backpropagation
* Parameter optimization
* Learning rate tuning
* Mini-batch training
* Model evaluation
* Character-level name generation

---

## Learning Objectives

This project focuses on developing a first-principles understanding of:

* Statistical language modeling
* Distributed representations (Embeddings)
* Neural probabilistic language models
* Forward and backward propagation
* Tensor operations in PyTorch
* Gradient-based optimization
* Numerical stability
* Training deep neural networks

---

## Tech Stack

* **Python**
* **PyTorch**
* **NumPy**
* **Matplotlib**
* **Jupyter Notebook**

---

## Repository Structure

```text
makemore/
├── README.md
├── names.txt
├── bigrams.ipynb
└── mlp.ipynb
```

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/DaOhesis/makemore.git
cd makemore
```

Install the dependencies:

```bash
pip install torch numpy matplotlib jupyter
```

Launch the notebooks:

```bash
# Bigram Language Model
jupyter notebook bigrams.ipynb

# Multi-Layer Perceptron
jupyter notebook mlp.ipynb
```

---

## References

* Andrej Karpathy — *Makemore* Series
* Bengio et al. (2003) — *A Neural Probabilistic Language Model*
* PyTorch Documentation

---

## Author

**Sanchayan Chakraborty**

If you found this repository helpful, consider giving it a ⭐.
