# Decoding Cognitive States: Using Transformers to Find the Linguistic Fingerprints of Reading Effort

<p align="center">
  A Research Project by <strong>Neuroquantix</strong>
</p>

## Abstract

Can the way we use language reveal the cognitive effort we’re expending? This project investigates whether it’s possible to distinguish between two different cognitive states—Normal Reading (NR) and Task-Specific Reading (TSR)—using only the linguistic properties of a text. Starting with classical machine learning models that achieved a baseline F1-score of ~0.74, we developed a novel hybrid approach. By combining hand-crafted syntactic features with powerful embeddings from a fine-tuned BERT transformer, our final MLP-based model successfully classified the reading state with an **F1-score of 0.9474**, demonstrating a strong, quantifiable link between language patterns and cognitive load.

---

## Interactive Demo

You can test the final model live on Hugging Face Spaces. Click the image below to open the demo in a new tab.


<a href="https://edaaaydinea-zuco-cognitive-analyzer.hf.space" target="_blank">
  <img src="https://neuroquantix.tech/wp-content/uploads/2025/07/image-4.png" alt="Interactive Demo Screenshot" width="850">
</a>
<p align="center"><em>Click the image to try the interactive demo!</em></p>

---

## Table of Contents

1.  [The Scientific Question](#the-scientific-question)
2.  [Our Approach](#our-approach)
    * [Baseline with Classical Models](#1-baseline-with-classical-models)
    * [Fine-Tuning BERT](#2-fine-tuning-bert-for-task-specific-embeddings)
    * [Hybrid Model Architecture](#3-the-hybrid-model)
3.  [Results](#results)
4.  [Technology Stack](#technology-stack)
5.  [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
6.  [Usage](#usage)
7.  [License](#license)
8.  [Contact](#contact)

---

## The Scientific Question

The human brain is not a static processor; it adapts its strategy based on the task at hand. When we read casually, our cognitive state is different from when we read to find a specific piece of information. This latter state, known as Task-Specific Reading (TSR), involves a higher degree of attention and cognitive effort.

Using the comprehensive [ZuCo 2.0 dataset](https://osf.io/2urht/), which uniquely pairs text with corresponding EEG and eye-tracking data, our research posed a challenging question: **Can we bypass the direct neural recordings and identify the cognitive state of the reader purely from the text they are processing?** In other words, does increased cognitive effort leave a detectable “fingerprint” on the linguistic and structural properties of language?

---

## Our Approach

### 1. Baseline with Classical Models

Our first phase involved a systematic evaluation of classical machine learning models (such as Logistic Regression, RandomForest, and SVM). We explored two distinct feature engineering paths:

* **LLM-based Embeddings:** We used a pre-trained sentence transformer (`all-MiniLM-L6-v2`) to create dense vector representations of each sentence.
* **Discrete Linguistic Features:** We engineered a rich set of 19 features, including readability scores (Flesch Reading Ease), lexical diversity (TTR), and advanced syntactic complexity metrics derived using **spaCy** (e.g., dependency distance, POS tag counts).

These initial models performed reasonably well, with the **RandomForest classifier** achieving the best baseline **F1-score of approximately 0.74** for the TSR class.

### 2. Fine-Tuning BERT for Task-Specific Embeddings

The breakthrough came from fine-tuning a `bert-base-uncased` model using the Hugging Face `transformers` library and **PyTorch**. This process, validated robustly using 5-fold Stratified Cross-Validation, transformed the general-purpose BERT into a specialist whose embeddings were highly discriminative for our specific cognitive state classification task.

### 3. The Hybrid Model

Our final and most successful model architecture was a hybrid one. We concatenated two feature sets:
1.  **Fine-Tuned Embeddings (768 dimensions):** Capturing the semantic and contextual essence.
2.  **Scaled Enhanced Discrete Features (19 dimensions):** Capturing explicit rules about syntax and readability.

This combined feature vector was then fed into an advanced **Multi-Layer Perceptron (MLP)**, architected in PyTorch with Dropout and BatchNorm to prevent overfitting.

---

## Results

The final hybrid model, evaluated on a held-out test set, achieved a remarkable **F1-score of 0.9474** for the TSR class.

This significant leap in performance provides a strong answer to our initial research question. Yes, cognitive effort leaves a quantifiable and highly predictive fingerprint in our use of language. This finding has exciting implications for developing non-invasive, text-based digital biomarkers to assess cognitive load, attention, or even mental fatigue.

---

## Technology Stack

* **Core Libraries:** Pandas, NumPy, Scikit-learn, NLTK, spaCy, Textstat
* **Deep Learning:** PyTorch, Hugging Face Transformers
* **Models:** Logistic Regression, RandomForest, SVM, LightGBM, MLP, BERT
* **Methodology:** Stratified K-Fold Cross-Validation, Fine-Tuning, Feature Engineering

---

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Python 3.9+
* pip

### Installation

1.  **Clone the repo**
    ```sh
    git clone https://github.com/neuroquantix/zuco-cognitive-analysis.git
    cd zuco-cognitive-analysis
    ```
2.  **Create and activate a virtual environment (recommended)**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install the required packages**
    ```sh
    pip install -r requirements.txt
    ```

---

## Usage

The complete methodology, code, and analysis are available for review in the project’s Jupyter Notebook.

To see the full analysis, data preprocessing, model training, and evaluation, please open and run the `Cognitive_State_Analysis.ipynb` notebook using Jupyter Lab or Jupyter Notebook.

```sh
jupyter lab Cognitive_State_Analysis.ipynb
