# AMA4951 Capstone Project: Fake News Detection

**Student:** Chuk Chak Sum  
**Student ID:** 24035259D  
**Supervisor:** Prof. JIANG Binyan  

## Overview

This capstone project develops a **fake news detection system** using a hybrid stacking ensemble of classical machine learning and transformer-based models. The goal is to improve the reliability and accuracy of automated fake news classification by combining multiple diverse datasets and model strengths.

Fake news on digital platforms poses significant societal risks, including influencing public opinion, disrupting democratic processes, and undermining trust in science and public health. Automated detection is essential because manual fact-checking cannot keep pace with the volume and speed of online misinformation.

## Key Features

- **Multi-dataset integration**: Merged and cleaned ISOT, LIAR, and FakeNewsNet datasets (52,157 samples after deduplication)
- **Diverse model ensemble**: Combines surface-level, stylistic, semantic, and deep contextual features
- **Leakage-aware pipeline**: Strict train/validation/test splits, out-of-fold predictions, and split-aware preprocessing
- **Stacking architecture**: Five base models + Logistic Regression meta-learner
- **Exploration of LLMs**: Evaluated Grok API with web search for external truth scoring (ultimately excluded due to performance)

## Datasets

| Dataset       | Type                  | Role                                      |
|---------------|-----------------------|-------------------------------------------|
| ISOT          | Long-form articles    | Primary source of full news articles      |
| FakeNewsNet   | Long-form articles    | Additional real/fake news articles        |
| LIAR          | Short claims          | Short-form political statements           |

- **Labeling**: Binary (Fake = 0, Real = 1). For LIAR, `true` and `mostly-true` → 1; all others → 0.
- **Final size**: 52,157 rows after deduplication and cleaning.
- **Split**: 80% train / 10% validation / 10% test (stratified).

## Preprocessing Pipeline

- Combined title + text
- Removed source indicators (e.g., “WASHINGTON (Reuters)”) to prevent shortcut learning
- Lowercasing, stopword removal (for classical models only), whitespace normalization
- Punctuation preserved for transformer models
- All cleaning applied **independently** per data split to avoid leakage

## Models

### Base Models

1. **Logistic Regression + Word TF-IDF** (unigrams + bigrams)
2. **Linear SVM + Character TF-IDF** (n-grams 3–5)
3. **Random Forest + Word TF-IDF** (100 trees, max_depth=10)
4. **Logistic Regression + Sentence Embeddings** (all-MiniLM-L6-v2)
5. **Fine-tuned DistilBERT** (max length 256, 3 epochs, AdamW 2e-5)

### Stacking Ensemble

- Out-of-fold (5-fold) predictions from all base models used as meta-features
- Meta-learner: Logistic Regression
- Final inference: Base models retrained on full training set → meta-learner

## Results (Test Set)

| Model                                      | Macro F1 | Accuracy | AUC-ROC |
|--------------------------------------------|----------|----------|---------|
| Logistic Regression (Word TF-IDF)          | 0.90     | 0.90     | 0.9673  |
| Linear SVM (Char TF-IDF)                   | 0.89     | 0.90     | 0.9651  |
| Random Forest (Word TF-IDF)                | 0.89     | 0.89     | 0.9535  |
| Logistic Regression (Sentence Embeddings)  | 0.85     | 0.85     | 0.9327  |
| DistilBERT                                 | 0.91     | 0.91     | **0.9804** |
| **Stacking Ensemble**                      | **0.92** | **0.92** | 0.9753 |

- DistilBERT was the strongest individual model.
- The stacking ensemble achieved the best overall F1 and accuracy by combining complementary signals.

