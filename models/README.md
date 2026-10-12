# Model Architecture & Checkpoints

This directory is designated for local checkpoints and configuration artifacts produced during fine-tuning.

## Supported Architectures
1. **Lexicon & Heuristic Baselines**:
   - Transparent Hinglish/English polar dictionary with polarity scores and negation flip rules.
2. **Classical Statistical Baselines**:
   - TF-IDF with Logistic Regression (`C=1.0`, `class_weight='balanced'`)
   - TF-IDF with Linear Support Vector Classifier (`LinearSVC`)
3. **Multilingual Pretrained Transformers**:
   - **mBERT**: `bert-base-multilingual-cased`
   - **XLM-RoBERTa**: `xlm-roberta-base`
   - **MuRIL**: `google/muril-base-cased` (Multilingual Representations for Indian Languages)

## Lazy Loading Principle
To preserve bandwidth, GPU/RAM resources, and user startup latency, models are loaded strictly on demand when invoked in training or inference sessions. No model checkpoints or large binary weights are downloaded or committed without user initiation.
