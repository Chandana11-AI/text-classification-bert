# Document Classification with BERT

An NLP experiment for assigning text documents to categories by fine-tuning a pretrained BERT sequence-classification model.

The public repository contains a Colab-exported experimental source file. It demonstrates the model-development workflow, but it is not a packaged application or a clean, reproducible command-line release.

## Problem

Document collections often need consistent categories before they can support search, routing or downstream analysis. This project explores contextual transformer representations for multiclass document classification.

My related Mphasis internship work included noisy OCR-extracted documents, preprocessing, BERT fine-tuning and retrieval workflows. The source published here covers the **text-classification portion**. It does not include a Tesseract/OCR implementation or the separate FAISS/LSH retrieval pipeline.

## Method

1. Read document text and category information from a downloaded document dataset.
2. Build category-to-index and index-to-category mappings.
3. Shuffle and partition the documents into train, validation and test subsets.
4. Explore merging additional JSONL training and validation material.
5. Tokenize with `BertTokenizerFast` using `bert-base-uncased`, truncation and padding.
6. Initialize `BertForSequenceClassification` with the number of target labels.
7. Fine-tune using Hugging Face `Trainer`.
8. Evaluate predictions and save the model/tokenizer for subsequent inference.

### Training settings in the source

| Setting | Recorded configuration |
|---|---|
| Base checkpoint | `bert-base-uncased` |
| Epochs | 3 |
| Training batch size | 16 |
| Evaluation batch size | 32 |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Evaluation/logging interval | 50 steps |
| Mixed precision | `fp16=True`; requires a compatible environment |

These are source settings, not a claim that this exact revision has been rerun successfully.

## Repository guide

| File | Contents |
|---|---|
| [src](src) | Notebook-style Python export containing data loading, tokenization, training, evaluation and prediction examples |
| [data](data) | Legacy dataset note; its link differs from the download URL in the source |
| [README.md](README.md) | Project explanation and current reproducibility status |

Despite their names, `src` and `data` are files, not directories. The repository does not currently contain `requirements.txt`, `run_classifier.py`, a trained checkpoint or a standalone inference service.

## How to explore it

Read [src](src) as a Colab experiment and reconstruct its cells in a notebook environment. The source imports PyTorch, Transformers, pandas, NumPy, scikit-learn, NLTK, datasets/evaluate and plotting libraries. It also contains notebook shell commands.

Before rerunning:

- Resolve the dataset mismatch: the source downloads Jensen Baxter's 10-category document collection, while the legacy note points to a different news dataset.
- Supply or remove the external `combined_train_unprompted.json` and `combined_val_unprompted.json` inputs; they are not committed.
- Correct the installation typo and indentation issues in the exported cells.
- Align labels with the merged text rows. The current source takes text from merged frames but labels from the original splits.
- Make the split reproducible and verify duplicates and label mappings.
- Make prediction device selection consistent; the example prediction function explicitly uses CUDA.

## Evaluation and results

The experiment includes an accuracy callback and train/validation/test evaluation calls. This public snapshot does not contain the experiment records needed to substantiate the earlier README's 94% accuracy or sub-300 ms latency claims, so those figures are not presented as verified results.

For a reproducible evaluation, record per-class precision/recall/F1, a confusion matrix, test-set provenance, tokenizer settings and hardware-specific inference timing.

## What this project demonstrates

- Applying pretrained transformers to supervised NLP.
- Label mapping, tokenization and custom PyTorch dataset construction.
- Configuring a Hugging Face training workflow.
- Saving a classifier/tokenizer and using a text-classification pipeline.
- Recognizing the difference between fine-tuning a pretrained BERT model and training a foundation model from scratch.

## Next steps

Package the notebook into training/inference modules, add dependency versions, fix the data/label alignment, publish a sanitized sample and record a fresh evaluation. OCR integration, retrieval and an API would be separate extensions.
