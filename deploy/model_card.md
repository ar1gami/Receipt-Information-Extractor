---
base_model: HuggingFaceTB/SmolVLM-500M-Instruct
library_name: peft
pipeline_tag: image-text-to-text
tags:
  - lora
  - receipts
  - information-extraction
---

# receipt-extractor-lora

A LoRA adapter for HuggingFaceTB/SmolVLM-500M-Instruct. Given a photo of a receipt, it writes the vendor, the date and the total as JSON.

It was trained for a portfolio project. The code, the evaluation and a FastAPI service that loads this adapter are in the GitHub repository: https://github.com/ar1gami/Receipt-Information-Extractor

## Training

The adapter was trained on 563 of the 626 SROIE training receipts, with the other 63 held out for validation. LoRA rank 16, alpha 32, dropout 0.05, applied to the language model with the image encoder frozen (8.9 million trainable parameters, 1.7 percent of the model). Three epochs (213 steps), learning rate 2e-4, effective batch size 8, float32, images resized to a longest side of 1536 pixels. Training took 3681 seconds on one Tesla T4.

## Results

Scored by exact match after normalizing, on the 347 receipts of the SROIE test split, with dates read day first.

| field | accuracy |
|---|---|
| vendor | 81.6% |
| date | 91.1% |
| total | 88.2% |
| all three | 67.4% |

On the 95 test receipts from shops that do not appear in the training set, all three fields were right on 36.8 percent. With 347 receipts each figure has an uncertainty of roughly plus or minus 5 points. The repository README has the full comparison with an OCR baseline, the quantized versions and an error analysis.

## Using it

This folder also holds run_config.json, which records the image size the adapter was trained with. The loader in the GitHub repository reads it and applies it. If you load the adapter with PEFT directly, set the longest image side to 1536 to match the training, or the results will not match the ones above.

## Limits

Only the vendor, the date and the total are extracted, and the code assumes one receipt per image. The model is much weaker on shops it did not see in training. The SROIE images are not included here, and their terms should be checked before the training data is reused.
