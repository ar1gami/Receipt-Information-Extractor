# Receipt extractor

Reads a photo of a receipt and returns the vendor, the date and the total as JSON.

Two approaches are built and measured on the same held-out receipts. The baseline runs Tesseract OCR and pulls the fields out with hand-written rules. The second approach fine-tunes a small open vision-language model with LoRA so that it reads the image and writes the JSON directly. The comparison covers accuracy per field, latency and model size, including a quantized version of the fine-tuned model. Either approach can be served through a FastAPI service.

## Results

Test set: 347 receipts from the SROIE test split (the "SROIE datasetv2" copy on Kaggle), dates read day first. The fine-tuned models were trained on the 626 SROIE training receipts, 10 percent of which were held out for validation. With 347 receipts, each accuracy has an uncertainty of roughly plus or minus 5 points, so differences of one or two points are not meaningful.

| model | vendor | date | total | all fields | parse failures | median ms | p95 ms | size MB |
|---|---|---|---|---|---|---|---|---|
| OCR and rules | 46.7 | 72.3 | 64.0 | 29.4 | not recorded | not measured | not measured | not applicable |
| vision-language model, no fine-tuning | 27.1 | 55.6 | 18.4 | 4.6 | 15.6% | 3303 | 5764 | 1015 |
| vision-language model, fine-tuned | 81.6 | 91.1 | 88.2 | 67.4 | 0.0% | 2134 | 2339 | 1015 |
| fine-tuned, int8 | 80.1 | 91.6 | 88.5 | 66.6 | 0.0% | 5944 | 6454 | 606 |
| fine-tuned, int4 | 75.2 | 88.8 | 81.8 | 55.6 | 0.0% | 2723 | 2995 | 424 |

Setup: SmolVLM-500M-Instruct, LoRA rank 16 on the language model with the image encoder frozen (8.9 million trainable parameters, 1.7 percent), 3 epochs, learning rate 2e-4, effective batch size 8, float32 training, images resized to a longest side of 1536 pixels. Training took 3681 seconds (about 61 minutes) on one Tesla T4 on Kaggle. Validation loss fell from 0.091 to 0.078 to 0.072 over the three epochs. Evaluation used float16 on the same T4, and latency is the median over 50 receipts. OCR latency was not measured, so no speed comparison with OCR is made. The OCR row was scored on a different machine from the model rows.

What the table shows:

- Fine-tuning is what makes the model usable. Without it the model fails to produce valid JSON on 15.6 percent of receipts and gets all three fields right on 4.6 percent. After fine-tuning it never fails to parse and gets all three right on 67.4 percent, against 29.4 percent for OCR with rules.
- int8 shrinks the model by 40 percent (1015 to 606 MB) for a change in accuracy that is within the noise. It was also 2.8 times slower than float16 on this GPU (5.9 s against 2.1 s), so on a T4 quantization saved memory and cost speed.
- int4 shrinks the model by 58 percent (to 424 MB) and loses about 12 points of all-fields accuracy, and it was 28 percent slower than float16.
- Merging the adapter into the base model changed nothing: on 20 receipts the scores were identical before and after.

The table is produced by `scripts/compare.py` from the reports that `scripts/evaluate.py` saves.

### Seen and unseen vendors

73 percent of the test receipts (252 of 347) come from a vendor that also appears in the training set, because SROIE contains many receipts from the same shops. Accuracy depends heavily on this.

| fine-tuned model | receipts | vendor | all fields |
|---|---|---|---|
| vendor appears in training | 252 | 93.3 | 79.0 |
| vendor not in training | 95 | 50.5 | 36.8 |

So part of the vendor score is the model having learned shop names it saw in training. On receipts from shops it has not seen, all-fields accuracy is 36.8 percent. Nine test receipts also have exactly the same vendor, date and total as a training receipt (probably the same receipt scanned twice). Removing them changes the fine-tuned all-fields score from 67.4 to 66.6 percent.

### Error analysis

The 136 wrong fields of the fine-tuned model (64 vendor, 41 total, 31 date) were grouped with simple rules that compare each answer with the label. I did not check each one against its image, so the groups are indicative.

- Vendor: 35 are small spelling slips such as "PETROK" for "PETRON", 8 are names cut short or with address text added, 5 are heavily garbled, and 16 are a different name from the label, often the brand on the sign instead of the registered company name.
- Total: 25 are a different amount from the label (13 smaller, 12 larger), which fits the model picking a subtotal, cash tendered or change line. 8 differ in one digit, 1 has two digits swapped, and 7 are off by a few cents, which may be rounding in the label.
- Date: the day or month is wrong in 19 cases, the year in 4, 2 are a different date, 1 has day and month swapped, and 5 are not valid dates (for example a time of day or a cut-off year).
- The int4 model has more errors in every field, with vendor names getting worse the most (86 wrong against 64).

What I would try next: more training data with a wider range of vendors (the unseen-vendor gap is the largest weakness), a check that the extracted total is consistent with the other amounts on the receipt, a higher image resolution for small print, and reading some of the label disagreements against the images to see how many are label errors.

## How it is evaluated

A prediction is scored per field by exact match after normalizing both sides. Vendor names are lowercased and stripped of punctuation. Totals become numbers with two decimals, so "$7.45", "7.45" and "RM7.45" are the same. Dates become ISO dates. A numeric date such as 05/03/2015 is ambiguous, so it is read month first unless `--day-first` is passed, and a date such as 14/03/2015 can only be day first, so it is read that way either way. A prediction the model never produced as valid JSON counts as wrong for every field and is also reported as a parse failure.

The test receipts are not used for training, for choosing settings or for adjusting the OCR rules. Rules and hyperparameters were chosen on the training data and validation split only, and each model in the results table was scored once on the full test set. Two small exceptions: about 20 test receipts were looked at early in the project before the OCR rules were written (the rules were then tuned on training receipts only), and a 20 receipt check of the merged model and a 10 receipt check of a tiny training run used test receipts, so those few are not untouched.

## Layout

```
receipts/     the code: normalization, metrics, OCR baseline, model loading, training loop
scripts/      command line tools (listed below)
app/          the FastAPI service
models/       where the LoRA adapter goes for the Docker build (the weights are not committed)
deploy/       text for the adapter's page on the Hugging Face hub
notebooks/    the Kaggle notebook for the GPU steps
tests/        the test suite
```

The command line tools:

- `scripts/prepare_sroie.py` converts a SROIE style download (one image and one JSON label file per receipt) into this project's format.
- `scripts/split_dataset.py` splits one data file into train.jsonl and test.jsonl.
- `scripts/make_synthetic.py` writes a small set of synthetic receipts, used for smoke tests only.
- `scripts/evaluate.py` scores the OCR baseline or a model on a data file, and can measure latency and model size.
- `scripts/finetune.py` fine-tunes a vision-language model with LoRA.
- `scripts/merge_adapter.py` merges the LoRA weights back into the base model.
- `scripts/compare.py` builds the results table from saved reports.
- `scripts/error_report.py` builds a sheet of failures to sort by hand, then counts what you wrote.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
sudo apt-get install tesseract-ocr      # or: brew install tesseract
python -m pytest
```

For fine-tuning on a GPU machine, install `requirements-train.txt` as well. On Colab or Kaggle, PyTorch is already installed, so install the other packages in that file only.

## Data format

One JSON object per line. Image paths are relative to the folder holding the file.

```
{"image": "images/001.jpg", "vendor": "Tim Hortons", "date": "2026-10-12", "total": "7.45"}
```

To try the pipeline without real data:

```
python scripts/make_synthetic.py data/synthetic 20
python scripts/evaluate.py data/synthetic/data.jsonl --predictor ocr
```

Synthetic receipts are clean rendered text, so they say nothing about real accuracy. They exist to check that the code runs.

## Running the baseline

```
python scripts/evaluate.py data/test.jsonl --predictor ocr --latency --out reports/ocr.json
```

## Fine-tuning

Fine-tuning needs a GPU. Try a very small run first and check that the loss goes down:

```
python scripts/finetune.py data/train.jsonl --model-id <hub id> --max-examples 20 --epochs 1
```

Then the full run, and evaluation on the test set:

```
python scripts/finetune.py data/train.jsonl --model-id <hub id> --output outputs/lora --epochs 2 --wandb
python scripts/evaluate.py data/test.jsonl --predictor vlm --model-id <hub id> --adapter outputs/lora --latency --out reports/tuned.json
```

SmolVLM style models cut each image into 512 pixel tiles and spend 64 tokens on each, so the size of the longest image side decides memory and time. `--image-longest-edge 1536` sets it (the log prints the model's own default when the flag is left out), and `--no-image-splitting` turns the tiles off, which is too coarse for receipts. If the GPU runs out of memory, `--freeze-vision` trains only the language model and `--gradient-checkpointing` trades speed for memory. The image size is saved with the adapter, and `evaluate.py`, `merge_adapter.py` and the API use it automatically when the flag is left out, so training and inference cannot silently disagree. Use the same image size for every row of the results table.

Latency is timed on the first 50 receipts by default (`--latency-images N` changes that), because timing repeats every prediction. Scores are saved before timing starts, and Ctrl+C during timing skips it without losing them.

To try a model on just a few receipts, add `--limit 5`. To read what it actually wrote, add `--show-raw 5`. When a score is unexpectedly low, this is usually where the reason shows up.

## Running the GPU steps on Kaggle

`notebooks/kaggle_finetune.ipynb` runs the whole fine-tuning part on a free Kaggle GPU: a check of the untouched model, a tiny training run, the full fine-tune, a merge, evaluation of the untouched, fine-tuned, 8-bit and 4-bit models, and the results table.

1. Zip the project folder without its data folder, and upload the zip as a private Kaggle dataset.
2. Create a notebook, upload kaggle_finetune.ipynb into it (File, Import Notebook), and attach two inputs: the zip dataset and the SROIE dataset ("SROIE datasetv2" by urbikn).
3. In the notebook settings choose the GPU T4 x2 accelerator and turn Internet on (it needs a phone-verified account).
4. Run all cells once. This does the checks only. Read each output.
5. Set DO_FULL_RUN to True, then Save Version and Save & Run All (Commit). Download receipt_results.zip from the Output tab when it finishes.

The notebook was run on a tiny random model standing in for SmolVLM, on a fake Kaggle folder layout, which tests the paths and commands. It has not been run on Kaggle itself or with the real model.

## Quantization

Merge the adapter into the base model, then load the merged model quantized and evaluate it the same way as everything else:

```
python scripts/merge_adapter.py --model-id <hub id> --adapter outputs/lora --output outputs/merged
python scripts/evaluate.py data/test.jsonl --predictor vlm --model-id outputs/merged --quantization 8bit --latency --out reports/int8.json
```

`8bit` and `4bit` use bitsandbytes and are meant for a CUDA GPU (the tests also run them on a CPU with a tiny model, where recent bitsandbytes supports it). `dynamic-int8` quantizes the linear layers to int8 on a CPU, which matters when the model will be served without a GPU. It cannot be combined with an adapter, so merge first. It uses PyTorch's `quantize_dynamic`, which PyTorch has marked as deprecated, and falls back to torchao (`pip install torchao`) if a newer PyTorch removes it. The saved report records which one was used, and the numbers from the two can differ, so say which you used when you report results.

## Comparing and error analysis

```
python scripts/compare.py ocr=reports/ocr.json tuned=reports/tuned.json int8=reports/int8.json
python scripts/error_report.py make data/test.jsonl reports/tuned.json --out reports/errors.csv
```

Open the CSV, look at each failed receipt, and write a short category in the last column (blurry, rotated, two totals, and so on). Then:

```
python scripts/error_report.py summarize reports/errors.csv
```

## Serving

```
uvicorn app.main:app --port 8080
curl -F "file=@receipt.jpg" http://localhost:8080/extract
```

The service uses the OCR baseline by default. To serve a model, set environment variables before starting it:

| variable | meaning |
|---|---|
| PREDICTOR | `ocr` (default) or `vlm` |
| MODEL_ID | hub id or local folder, required for `vlm` |
| ADAPTER_PATH | folder on this machine holding the LoRA adapter (it must contain adapter_config.json), optional. A hub id does not work, because the image size of the training run is read from that folder |
| QUANTIZATION | `8bit`, `4bit` or `dynamic-int8`, optional |
| DTYPE | `auto` (default), `bfloat16`, `float16` or `float32` |
| IMAGE_LONGEST_EDGE | longest image side in pixels, SmolVLM style models, optional (an adapter's training value is used when unset) |
| IMAGE_SPLITTING | `on` or `off`, SmolVLM style models, optional |
| MAX_UPLOAD_MB | largest accepted upload, default 10 |

The model is loaded once when the service starts. `/health` reports which predictor is loaded. A file that is not an image gets a 400, an oversized upload a 413, and a model that produced no usable JSON a 422.

## Docker

One Dockerfile builds three images. Run the commands from the project folder.

The OCR baseline (small):

```
docker build -t receipt-extractor .
docker run --rm -p 8080:8080 receipt-extractor
```

The fine-tuned model on a CPU. First copy the adapter folder from the Kaggle run (the one with adapter_config.json in it) to `models/lora`, then:

```
docker build --build-arg VARIANT=vlm -t receipt-extractor-vlm .
docker run --rm -p 8080:8080 receipt-extractor-vlm
```

This installs CPU-only PyTorch and downloads the base model (about 1 GB) from the Hugging Face hub when the container starts. To download it while building instead, add `--build-arg BASE_MODEL=HuggingFaceTB/SmolVLM-500M-Instruct`. The image is then about 1 GB larger and starts without network access. The build also loads the model and the adapter once, the same way the server does at start up, so a wrong adapter folder fails the build and not the deployment.

To build without a local adapter folder, put the adapter on the Hugging Face hub as a public model repo and pass it as a build argument. The download replaces `models/lora` inside the image:

```
docker build --build-arg VARIANT=vlm --build-arg BASE_MODEL=HuggingFaceTB/SmolVLM-500M-Instruct --build-arg ADAPTER_REPO=<user>/receipt-extractor-lora -t receipt-extractor-vlm .
```

`--build-arg ADAPTER_REVISION=<commit hash>` pins the download to one version of that repo. `deploy/model_card.md` is the text for the model repo's page.

Try it at http://localhost:8080/docs, or from PowerShell:

```
curl.exe -F "file=@receipt.jpg" http://localhost:8080/extract
```

The response includes `latency_ms`, the time the prediction took.

Things to know:

- The image is CPU only, so the model is much slower than the 2.1 second median in the results table, which was measured on a GPU. In a container on one Windows laptop, the fine-tuned model took about 18 seconds per receipt: the same receipt sent three times took 18.7, 18.4 and 18.2 seconds. The CPU model and the number of cores given to Docker were not recorded, so treat this as one data point. The `latency_ms` in each response is the real number for your machine.
- On a CPU the model runs in float32, and the results table was measured in float16 on a GPU, so a few answers can differ from the table.
- The model is 1015 MB in float16 and about twice that in memory in float32. Memory use of the container was not measured, so give Docker at least 6 GB.
- Without `BASE_MODEL`, every new container downloads the base model again. Either build with `BASE_MODEL`, or keep the download between runs with `-v hf-cache:/opt/hf`.
- The image sets `PREDICTOR`, `MODEL_ID` and `ADAPTER_PATH` for you (`/srv/models/lora`, where `models/lora` is copied to). Any of them can be changed with `docker run -e`. To try another adapter without rebuilding, mount it: `docker run -v <folder>:/srv/models/lora:ro ...`.
- If `ADAPTER_PATH` does not point at an adapter folder, the server stops at start up and says so. It never silently serves the untuned model.
- Model weights are not committed to git. See `models/README.md`.

## Tests

```
python -m pytest
```

The tests cover normalization and scoring on many edge cases, the OCR rules, parsing of model output, the data tools, the API, and the Docker and CI files. The Dockerfile cannot be built in the tests, so its shell steps are run with stand-in commands, and its model loading step is run for real against a tiny model. When PyTorch, Transformers and PEFT are installed, they also fine-tune, save, merge, quantize, reload and serve tiny random models on a CPU (one built like LLaVA, one built from the classes SmolVLM uses), and run the Kaggle notebook on a fake Kaggle folder layout, which checks the model code and the notebook against the installed library versions. Those tests say nothing about how well a real model reads receipts. That is what the evaluation scripts are for.

## Limitations

- Only three fields are extracted, and the code assumes one receipt per image.
- Receipts often show more than one amount (subtotal, cash tendered, amount due). Which one counts as the total is a judgment, and the two approaches can disagree on it.
- Tesseract is run with its English data, so receipts in other languages or scripts will do worse.
- Numeric dates are ambiguous between day first and month first. The dataset's convention has to be passed to the evaluation.
- 73 percent of the test receipts come from vendors that are also in the training set, and accuracy on unseen vendors is much lower (see Seen and unseen vendors above). Nine test receipts duplicate a training receipt.
- Some labels may not match the printed receipt, for example a total recorded before rounding. This was not checked against the images.
- Latency was measured on a single Tesla T4 with bitsandbytes quantization. It will differ on other hardware, and quantization may be faster elsewhere.
- Receipts contain personal information. Keep receipt images out of version control.
