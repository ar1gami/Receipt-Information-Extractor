Put the LoRA adapter from the Kaggle run here, in a folder called lora:

    models/lora/adapter_config.json
    models/lora/adapter_model.safetensors
    models/lora/run_config.json
    (and the processor and tokenizer files that scripts/finetune.py saved next to them)

The Docker build copies this folder into the image, and the server reads the
adapter from /srv/models/lora. The weights are not committed to git (see
.gitignore). Keep a copy of the folder somewhere safe, because it is the result
of an hour of training.
