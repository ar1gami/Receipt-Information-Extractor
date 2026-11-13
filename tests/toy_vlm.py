"""Build a tiny, randomly initialized vision language model and processor offline.

This is not a useful model. It exists so the real Transformers and PEFT code
paths (loading, chat templates, LoRA training, saving, merging, generating) can
be exercised on a CPU in seconds without downloading anything. It uses the
real LLaVA classes with a handful of tiny layers and a character-level
tokenizer.
"""
import string
from pathlib import Path

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' %}<image>"
    "{% elif content['type'] == 'text' %}{{ content['text'] }}"
    "{% endif %}{% endfor %}"
    "{% if message['role'] == 'assistant' %}</s>{% else %}\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}ASSISTANT:{% endif %}"
)

SPECIALS = ["<pad>", "<unk>", "<s>", "</s>", "<image>"]

def build_toy_vlm(folder, seed: int = 0, init_std: float = 0.25) -> Path:
    """Write the tiny model and processor into folder and return it.

    init_std is the scale of the random weights. The default is much larger
    than the usual 0.02 on purpose. LoRA only adds small updates on top of
    frozen weights, and on a random model with tiny weights those updates
    barely change the output, so training looks stuck. Larger weights behave
    more like a pretrained model, whose frozen weights already carry signal.
    """
    import torch
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionConfig,
        LlamaConfig,
        LlavaConfig,
        LlavaForConditionalGeneration,
        LlavaProcessor,
        PreTrainedTokenizerFast,
    )

    folder = Path(folder)
    chars = [c for c in string.printable if c not in "\t\r\x0b\x0c"]
    vocab = {token: i for i, token in enumerate(SPECIALS + chars)}

    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    # One token per character, newlines included.
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        additional_special_tokens=["<image>"],
    )

    torch.manual_seed(seed)
    vision = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=32,
        patch_size=16,
        projection_dim=16,
        initializer_range=init_std,
    )
    text = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=1024,
        pad_token_id=vocab["<pad>"],
        bos_token_id=vocab["<s>"],
        eos_token_id=vocab["</s>"],
        initializer_range=init_std,
    )
    config = LlavaConfig(
        vision_config=vision,
        text_config=text,
        image_token_id=vocab["<image>"],
        image_token_index=vocab["<image>"],
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        initializer_range=init_std,
    )
    model = LlavaForConditionalGeneration(config)

    image_processor = CLIPImageProcessor(
        size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}
    )
    processor = LlavaProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        patch_size=16,
        vision_feature_select_strategy="default",
        num_additional_image_tokens=1,
        chat_template=CHAT_TEMPLATE,
    )
    model.save_pretrained(folder)
    processor.save_pretrained(folder)
    return folder

# SmolVLM style chat template (Idefics3 family): "User:<image>text<end_of_utterance>\nAssistant: answer<end_of_utterance>\n"
IDEFICS3_CHAT_TEMPLATE = (
    "<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}"
    "{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}"
    "{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}"
    "{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}"
    "<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
)

def build_toy_idefics3(folder, seed: int = 0, init_std: float = 0.25, vision_hidden: int = 32) -> Path:
    """A tiny random model built from the same classes as SmolVLM (Idefics3).

    SmolVLM splits each image into tiles plus a global view and expands every
    tile into a run of image tokens, so one receipt can become thousands of
    tokens. This toy keeps that structure at a tiny size, so the code that
    controls image size and masks the prompt is exercised on the real classes.

    vision_hidden is the image encoder's width. bitsandbytes quantization needs
    every linear layer's input width to be a multiple of 64 (SmolVLM's are), so
    the quantization tests build the toy with vision_hidden=64.
    """
    import torch
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import (
        Idefics3Config,
        Idefics3ForConditionalGeneration,
        Idefics3ImageProcessor,
        Idefics3Processor,
        Idefics3VisionConfig,
        LlamaConfig,
        PreTrainedTokenizerFast,
    )

    folder = Path(folder)
    chars = [c for c in string.printable if c not in "\t\r\x0b\x0c"]
    row_cols = [f"<row_{i + 1}_col_{j + 1}>" for i in range(6) for j in range(6)]
    specials = SPECIALS[:4] + ["<fake_token_around_image>", "<image>", "<end_of_utterance>", "<global-img>", "<|im_start|>"]
    vocab = {token: i for i, token in enumerate(specials + row_cols + chars)}

    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="<end_of_utterance>",
    )
    # Keep the long special strings whole instead of splitting them into characters.
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<fake_token_around_image>", "<image>", "<end_of_utterance>", "<global-img>", "<|im_start|>"] + row_cols}
    )
    
    torch.manual_seed(seed)
    vision = Idefics3VisionConfig(
        hidden_size=vision_hidden,
        intermediate_size=2 * vision_hidden,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=32,
        patch_size=8,
        initializer_range=init_std,
    )
    text = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        pad_token_id=vocab["<pad>"],
        bos_token_id=vocab["<s>"],
        eos_token_id=vocab["<end_of_utterance>"],
        initializer_range=init_std,
    )
    # 32 / 8 = 4 patches per side, 16 patches, shuffled by scale_factor 2 gives 4 tokens per tile.
    config = Idefics3Config(
        vision_config=vision,
        text_config=text,
        image_token_id=vocab["<image>"],
        scale_factor=2,
        initializer_range=init_std,
    )
    model = Idefics3ForConditionalGeneration(config)
    model.generation_config.eos_token_id = vocab["<end_of_utterance>"]
    model.generation_config.pad_token_id = vocab["<pad>"]

    image_processor = Idefics3ImageProcessor(
        size={"longest_edge": 64},
        max_image_size={"longest_edge": 32},
        do_image_splitting=True,
    )
    processor = Idefics3Processor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        image_seq_len=4,
        chat_template=IDEFICS3_CHAT_TEMPLATE,
    )
    model.save_pretrained(folder)
    processor.save_pretrained(folder)
    return folder
