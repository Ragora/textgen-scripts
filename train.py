import multiprocessing

from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import GPT2ConfigCPU, build_gpt2_config
from aitextgen import aitextgen

# pip install -U deepspeed

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # The name of the downloaded Shakespeare text for training
    file_name = "pg100.txt"

    # Train a custom BPE Tokenizer on the downloaded text
    # This will save one file: `aitextgen.tokenizer.json`, which contains the
    # information needed to rebuild the tokenizer.
    train_tokenizer(file_name)
    tokenizer_file = "aitextgen.tokenizer.json"

    # GPT2ConfigCPU is a mini variant of GPT-2 optimized for CPU-training
    # e.g. the # of input tokens here is 64 vs. 1024 for base GPT-2.
    #config = GPT2ConfigCPU()
    config = build_gpt2_config(
        vocab_size = 10000,
        bos_token_id = 0,
        eos_token_id = 0,
        max_length = 1024,
        dropout = 0.0)

    # Instantiate aitextgen using the created tokenizer and config

    #ai = aitextgen(tf_gpt2="124M", to_gpu=False, tokenizer_file=tokenizer_file, config=config)
    ai = aitextgen(to_gpu=False, tokenizer_file=tokenizer_file, config=config)

    # You can build datasets for training by creating TokenDatasets,
    # which automatically processes the dataset with the appropriate size.
    data = TokenDataset(file_name, tokenizer_file=tokenizer_file, block_size=64)

    # Train the model! It will save pytorch_model.bin periodically and after completion to the `trained_model` folder.
    # On a 2020 8-core iMac, this took ~25 minutes to run.
    #ai.train(file_name, use_deepspeed=True, n_gpu=1, batch_size=41, num_workers=8, from_cache=False, fp16=True, learning_rate=1e-3, num_steps=3000, generate_every=1000, save_every=1000, line_by_line=False)

    # CPU Training
    ai.train(data, use_deepspeed=False, n_gpu=0, batch_size=41, num_workers=8, from_cache=False, fp16=False, learning_rate=1e-3, num_steps=3000, generate_every=1000, save_every=1000, line_by_line=False)

    # num_workers=8,
    # Generate text from it!
   # ai.generate(10, prompt="ROMEO:")

    # With your trained model, you can reload the model at any time by
    # providing the folder containing the pytorch_model.bin model weights + the config, and providing the tokenizer.
   # ai2 = aitextgen(model_folder="trained_model",
    #                tokenizer_file="aitextgen.tokenizer.json")

    #ai2.generate(10, prompt="ROMEO:")