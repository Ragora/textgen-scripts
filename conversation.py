import multiprocessing

from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import GPT2ConfigCPU
from aitextgen import aitextgen

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
    config = GPT2ConfigCPU()

    max_length = 128
    min_length = 64
    lstrip = True
    temperature = 0.25
    seed = 69

    message_count = 32

    # Instantiate aitextgen using the created tokenizer and config
    ai = aitextgen(tokenizer_file=tokenizer_file, config=config)

    # You can build datasets for training by creating TokenDatasets,
    # which automatically processes the dataset with the appropriate size.
    data = TokenDataset(file_name, tokenizer_file=tokenizer_file, block_size=64)


    # With your trained model, you can reload the model at any time by
    # providing the folder containing the pytorch_model.bin model weights + the config, and providing the tokenizer.
    ai2 = aitextgen(model_folder="trained_model",
                    tokenizer_file="aitextgen.tokenizer.json")

   # current_text = "ROMEO:"
    current_text = ""
    for _ in range(message_count):
        next_text_options = ai.generate(1, seed=seed, return_as_list=True, max_length=max_length, temperature=temperature, lstrip=lstrip, prompt="")
        next_text = next_text_options[0]

        # Next is a completion of the previous
        #print(next_text)
        current_text += next_text + "\n\n"


    print(current_text)

    # Train the model! It will save pytorch_model.bin periodically and after completion to the `trained_model` folder.
    # On a 2020 8-core iMac, this took ~25 minutes to run.
    #ai.train(data, batch_size=4, num_workers=8, num_steps=50000, generate_every=5000, save_every=5000)
    