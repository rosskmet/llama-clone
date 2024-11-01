# Import necessary libraries
import torch
from torch import nn
from torch.nn import functional as F



### Step 1: Input Block ###

# Using Tiny Shakespeare dataset for character-level tokenizer. Some part of the following character-level tokenizer is referenced from Andrej karpathy's GitHub (https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py) which I found is explained very well.
# Load tiny_shakespeare data file (https://github.com/tamangmilan/llama3/blob/main/tiny_shakespeare.txt)

class InputBlock():

    # device: str = 'cuda' if torch.cuda.is_available() else 'cpu'   # Assign device to cuda or cpu based on availability

    def __init__(self, prompts: str):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'   # Assign device to cuda or cpu based on availability
        self.data, self.vocab = self.prepareVocab()
        self.itos, self.stoi = self.createMapping(self.vocab)
        self.token_bos, self.token_eos, self.token_pad = self.defineTensors()
        self.encoded_tokens = self.encode(prompts)
        self.decoded_text = self.decode(self.encoded_tokens)

    def prepareVocab(self):

        # Load tiny_shakespeare data file.
        with open('raw_data/tiny_shakespeare.txt', 'r') as f:
            data = f.read()

        # Prepare vocabulary by taking all the unique characters from the tiny_shakespeare data
        vocab = sorted(list(set(data)))

        # Training Llama 3 model requires addtional tokens such as <|begin_of_text|>, <|end_of_text|> and <|pad_id|>, we'll add them into vocabulary
        vocab.extend(['<|begin_of_text|>','<|end_of_text|>','<|pad_id|>'])
        vocab_size = len(vocab)

        return data, vocab

    def createMapping(self, vocab):
        # Create a mapping between characters with corresponding integer indexes in vocabulary.
        # This is important to build tokenizers encode and decode functions.
        itos = {i:ch for i, ch in enumerate(vocab)}
        stoi = {ch:i for i, ch in enumerate(vocab)}

        return itos, stoi

    # Tokenizers encode function: take a string, output a list of integers
    def encode(self, s):
        return [self.stoi[ch] for ch in s]

    # Tokenizers decode function: take a list of integers, output a string
    def decode(self, l):
        return ''.join(self.itos[i] for i in l)

    def defineTensors(self):
    # Define tensor token variable to be used later during model training
        token_bos = torch.tensor([self.stoi['<|begin_of_text|>']], dtype=torch.int, device=self.device)
        token_eos = torch.tensor([self.stoi['<|end_of_text|>']], dtype=torch.int, device=self.device)
        token_pad = torch.tensor([self.stoi['<|pad_id|>']], dtype=torch.int, device=self.device)

        return token_bos, token_eos, token_pad

# if __name__ == '__main__':

#     prompts = "Hello World"
#     ib = InputBlock(prompts)

    ### Test: Input Block Code ###
    # You need take out the triple quotes below to perform testing

    # print(f"Lenth of shakespeare in character: {len(ib.data)}")
    # print(f"The vocabulary looks like this: {''.join(ib.vocab)}\n")
    # print(f"Vocab size: {len(ib.vocab)}")
    # print(f"encoded_tokens: {ib.encoded_tokens}")
    # print(f"decoded_text: {ib.decoded_text}")

    ### Test Results: ###
    """
    Lenth of shakespeare in character: 1115394
    The vocabulary looks like this: 
    !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz<|begin_of_text|><|end_of_text|><|pad_id|>

    Vocab size: 68
    encoded_tokens: [20, 43, 50, 50, 53, 1, 35, 53, 56, 50, 42]
    decoded_text: Hello World
    """