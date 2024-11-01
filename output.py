# Import necessary libraries
import torch
from torch import nn
from torch.nn import functional as F

from decoder import Attention, FeedForward, RMSNorm, RoPE, TransformerBlock, repeat_kv
from model import ModelArgs


## Step3: The Output Block
# This is the Llama 3 model. Again, the class name is maintained as Transformer to match with Meta Llama 3 model.

class Transformer(nn.Module):
  def __init__(self, params: ModelArgs, vocab_size: int):
    super().__init__()
    # set all the ModelArgs in params variable
    self.params = params
    self.vocab_size = vocab_size
    # Initilizate embedding class from the input block
    self.tok_embeddings = nn.Embedding(vocab_size, params.dim)

    # Initialize the decoder block and store it inside the ModuleList. 
    # This is because we've 4 decoder blocks in our Llama 3 model. (Official Llama 3 has 32 blocks)
    self.layers = nn.ModuleList()
    for layer_id in range(params.n_layers):
      self.layers.append(TransformerBlock(args=params))

    # Initilizate RMSNorm for the output block
    self.norm = RMSNorm(params.dim, eps = params.norm_eps)
    
    # Initilizate linear layer at the output block.
    self.output = nn.Linear(params.dim, vocab_size, bias=False)

  def forward(self, x, start_pos=0, targets=None):
    
    # start_pos = token position for inference mode, inference = True for inference and False for training mode
    # x is the batch of token_ids generated from the texts or prompts using tokenizers.
    # x[bsz, seq_len] -> h[bsz, seq_len, dim]
    h = self.tok_embeddings(x)

    # If the target is none, Inference mode is activated and set to "True" and "False" if Training mode is activated.
    if targets is None:
      inference = True
    else:
      inference = False

    # The embeddings (h) will then pass though all the decoder blocks.
    for layer in self.layers:
      h = layer(h, start_pos, inference)


    # The output from the final decoder block will feed into the RMSNorm
    h = self.norm(h)
    # h = RMSNorm(self.params.dim, eps = self.params.norm_eps).forward(h)

    # After normalized, the embedding h will then feed into the Linear layer. 
    # The main task of the Linear layer is to generate logits that maps the embeddings with the vocabulary size.
    # h[bsz, seq_len, dim] -> logits[bsz, seq_len, vocab_size]
    logits = self.output(h).float()
    loss = None

    # Inference mode is activated if the targets is not available
    if targets is None:
      loss = None
    # Training mode is activated if the targets are available. And Loss will be calculated for further model training. 
    else:
      loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

    return logits, loss