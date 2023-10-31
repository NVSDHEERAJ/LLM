import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from tqdm import tqdm
import numpy as np
from model import RecurrentTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

new_model = RecurrentTransformer()
new_model.load_state_dict(torch.load('trained_model.pt'))
#new_model.state_dict()
new_model = new_model.to(device)

enc = tiktoken.get_encoding("gpt2")
vocab_size = 50257

with torch.no_grad():
    # generate from the model
    while True:
        inp = str(input("Enter a Question\n"))
        if(inp == 'bye'):
            break
        enc_inp = torch.tensor(enc.encode(inp), dtype=torch.long)
        context = enc_inp.unsqueeze(0).to(device)
        print(enc.decode(new_model.generate(context, max_new_tokens=100)[0].tolist()))
