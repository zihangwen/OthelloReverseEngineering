# %%

import plotly.express as px
import torch
from tqdm import tqdm
from transformers.utils import to_numpy
import torch
from datasets import load_dataset

import torch as t
import numpy as np
import einops
import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.utils import to_numpy
from torch import Tensor
from IPython.display import HTML, display
from jaxtyping import Bool, Float, Int
import neuron_simulation.neel_utils as neel_utils

from circuits.othello_engine_utils import OthelloBoardState, to_string
from circuits.othello_utils import board_to_occupied_64


DEFAULT_DTYPE = t.int16

# device = "cuda" if t.cuda.is_available() else "cpu"
device = 'cpu'
t.set_grad_enabled(False)

print(f"Using device: {device}")


# %%

model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
dataset_size = 50
custom_functions = [
    othello_utils.games_batch_to_input_tokens_flipped_bs_classifier_input_BLC,
]
model = utils.get_model(model_name, device)

n_inputs = dataset_size
split = "train"
max_str_length = 59


dataset = load_dataset("adamkarvonen/othello_45MB_games", streaming=False)
encoded_othello_inputs_bL = []
decoded_othello_inputs_bL = []
for i, example in enumerate(dataset[split]):
    if i >= n_inputs:
        break
    encoded_input = example["tokens"][:max_str_length]
    decoded_input = to_string(encoded_input)
    encoded_othello_inputs_bL.append(encoded_input)
    decoded_othello_inputs_bL.append(decoded_input)

data = {}
data["encoded_inputs"] = encoded_othello_inputs_bL
data["decoded_inputs"] = decoded_othello_inputs_bL

# %%
batch_str_moves = decoded_othello_inputs_bL
iterable = batch_str_moves

game_stack = []
game = batch_str_moves[0]
if isinstance(game, t.Tensor):
    game = game.flatten()

board = OthelloBoardState()
states = []

# for i, move in enumerate(game):
i = 0
move = game[i]

state = t.zeros(64 + 64 + 5, dtype=DEFAULT_DTYPE)
if move >= 0:
    if move > 63:
        raise ValueError(f"Move {move} is out of bounds")
    state[move] = 1
occupied_64 = board_to_occupied_64(board.state)
state[64:128] = occupied_64

offset = 128
row = move // 8
col = move % 8
state[offset + 0] = row
state[offset + 1] = col
state[offset + 2] = i
state[offset + 3] = i % 2 == 1
state[offset + 4] = i % 2 == 0

states.append(state)

board.umpire(move)

states = t.stack(states, axis=0)
game_stack.append(states)

px.imshow(to_numpy(board.state), y=list("ABCDEFGH"), x=[str(i) for i in range(8)], aspect="equal")

# %%
