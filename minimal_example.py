# %%
# Minimal Example for OthelloUnderstanding Codebase
# This example demonstrates how to:
# 1. Load + run the Othello model
# 2. Load + run linear probes
# 3. Load + visualize decision trees
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

device = "cuda" if t.cuda.is_available() else "cpu"
t.set_grad_enabled(False)

print(f"Using device: {device}")

# %%
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
dataset_size = 50
custom_functions = [
    othello_utils.games_batch_to_input_tokens_flipped_bs_classifier_input_BLC,
]
model = utils.get_model(model_name, device)
train_data = construct_othello_dataset(
    custom_functions=custom_functions,
    n_inputs=dataset_size,
    split="train",
    device=device,
)
test_data = construct_othello_dataset(
    custom_functions=custom_functions,
    n_inputs=dataset_size//2,
    split="test", 
    device=device,
)

# %%
for key in train_data.keys():
    if isinstance(train_data[key][0], t.Tensor):
        print(f"{key} : {train_data[key][0].shape}")
    print(f"{key} : {train_data[key][0]}")

# %%
# First 10 moves of game 1
sample_input = t.tensor(train_data["encoded_inputs"][0][:10]).to(device)
with model.trace(sample_input):
    logits = model.unembed.output.save()
logprobs = logits.log_softmax(dim=-1)

# %%
MIDDLE_SQUARES = [27, 28, 35, 36]
ALL_SQUARES = [i for i in range(64) if i not in MIDDLE_SQUARES]

logprobs_board = t.full(size=(8, 8), fill_value=-13.0, device=device)
logprobs_board.flatten()[ALL_SQUARES] = logprobs[
    0, 0, 1:
]  # the [1:] is to filter out logits for the "pass" move

neel_utils.plot_board_values(logprobs_board, title="Example Log Probs", width=500)

# %%
TOKEN_IDS_2D = np.array(
    [str(i) if i in ALL_SQUARES else "" for i in range(64)]
).reshape(8, 8)
BOARD_LABELS_2D = np.array(
    ["ABCDEFGH"[i // 8] + f"{i % 8}" for i in range(64)]
).reshape(8, 8)

print(TOKEN_IDS_2D)
print(BOARD_LABELS_2D)

neel_utils.plot_board_values(
    t.stack([logprobs_board, logprobs_board]),  # shape (2, 8, 8)
    title="Example Log Probs (with annotated token IDs)",
    width=800,
    text=np.stack([TOKEN_IDS_2D, BOARD_LABELS_2D]),  # shape (2, 8, 8)
    board_titles=["Labelled by token ID", "Labelled by board label"],
)

# %%
logprobs_multi_board = t.full(size=(10, 8, 8), fill_value=-13.0, device=device)
logprobs_multi_board.flatten(1, -1)[:, ALL_SQUARES] = logprobs[
    0, :, 1:
]  # we now do all 10 moves at once

neel_utils.plot_board_values(
    logprobs_multi_board,
    title="Example Log Probs",
    width=1000,
    boards_per_row=5,
    board_titles=[f"Logprobs after move {i}" for i in range(1, 11)],
)

# %%
board_states = t.zeros((10, 8, 8), dtype=t.int32)
legal_moves = t.zeros((10, 8, 8), dtype=t.int32)

board = neel_utils.OthelloBoardState()
for i, token_id in enumerate(sample_input.squeeze()):
    # board.umpire takes a square index (i.e. from 0 to 63) and makes a move on the board
    board.umpire(neel_utils.id_to_square(token_id))

    # board.state gives us the 8x8 numpy array of 0 (blank), -1 (black), 1 (white)
    board_states[i] = t.from_numpy(board.state)

    # board.get_valid_moves() gives us a list of the indices of squares that are legal to play next
    legal_moves[i].flatten()[board.get_valid_moves()] = 1

# Turn `legal_moves` into strings, with "o" where the move is legal and empty string where illegal
legal_moves_annotation = np.where(to_numpy(legal_moves), "o", "").tolist()

neel_utils.plot_board_values(
    board_states,
    title="Board states",
    width=1000,
    boards_per_row=5,
    board_titles=[f"State after move {i}" for i in range(1, 11)],
    text=legal_moves_annotation,
)

# %%
board_seqs_id = t.tensor(train_data["encoded_inputs"]).long()
board_seqs_square = t.tensor(train_data["decoded_inputs"]).long()

# %%
def get_board_states_and_legal_moves(
    games_square: Int[Tensor, "n_games n_moves"],
) -> tuple[
    Int[Tensor, "n_games n_moves rows cols"],
    Int[Tensor, "n_games n_moves rows cols"],
    list,
]:
    """
    Returns the following:
        states:                 (n_games, n_moves, 8, 8): tensor of board states after each move
        legal_moves:            (n_games, n_moves, 8, 8): tensor of 1s for legal moves, 0s for illegal moves
        legal_moves_annotation: (n_games, n_moves, 8, 8): list containing strings of "o" for legal moves (for plotting)
    """
    # Create tensors to store the board state & legal moves
    n_games, n_moves = games_square.shape
    states = t.zeros((n_games, 60, 8, 8), dtype=t.int32)
    legal_moves = t.zeros((n_games, 60, 8, 8), dtype=t.int32)

    # Loop over each game, populating state & legal moves tensors after each move
    for n in range(n_games):
        board = neel_utils.OthelloBoardState()
        for i in range(n_moves):
            board.umpire(games_square[n, i].item())
            states[n, i] = t.from_numpy(board.state)
            legal_moves[n, i].flatten()[board.get_valid_moves()] = 1

    # Convert legal moves to annotation
    legal_moves_annotation = np.where(to_numpy(legal_moves), "o", "").tolist()

    return states, legal_moves, legal_moves_annotation


num_games = 50

focus_games_id = board_seqs_id[:num_games]  # shape [50, 60]
focus_games_square = board_seqs_square[:num_games]  # shape [50, 60]
focus_states, focus_legal_moves, focus_legal_moves_annotation = (
    get_board_states_and_legal_moves(focus_games_square)
)

print("focus states:", focus_states.shape)
print("focus_legal_moves", tuple(focus_legal_moves.shape))

# Plot the first 10 moves of the first game
neel_utils.plot_board_values(
    focus_states[0, :10],
    title="Board states",
    width=1000,
    boards_per_row=5,
    board_titles=[
        f"Move {i}, {'white' if i % 2 == 1 else 'black'} to play" for i in range(1, 11)
    ],
    text=np.where(to_numpy(focus_legal_moves[0, :10]), "o", "").tolist(),
)

# %%
focus_logits, focus_cache = model.run_with_cache(focus_games_id[:, :-1].to(device))

# %%
probe_dict = {i : t.load(
    f"linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{i}.pth", map_location=str(device), weights_only="True"
)['linear_probe'].squeeze() for i in range(model.cfg.n_layers)}


# %%
def plot_probe_outputs(
    cache: ActivationCache,
    probe_dict : dict[int, Tensor],
    layer: int,
    game_index: int,
    move: int,
    title: str = "Probe outputs",
):
    linear_probe = probe_dict[layer]
    residual_stream = cache["resid_post", layer][game_index, move]
    probe_out = einops.einsum(
        residual_stream,
        linear_probe,
        "d_model, d_model row col options -> options row col",
    )

    neel_utils.plot_board_values(
        probe_out.softmax(dim=0),
        title=title,
        width=900,
        height=400,
        board_titles=["P(Mine)", "P(Empty)", "P(Their's)"],
        # text=BOARD_LABELS_2D,
    )

# %%
game_index = 0
move = 29

neel_utils.plot_board_values(
    focus_states[game_index, move],
    title="Focus game states",
    width=400,
    height=400,
    text=focus_legal_moves_annotation[game_index][move],
)

for layer in range(model.cfg.n_layers):
    plot_probe_outputs(
        focus_cache,
        probe_dict,
        layer,
        game_index,
        move,
        title=f"Layer {layer} probe outputs after move {move}",
    )

# %%
l5_probe = probe_dict[5]
blank_probe = l5_probe[..., 1] - (l5_probe[..., 0] + l5_probe[..., 2]) / 2
my_probe = l5_probe[..., 0] - l5_probe[..., 2]
# Scale the probes down to be unit norm per cell
blank_probe_normalised = blank_probe / blank_probe.norm(dim=0, keepdim=True)
my_probe_normalised = my_probe / my_probe.norm(dim=0, keepdim=True)

# Set the center blank probes to 0, since they're never blank so the probe is meaningless
blank_probe_normalised[:, [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

#%%
def get_w_in(
    model: HookedTransformer,
    layer: int,
    neuron: int,
    normalize: bool = False,
) -> Float[Tensor, "d_model"]:
    """
    Returns the input weights for the given neuron.

    If normalize is True, the weight is normalized to unit norm.
    """
    w_in = model.W_in[layer, :, neuron].detach().clone()
    if normalize:
        w_in /= w_in.norm(dim=0, keepdim=True)
    return w_in


def get_w_out(
    model: HookedTransformer,
    layer: int,
    neuron: int,
    normalize: bool = False,
) -> Float[Tensor, "d_model"]:
    """
    Returns the output weights for the given neuron.

    If normalize is True, the weight is normalized to unit norm.
    """
    w_out = model.W_out[layer, neuron, :].detach().clone()
    if normalize:
        w_out /= w_out.norm(dim=0, keepdim=True)
    return w_out


def calculate_neuron_input_weights(
    model: HookedTransformer, probe: Float[Tensor, "d_model row col"], layer: int, neuron: int
) -> Float[Tensor, "rows cols"]:
    """
    Returns tensor of the input weights for the given neuron, at each square on the board, projected
    along the corresponding probe directions.

    Assume probe directions are normalized. You should also normalize the model weights.
    """
    w_in = get_w_in(model, layer, neuron, normalize=True)

    return einops.einsum(w_in, probe, "d_model, d_model row col -> row col")


def calculate_neuron_output_weights(
    model: HookedTransformer, probe: Float[Tensor, "d_model row col"], layer: int, neuron: int
) -> Float[Tensor, "rows cols"]:
    """
    Returns tensor of the output weights for the given neuron, at each square on the board,
    projected along the corresponding probe directions.

    Assume probe directions are normalized. You should also normalize the model weights.
    """
    w_out = get_w_out(model, layer, neuron, normalize=True)

    return einops.einsum(w_out, probe, "d_model, d_model row col -> row col")

layer = 5
neuron = 4

w_in_L5N1393_blank = calculate_neuron_input_weights(model, blank_probe_normalised, layer, neuron)
w_in_L5N1393_my = calculate_neuron_input_weights(model, my_probe_normalised, layer, neuron)

neel_utils.plot_board_values(
    t.stack([w_in_L5N1393_blank, w_in_L5N1393_my]),
    title=f"Input weights in terms of the probe for neuron L{layer}N{neuron}",
    board_titles=["Blank In", "My In"],
    width=650,
    height=380,
)

w_out_L5N1393 = get_w_out(model, layer, neuron, normalize=True)
W_U_normalized = model.W_U[:, 1:] / model.W_U[:, 1:].norm(
    dim=0, keepdim=True
)  # normalize, slice off logits for "pass"
cos_sim = w_out_L5N1393 @ W_U_normalized  # shape (60,)

# Turn into a (rows, cols) tensor, using indexing
cos_sim_rearranged = t.zeros((8, 8), device=device)
cos_sim_rearranged.flatten()[ALL_SQUARES] = cos_sim

# Plot results
neel_utils.plot_board_values(
    cos_sim_rearranged,
    title=f"Cosine sim of neuron L{layer}N{neuron} with W<sub>U</sub> directions",
    width=450,
    height=380,
)

# %%