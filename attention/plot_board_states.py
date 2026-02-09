# %%
import os
from pathlib import Path

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(BASE_PATH)
BASE_PATH = Path(BASE_PATH)
os.chdir(BASE_PATH)

from utils import circuits_utils
from utils.plot_utils import plot_board_states

# %%
test_size = 500
device = "cpu"
custom_functions = [
    # othello_utils.games_batch_to_input_tokens_flipped_bs_classifier_input_BLC,
    # othello_utils.games_batch_to_input_tokens_flipped_pbs_classifier_input_BLC,
    # othello_utils.games_batch_to_valid_moves_BLRRC, # (legal move)
]
test_data = circuits_utils.construct_othello_dataset(
    custom_functions=custom_functions,
    n_inputs=test_size,
    split="test", 
    device=device,
)

# %%
game_idx = 25
move = 31
plot_board_states(
    data=test_data,
    game_index=game_idx,
    move=move,
    save_path=BASE_PATH / "figures" / "attention_plots" / "attention_one_game",
    figure_type="png",
)

# %%
