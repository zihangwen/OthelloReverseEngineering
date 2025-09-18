import numpy as np
import torch as t
import sys
import os
import argparse
from tqdm import tqdm

# Add parent directory to path to import circuits modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuits import othello_utils, utils
from circuits.eval_sae_as_classifier import construct_othello_dataset

device = t.device("cuda" if t.cuda.is_available() else "cpu")


def load_probe_models():
    model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
    model = utils.get_model(model_name, device=device)

    # Load board state probes
    board_state_probes, flipped_probes, played_probes = {}, {}, {}
    for layer in range(model.cfg.n_layers):
        bpath = f"../linear_probes/Othello-GPT-Transformer-Lens_othello_mine_yours_probe_layer_{layer}.pth"
        fpath = f"../linear_probes_flipped/resid_{layer}_flipped.pth"
        ppath = f"../linear_probes_just_played/resid_{layer}_played.pth"

        if os.path.exists(bpath):
            pdata = t.load(bpath, map_location=str(device))
            if isinstance(pdata, dict) and "linear_probe" in pdata:
                board_state_probes[layer] = pdata["linear_probe"].squeeze()
        if os.path.exists(fpath):
            flipped_probes[layer] = t.load(fpath, map_location=str(device)).squeeze()
        if os.path.exists(ppath):
            played_probes[layer] = t.load(ppath, map_location=str(device)).squeeze()

    return model, board_state_probes, flipped_probes, played_probes


def extract_probe_features(model, board_state_probes, flipped_probes, played_probes,
                          encoded_inputs, state_stack, target_layer=1, move_range=(5, 30)):

    probe_layer = target_layer - 1
    start_move, end_move = move_range

    valid_games, valid_states = [], []
    for game_idx, game_tokens in enumerate(encoded_inputs):
        if len(game_tokens) >= end_move:
            valid_games.append(game_tokens[start_move:end_move])
            valid_states.append(state_stack[game_idx, start_move:end_move])

    if not valid_games:
        raise ValueError(f"No games with ≥ {end_move} moves")
    
    middle_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]  # D3, D4, E3, E4
    non_middle_mask = np.ones(64, dtype=bool)
    for r, c in middle_squares:
        non_middle_mask[r*8 + c] = False
    
    unified_features = {"mine_minus_theirs": [], "blank_minus_occupied": [], "flipped_minus_not": [], "placed": []}
    neuron_activations = []

    for game_moves, game_states in tqdm(zip(valid_games, valid_states), desc=f"Layer {target_layer}"):
        game_tensor = t.tensor(game_moves, device=device).unsqueeze(0)
        with t.no_grad():
            with model.trace(game_tensor, scan=False, validate=False):
                mlp_acts = model.blocks[target_layer].mlp.hook_post.output.save()
                resid_pre_all = model.blocks[target_layer].hook_resid_pre.output.save()
        neuron_activations.append(mlp_acts)

        game_mine_minus_theirs, game_blank_minus_occupied = [], []
        game_flipped_minus_not, game_placed = [], []

        for move_idx, state in enumerate(game_states):
            residual = resid_pre_all[0, move_idx, :]  # [d_model]

            # 1) Mine - Theirs  (use probes from probe_layer = target_layer-1)  
            if probe_layer in board_state_probes:
                board_probe = board_state_probes[probe_layer]  # [d_model, 8, 8, 3]
                mine_proj   = t.einsum('d,drc->rc', residual, board_probe[:, :, :, 0])  # [8,8]
                theirs_proj = t.einsum('d,drc->rc', residual, board_probe[:, :, :, 2])  # [8,8]
                mine_minus_theirs = (mine_proj - theirs_proj).flatten().detach().numpy()
            else:
                mine_state   = state[:, :, 0].flatten().cpu().numpy()
                theirs_state = state[:, :, 2].flatten().cpu().numpy()
                mine_minus_theirs = mine_state - theirs_state

            # 2) Blank - Occupied (60)
            if probe_layer in board_state_probes:
                empty_proj = t.einsum('d,drc->rc', residual, board_probe[:, :, :, 1])  # [8,8]
                occupied_proj = 0.5 * (mine_proj + theirs_proj)
                bmo_all = (empty_proj - occupied_proj).flatten().detach().numpy()
                blank_minus_occupied = bmo_all[non_middle_mask]
            else:
                empty_state = state[:, :, 1].flatten().cpu().numpy()
                mine_state  = state[:, :, 0].flatten().cpu().numpy()
                theirs_state= state[:, :, 2].flatten().cpu().numpy()
                occupied_state = 0.5 * (mine_state + theirs_state)
                bmo_all = (empty_state - occupied_state)
                blank_minus_occupied = bmo_all[non_middle_mask]

            # 3) Flipped - Not Flipped (64)
            if probe_layer in flipped_probes:
                flipped_probe = flipped_probes[probe_layer]  # [d_model, 8, 8, 2]
                flipped_proj = t.einsum('d,drc->rc', residual, flipped_probe[:, :, :, 0])
                not_flipped_proj = t.einsum('d,drc->rc', residual, flipped_probe[:, :, :, 1])
                flipped_minus_not = (flipped_proj - not_flipped_proj).flatten().detach().numpy()
            else:
                if move_idx > 0:
                    prev_state = game_states[move_idx - 1]
                    was_theirs = prev_state[:, :, 2]
                    now_mine   = state[:, :, 0]
                    flipped = (was_theirs & now_mine).flatten().cpu().numpy().astype(float)
                    flipped_minus_not = flipped - (1.0 - flipped)
                else:
                    flipped_minus_not = np.zeros(64)

            # 4) Placed (60)
            if probe_layer in played_probes:
                played_probe = played_probes[probe_layer]  # [d_model, 8, 8]
                placed_all = t.einsum('d,dr->r', residual, played_probe.view(-1, 64)).detach().numpy()
                placed = placed_all[non_middle_mask]
            else:
                if move_idx > 0:
                    prev_state = game_states[move_idx - 1]
                    was_empty = prev_state[:, :, 1]
                    now_occ   = (state[:, :, 0] | state[:, :, 2])
                    placed_all = (was_empty & now_occ).flatten().cpu().numpy().astype(float)
                    placed = placed_all[non_middle_mask]
                else:
                    placed = np.zeros(60)

            game_mine_minus_theirs.append(mine_minus_theirs)
            game_blank_minus_occupied.append(blank_minus_occupied)
            game_flipped_minus_not.append(flipped_minus_not)
            game_placed.append(placed)

        unified_features["mine_minus_theirs"].append(np.stack(game_mine_minus_theirs))
        unified_features["blank_minus_occupied"].append(np.stack(game_blank_minus_occupied))
        unified_features["flipped_minus_not"].append(np.stack(game_flipped_minus_not))
        unified_features["placed"].append(np.stack(game_placed))

    for k in unified_features:
        unified_features[k] = np.stack(unified_features[k])

    neuron_activations = t.cat(neuron_activations, dim=0)
    n_games, n_moves = unified_features["mine_minus_theirs"].shape[:2]
    neuron_activations = neuron_activations.view(n_games, n_moves, -1)

    return unified_features, neuron_activations


def create_final_dataset(unified_features, neuron_activations):
    mine_minus_theirs = unified_features["mine_minus_theirs"].reshape(-1, 64)
    blank_minus_occupied = unified_features["blank_minus_occupied"].reshape(-1, 60)
    flipped_minus_not = unified_features["flipped_minus_not"].reshape(-1, 64)
    placed = unified_features["placed"].reshape(-1, 60)

    X = np.concatenate([mine_minus_theirs, blank_minus_occupied, flipped_minus_not, placed], axis=1)
    y = neuron_activations.cpu().numpy().reshape(-1, neuron_activations.shape[-1])

    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True, help="Layer to process (1–5)")
    parser.add_argument("--save_meta", action="store_true", help="Also save feature_names and metadata")
    args = parser.parse_args()

    print("Loading probe models...")
    model, board_state_probes, flipped_probes, played_probes = load_probe_models()

    print("Loading dataset...")
    test_data = construct_othello_dataset(
        custom_functions=[othello_utils.games_batch_to_state_stack_mine_yours_BLRRC],
        split="test", device=device, n_inputs=5000
    )
    encoded_inputs = test_data["encoded_inputs"]
    state_stack = test_data["games_batch_to_state_stack_mine_yours_BLRRC"].to(device)

    # Only process the assigned layer
    layer = args.layer
    print(f"\nProcessing Layer {layer}...")

    unified_features, neuron_activations = extract_probe_features(
        model, board_state_probes, flipped_probes, played_probes,
        encoded_inputs, state_stack, target_layer=layer, move_range=(5, 30)
    )

    X, y = create_final_dataset(unified_features, neuron_activations)

    np.save(f"unified_features_X_layer_{layer}.npy", X)
    np.save(f"unified_features_y_layer_{layer}.npy", y)
    print(f"Saved X + y for Layer {layer}")

    if args.save_meta:
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        np.save(f"feature_names_layer_{layer}.npy", feature_names)

        metadata = {"n_samples": X.shape[0], "n_features": X.shape[1],
                    "n_neurons": y.shape[1], "target_layer": layer, "move_range": (5, 30)}
        np.save(f"metadata_layer_{layer}.npy", metadata)
        print(f"Also saved feature_names + metadata for Layer {layer}")


if __name__ == "__main__":
    main()