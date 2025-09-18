# %%
import os
import numpy as np
import pandas as pd
import wittgenstein as lw
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import Parallel, delayed
import json


# -----------------------------
# 1. Data Loading
# -----------------------------
def load_neuron_activations(base_path=None, layers=5):
    if base_path is None:
        base_path = "/Users/srujanamedicherla/Desktop/Algoverse_project/OthelloUnderstanding/fine_grained_analysis"

    layer_activations, layer_features = {}, {}
    print("Loading neuron activation data and input features...")

    for layer in range(1, layers + 1):
        for kind, store in [("y", layer_activations), ("X", layer_features)]:
            fname = f"unified_features_{kind}_layer_{layer}.npy"
            fpath = os.path.join(base_path, fname)
            try:
                arr = np.load(fpath)
                store[layer] = arr
                print(f"Layer {layer}: Loaded {kind.upper()} with shape {arr.shape}")
            except FileNotFoundError:
                print(f"Warning: {kind.upper()} file not found for layer {layer}: {fpath}")

    return layer_activations, layer_features


# -----------------------------
# 2. Train RIPPER on one neuron
# -----------------------------
def train_ripper_on_neuron(layer_features, layer_activations, target_layer, target_neuron, sample_size=50000):
    X = layer_features[target_layer]
    Y = layer_activations[target_layer][:, target_neuron]

    # Binary labels: Strong (1) = top 10% activations
    p90 = np.percentile(Y, 90)
    y_binary = (Y >= p90).astype(int)

    # Sample for efficiency
    n_samples = min(sample_size, len(Y))
    idx = np.random.choice(len(Y), size=n_samples, replace=False)
    X_sampled, y_sampled = X[idx], y_binary[idx]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sampled, y_sampled, test_size=0.2, stratify=y_sampled, random_state=42
    )

    # Convert to DataFrame (required by wittgenstein)
    df_train, df_test = pd.DataFrame(X_train), pd.DataFrame(X_test)
    df_train['label'], df_test['label'] = y_train, y_test

    ripper = lw.RIPPER()
    ripper.fit(df_train.drop(columns=['label']), df_train['label'], pos_class=1)

    # Evaluate
    y_pred = ripper.predict(df_test.drop(columns=['label']))
    report = classification_report(df_test['label'], y_pred, target_names=['Non-Strong', 'Strong'], output_dict=True)

    return {
        "layer": target_layer,
        "neuron": target_neuron,
        "rules": str(ripper.ruleset_),
        "threshold": float(p90),
        "class_balance": {
            "non_strong": int((y_binary == 0).sum()),
            "strong": int((y_binary == 1).sum())
        },
        "test_report": report
    }


# -----------------------------
# 3. Parallel wrapper
# -----------------------------
def train_and_collect(layer, neuron, layer_features, layer_activations):
    try:
        return train_ripper_on_neuron(layer_features, layer_activations, layer, neuron)
    except Exception as e:
        return {"layer": layer, "neuron": neuron, "error": str(e)}


# -----------------------------
# 4. Main: One Machine = One Layer
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True, help="Which layer to train on (1–5)")
    parser.add_argument("--cores", type=int, default=32, help="Number of CPU cores")
    parser.add_argument("--neurons", type=int, default=2048, help="Total neurons in layer")
    args = parser.parse_args()

    # Load data
    layer_activations, layer_features = load_neuron_activations()

    # Get all neurons in this layer
    assigned_neurons = list(range(args.neurons))
    print(f"Machine assigned: Layer {args.layer}, Neurons 0 → {args.neurons-1}")

    # Train in parallel across neurons
    results = Parallel(n_jobs=args.cores, verbose=5)(
        delayed(train_and_collect)(args.layer, n, layer_features, layer_activations)
        for n in assigned_neurons
    )

    # Save results
    os.makedirs("ripper_results", exist_ok=True)
    out_path = f"ripper_results/layer{args.layer}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved results for Layer {args.layer} to {out_path}")