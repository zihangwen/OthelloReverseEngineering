from ast import main
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
import torch
from nnsight import NNsight
import json
from typing import Any
from datasets import load_dataset
from einops import rearrange
from jaxtyping import Int, Float, jaxtyped
from torch import Tensor
import os
from tqdm import tqdm
from transformers import GPT2LMHeadModel
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils
from enum import Enum
from typing import Optional, Union
import pandas as pd


def get_model(model_name: str, device: torch.device) -> NNsight:
    if model_name == "Baidicoot/Othello-GPT-Transformer-Lens":
        tf_model = HookedTransformer.from_pretrained("Baidicoot/Othello-GPT-Transformer-Lens")
        model = NNsight(tf_model).to(device)
        return model
    
    if model_name == "mntss/Othello-GPT":
        tf_model = HookedTransformer.from_pretrained_no_processing("Baidicoot/Othello-GPT-Transformer-Lens")
        state_dict = tl_utils.download_file_from_hf("mntss/Othello-GPT", "tl_model.pth")
        tf_model.load_state_dict(state_dict)
        model = NNsight(tf_model).to(device)
        return model

    if (
        model_name == "adamkarvonen/RandomWeights8LayerOthelloGPT2"
        or model_name == "adamkarvonen/RandomWeights8LayerChessGPT2"
    ):
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        model = NNsight(model).to(device)
        return model

    if model_name == "adamkarvonen/8LayerChessGPT2":
        # Old method of loading model from nanogpt weights
        # model = convert_nanogpt_model(
        #     f"{model_path}lichess_8layers_ckpt_no_optimizer.pt", torch.device(device)
        # )
        # tokenizer = NanogptTokenizer(meta_path=f"{model_path}meta.pkl")
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        model = NNsight(model).to(device)
        return model

    raise ValueError("Model not found.")

