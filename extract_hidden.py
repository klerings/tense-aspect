import torch
import random

import sys
from transformers import set_seed
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import defaultdict

from helper import get_nnsight_model, read_config, empty_cache


class TimeSample:
    def __init__(self, sid, sent, tense, aspect, file_name):
        self.sid = sid
        self.sent = sent

        self.tense = tense
        self.aspect = aspect
        self.full_tense = f"{tense}_{aspect}"
        self.file_name = file_name

    def __str__(self):
        return (
            f"{self.sent}\nTENSE: {self.tense}\nASPECT: {self.aspect}\n{self.file_name}"
        )


def load_data(sources):
    sc = 0
    all_samples = defaultdict(list)
    DATA_DIR = "data/raw_data/"
    DATA_SOURCES = sources
    for source in DATA_SOURCES:
        data_path = DATA_DIR + source
        for f in os.listdir(data_path):
            with open(f"{data_path}/{f}", "r") as f1:
                samples = [line.strip() for line in f1.readlines()]
                tense_aspect = f[:-4]
                if "simple" in tense_aspect:
                    aspect = 2
                elif "perfect_progressive" in tense_aspect:
                    aspect = 5
                elif "perfect" in tense_aspect:
                    aspect = 4
                elif "progressive" in tense_aspect:
                    aspect = 3
                if "future" in tense_aspect:
                    tense = 1
                elif "past" in tense_aspect:
                    tense = -1
                elif "present" in tense_aspect:
                    tense = 0
                for sample in samples:
                    s = TimeSample(sc, sample, tense, aspect, f)
                    if s.sent not in [s2.sent for s2 in all_samples[s.full_tense]]:
                        all_samples[s.full_tense].append(s)
                        sc += 1
    return all_samples


def get_features(samples, model, blocks, config, dir):
    X_emb = {}
    X_resid = {b: dict() for b in range(blocks)}
    y_tense = {}
    y_aspect = {}
    y_full_tense = {}

    for s in tqdm(samples):
        sid = s.sid

        with model.trace(s.sent):
            hidden_resid = {}

            # get the hidden states before the first transformer block
            def get_embedding(model, layer_info):
                parts = layer_info[0].split(".")
                target = model
                for part in parts:
                    target = getattr(target, part)
                if layer_info[-1] == 0:
                    hs = target.output[0][0]
                else:
                    hs = target.output
                return hs

            emb = get_embedding(model, config["layer_names"]["embed"])[0].save()

            def get_layers(model, layer_info, layer_idx):
                """
                Generalized function to get the hidden states of a specific layer.
                Args:
                    model: The model being patched.
                    layer_info: The layer attribute names.
                    layer_idx: idx of block
                Returns:
                    The hidden states of the specified layer.
                """
                parts = layer_info[0].split(".")
                target = model
                for part in parts:
                    if part == "{idx}":
                        target = target[layer_idx]
                    else:
                        target = getattr(target, part)

                if layer_info[-1] == 0:
                    hs = target.output[0]
                else:
                    hs = target.output
                return hs

            for layer_idx in range(blocks):
                hidden_resid[layer_idx] = get_layers(
                    model, config["layer_names"]["resid"], layer_idx
                )[0].save()
        empty_cache()

        X_emb[sid] = emb.detach().cpu().numpy()

        for (layer_idx, resid) in hidden_resid.items():
            X_resid[layer_idx][sid] = resid.detach().cpu().numpy()

        y_tense[sid] = s.tense
        y_aspect[sid] = s.aspect
        y_full_tense[sid] = s.full_tense

    save_features(X_emb, X_resid, y_tense, y_aspect, y_full_tense, dir, blocks)


def save_features(X_emb, X_resid, y_tense, y_aspect, y_full_tense, dir, blocks):
    """
    save all hidden states (resid/emb) of all layers given a dataset
    """
    for layer_idx in range(blocks):
        with open(f"{dir}/resid{layer_idx}.pkl", "wb") as f1:
            pickle.dump(X_resid[layer_idx], f1)

    with open(f"{dir}/emb.pkl", "wb") as f2:
        pickle.dump(X_emb, f2)

    with open(f"{dir}/labels_tense.pkl", "wb") as f3:
        pickle.dump(y_tense, f3)
    with open(f"{dir}/labels_aspect.pkl", "wb") as f4:
        pickle.dump(y_aspect, f4)
    with open(f"{dir}/labels_full_tense.pkl", "wb") as f5:
        pickle.dump(y_full_tense, f5)


if __name__ == "__main__":

    # Preliminaries
    seed_value = 10
    set_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.set_grad_enabled(
        False
    )  # turn of automatic differentiation to save GPU memory because we focus on inference not training
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_name = sys.argv[1]

    # load model
    config = read_config(model_name)
    model, blocks = get_nnsight_model(config)
    print("model loaded")

    train_samples = load_data(sources=["propbank", "synthetic"])
    print(f"TRAIN samples loaded")
    test_samples = load_data(sources=["bigbench"])
    print(f"TEST samples loaded")

    train_dir = f"data/train/samples/hidden_states/{model_name}"
    test_dir = f"data/test/samples/hidden_states/{model_name}"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_data = []
    test_data = []

    for tense in train_samples.keys():
        train_data.extend(train_samples[tense])
        test_data.extend(test_samples[tense])

    get_features(train_data, model, blocks, config, dir=train_dir)
    print("TRAIN features retrieved")

    get_features(test_data, model, blocks, config, dir=test_dir)
    print("TEST features retrieved")
