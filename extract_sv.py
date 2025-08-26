import numpy as np
import os
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
import sys
from transformers import set_seed
from sklearn.covariance import ledoit_wolf

from helper import read_config, get_nnsight_model, load_label_info
from probing import load_data, prepare_data


def estimate_single_dir_from_emb(emb):
    category_mean = emb.mean(dim=0)
    cov = ledoit_wolf(emb.numpy())
    cov = torch.tensor(cov[0])
    pseudo_inv = torch.linalg.pinv(cov)
    lda_dir = pseudo_inv @ category_mean
    lda_dir = lda_dir / torch.norm(lda_dir)
    lda_dir = (category_mean @ lda_dir) * lda_dir

    return lda_dir, category_mean


def get_dirs_for_target(X, y, lookup):
    dirs = {}
    for name, target_values in lookup.items():
        cat_emb = torch.tensor(X[np.isin(y, target_values)])
        lda_dir, category_mean = estimate_single_dir_from_emb(cat_emb)
        dirs[name] = {"lda": lda_dir, "mean": category_mean}
    return dirs


def get_random_dir(X, num):
    idx = np.random.choice(X.shape[0], num, replace=False)
    cat_emb = torch.tensor(X[idx])
    lda_dir, category_mean = estimate_single_dir_from_emb(cat_emb)
    dirs = {}
    dirs[f"random_n{num}"] = {"lda": lda_dir, "mean": category_mean}
    return dirs


def get_all_dirs(X, y, label_lookup):
    target_dirs = {}
    general_dir = get_dirs_for_target(
        X, y, {"general": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
    )
    random_dir_320 = get_random_dir(X, 320)
    random_dir_240 = get_random_dir(X, 240)
    random_dir_80 = get_random_dir(X, 80)
    tense_dirs = get_dirs_for_target(X, y, label_lookup["tense"])
    aspect_dirs = get_dirs_for_target(X, y, label_lookup["aspect"])
    fulltense_dirs = get_dirs_for_target(X, y, label_lookup["full_tense"])
    target_dirs.update(tense_dirs)
    target_dirs.update(aspect_dirs)
    target_dirs.update(fulltense_dirs)
    target_dirs.update(general_dir)
    target_dirs.update(random_dir_320)
    target_dirs.update(random_dir_240)
    target_dirs.update(random_dir_80)
    return target_dirs


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

    model_name = sys.argv[1]
    config = read_config(model_name)
    _, blocks = get_nnsight_model(config)
    train_samples = load_data(sources=["propbank", "synthetic"])
    test_samples = load_data(sources=["bigbench"])
    print("TRAIN and TEST data loaded")

    train_dir = f"data/train/samples/hidden_states/{model_name}"
    test_dir = f"data/test/samples/hidden_states/{model_name}"

    label_info = load_label_info()

    label_mapper = label_info["label_mapper"]
    label_lookup = label_info["label_lookup"]

    agg = "sum_norm"

    for layer_idx in tqdm(range(blocks)):

        X_train, y_train, X_test, y_test, _ = prepare_data(
            "full_tense", layer_idx, "resid", agg, train_dir, test_dir, label_mapper
        )

        X_full = np.append(X_train, X_test, axis=0)
        y_full = np.append(y_train, y_test, axis=0)

        target_dirs = get_all_dirs(X_train, y_train, label_lookup)
        target_dirs_full = get_all_dirs(X_full, y_full, label_lookup)

        RESULT_DIR = f"results/{model_name}/feature_vectors/l{layer_idx}"
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(f"{RESULT_DIR}/target_dirs.pkl", "wb") as f1:
            pickle.dump(target_dirs, f1)
        with open(f"{RESULT_DIR}/target_dirs_full.pkl", "wb") as f2:
            pickle.dump(target_dirs_full, f2)
