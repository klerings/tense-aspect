import os
import joblib
import random
import numpy as np
from sklearn.utils import resample
import pickle
import sys
import plotly.graph_objects as go
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from transformers import set_seed
from sklearn.metrics import f1_score

from extract_hidden import TimeSample, load_data
from helper import read_config, get_nnsight_model, load_label_info


def train_probing_clf(x_train, y_train):
    lr = LogisticRegression(max_iter=3000, multi_class="multinomial", solver="lbfgs")
    lr.fit(x_train, y_train)
    return lr


def evaluate_probing_clf(clf, x_test, y_test):
    y_hat = clf.predict(x_test)
    return f1_score(y_test, y_hat, labels=list(set(y_test)), average="macro"), y_hat


def train_and_eval(X_train, y_train, X_test, y_test, return_clf=False):

    clf = train_probing_clf(X_train, y_train)
    score, preds = evaluate_probing_clf(clf, X_test, y_test)
    if return_clf:
        return score, preds, clf
    else:
        return score, preds


def center_activations(X_train, X_test, return_mean=False):
    ds_mean = X_train.mean(axis=0, keepdims=True)
    X_train_centered = X_train - ds_mean
    X_test_centered = X_test - ds_mean
    if return_mean:
        return X_train_centered, X_test_centered, ds_mean
    else:
        return X_train_centered, X_test_centered


def get_f1_score(y_test, y_pred):
    return f1_score(y_test, y_pred, labels=list(set(y_test)), average="macro")


def load_features(feature, layer_idx, target, dir, train_class_mode="mixed"):
    """
    load all hidden states (resid/mlp/emb) of all layers given a dataset
    """
    with open(f"{dir}/labels_tense.pkl", "rb") as f1:
        y_dict_tense = pickle.load(f1)

    with open(f"{dir}/labels_aspect.pkl", "rb") as f2:
        y_dict_aspect = pickle.load(f2)

    with open(f"{dir}/labels_full_tense.pkl", "rb") as f3:
        y_dict_fulltense = pickle.load(f3)

    if feature == "emb":
        with open(f"{dir}/emb.pkl", "rb") as f4:
            X_dict = pickle.load(f4)
    else:
        with open(f"{dir}/{feature}{layer_idx}.pkl", "rb") as f5:
            X_dict = pickle.load(f5)

    X = []
    y = []
    if target == "tense":
        y_dict = y_dict_tense
    elif target == "aspect":
        y_dict = y_dict_aspect
    elif target == "full_tense":
        y_dict = y_dict_fulltense

    if train_class_mode == "mixed":
        for sid, label in y_dict.items():
            y.append(label)
            X.append(X_dict[sid])
    elif train_class_mode == "simplified":
        if target == "tense":
            for sid, label in y_dict.items():
                # only keep simple aspect
                if y_dict_aspect[sid] == 2:
                    y.append(label)
                    X.append(X_dict[sid])
        elif target == "aspect":
            for sid, label in y_dict.items():
                # only keep simple aspect
                if y_dict_tense[sid] == 0:
                    y.append(label)
                    X.append(X_dict[sid])
        else:
            raise Exception("simplified training data not available for fulltense")
    return X, y


def train_resid_probe(
    target, layer_idx, layer_type, agg, train_dir, test_dir, label_mapper
):
    X_train, y_train, X_test, y_test, train_ds_mean = prepare_data(
        target, layer_idx, layer_type, agg, train_dir, test_dir, label_mapper
    )
    score, preds, clf = train_and_eval(
        X_train, y_train, X_test, y_test, return_clf=True
    )

    return score, preds, y_test, clf, train_ds_mean


def create_heatmap(matrix, title):
    fig = go.Figure()

    # Convert dictionary to a 2D list for the heatmap
    z_values = list(matrix.values())
    y_labels = list(matrix.keys())
    x_labels = [f"Layer {i}" for i in range(0, blocks + 1)]

    fig.add_trace(
        go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            colorscale="RdBu",
            colorbar=dict(title="F1-Score"),
            hoverongaps=False,
            text=[[f"{value:.2f}" for value in row] for row in z_values],
            hovertemplate="Position: %{y}<br>Layer: %{x}<br>F1-Score: %{text}<extra></extra>",
            zmin=0,
            zmax=1,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Layers",
        yaxis_title="Position",
        template="plotly_white",
    )

    return fig


def prepare_data(target, layer_idx, layer_type, agg, train_dir, test_dir, label_mapper):
    """load data, aggregate sequence representation, balance and shuffle train data"""
    X_train, y_train = load_features(layer_type, layer_idx, target, train_dir, "mixed")
    X_test, y_test = load_features(layer_type, layer_idx, target, test_dir, "mixed")
    if target == "full_tense":
        y_train = [label_mapper[y] for y in y_train]
        y_test = [label_mapper[y] for y in y_test]
    # aggregate hidden states
    if agg == "mean":
        X_train = np.array([np.mean(x, axis=0) for x in X_train])
        X_test = np.array([np.mean(x, axis=0) for x in X_test])
    elif agg == "sum":
        X_train = np.array([np.sum(x, axis=0) for x in X_train])
        X_test = np.array([np.sum(x, axis=0) for x in X_test])
    elif agg == "sum_norm":
        X_train = np.array([np.sum(x, axis=0) / np.sqrt(x.shape[0]) for x in X_train])
        X_test = np.array([np.sum(x, axis=0) / np.sqrt(x.shape[0]) for x in X_test])
    elif agg == "last_tok":
        X_train = np.array([x[-1] for x in X_train])
        X_test = np.array([x[-1] for x in X_test])
    # downsample train data to smallest class
    classes, counts = np.unique(y_train, return_counts=True)
    min_count = np.min(counts)
    X_train_resampled, y_train_resampled = [], []

    for cls in classes:
        cls_indices = np.where(y_train == cls)[0]
        sampled_indices = resample(
            cls_indices, replace=False, n_samples=min_count, random_state=42
        )
        X_train_resampled.append(X_train[sampled_indices])
        y_train_resampled.append(np.array(y_train)[sampled_indices])

    X_train_balanced = np.vstack(X_train_resampled)
    y_train_balanced = np.hstack(y_train_resampled)

    shuffle_indices = np.random.permutation(len(y_train_balanced))
    X_train_balanced = X_train_balanced[shuffle_indices]
    y_train_balanced = y_train_balanced[shuffle_indices]

    # center activations by subtracting the mean
    y_test = np.array(y_test)
    X_train_balanced, X_test, ds_mean = center_activations(
        X_train_balanced, X_test, return_mean=True
    )
    return X_train_balanced, y_train_balanced, X_test, y_test, ds_mean


def probe_full_layer(
    train_dir,
    test_dir,
    model_name,
    label_mapper,
    targets,
    blocks,
    aggs,
):
    for target in targets:
        print(f"processing: {target}")
        matrix = {}
        for agg in aggs:
            print(f"agg: {agg}")
            matrix[f"{agg}"] = []

            # train probe on embedding layer
            score_emb, preds_emb, y_test, clf_emb, _ = train_resid_probe(
                target, None, "emb", agg, train_dir, test_dir, label_mapper
            )
            matrix[f"{agg}"].append(score_emb)
            os.makedirs(
                f"results/{model_name}/probing_models/{target}/embedding", exist_ok=True
            )
            joblib.dump(
                clf_emb,
                f"results/{model_name}/probing_models/{target}/embedding/probe_model_{agg}.joblib",
                compress=3,
            )
            print("finished emb")
            # train probe per resid layer
            ds_mean_per_layer = {}
            for layer_idx in range(blocks):
                print(f"layer: {layer_idx}")
                (
                    score_resid,
                    preds_resid,
                    _,
                    clf_resid,
                    train_ds_mean,
                ) = train_resid_probe(
                    target, layer_idx, "resid", agg, train_dir, test_dir, label_mapper
                )
                matrix[f"{agg}"].append(score_resid)
                os.makedirs(
                    f"results/{model_name}/probing_models/{target}/l{layer_idx}",
                    exist_ok=True,
                )
                joblib.dump(
                    clf_resid,
                    f"results/{model_name}/probing_models/{target}/l{layer_idx}/probe_model_{agg}.joblib",
                    compress=3,
                )
                ds_mean_per_layer[layer_idx] = train_ds_mean
            with open(
                f"results/{model_name}/probing_models/{target}/train_ds_mean_for_centering_{agg}.pkl",
                "wb",
            ) as f:
                pickle.dump(ds_mean_per_layer, f)

        with open(f"results/{model_name}/full_layer_probe_{target}.pkl", "wb") as f2:
            pickle.dump(matrix, f2)
        os.makedirs(f"plots/{model_name}", exist_ok=True)
        fig_resid = create_heatmap(
            matrix, f"Probing {target} on Residual Stream per Layer"
        )

        fig_resid.write_html(f"plots/{model_name}/probing_full_layer_{target}.html")


if __name__ == "__main__":
    # Preliminaries
    seed_value = 10
    set_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

    model_name = sys.argv[1]

    config = read_config(model_name)
    # load model
    _, blocks = get_nnsight_model(config)
    print("model loaded")
    train_samples = load_data(sources=["propbank", "synthetic"])
    test_samples = load_data(sources=["bigbench"])
    print("data loaded")

    train_dir = f"data/train/samples/hidden_states/{model_name}"
    test_dir = f"data/test/samples/hidden_states/{model_name}"

    label_info = load_label_info()
    targets = ["tense", "aspect", "full_tense"]
    aggs = ["sum", "sum_norm", "mean", "last_tok"]

    probe_full_layer(
        train_dir,
        test_dir,
        model_name,
        label_info["label_mapper"],
        targets,
        blocks,
        aggs,
    )
