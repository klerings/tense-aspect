import pandas as pd
import joblib
import numpy as np
import os
import torch
from tqdm import tqdm
from collections import Counter
from helper import empty_cache


def process_unsteered_results(
    model_name,
    file_suffix,
    clf_dict,
    probe_layers,
    layer_means,
    model,
    tokenizer,
    label_dict_tense,
    label_dict_aspect,
    label_dict_full_tense,
    nlp,
):
    if "greedy" in file_suffix:
        actual_suffix = "_greedy"
    else:
        actual_suffix = ""
    df = pd.read_csv(
        f"results/{model_name}/lda_steering/unsteered/results{actual_suffix}.csv"
    )
    output_path = f"results/{model_name}/lda_steering/unsteered/results{actual_suffix}_processed.csv"
    df = process_df(
        df,
        clf_dict,
        probe_layers,
        layer_means,
        model,
        tokenizer,
        label_dict_tense,
        label_dict_aspect,
        label_dict_full_tense,
        nlp,
        output_path,
    )
    return df


def process_df(
    df,
    clf_dict,
    probe_layers,
    layer_means,
    model,
    tokenizer,
    label_dict_tense,
    label_dict_aspect,
    label_dict_full_tense,
    nlp,
    output_path,
    unsteered_pp=None,
):
    # create columns to be updated
    for col_name in [
        "tense",
        "aspect",
        "full_tense",
        "diversity",
        "4-gram-rep",
        "2-gram-rep",
        "3-gram-rep",
        "1-gram-rep",
    ]:
        df[col_name] = "unknown"
    df["complete_sent"] = False

    # only consider first sentence
    df.fillna("", inplace=True)
    df["output_split"] = df["output"].apply(lambda x: x.split(".")[0])

    # degenerate filter 1: diversity and repetition
    quality_results = df["output_split"].apply(lambda x: measure_text_diversity(x))
    for metric in ["diversity", "1-gram-rep", "2-gram-rep", "3-gram-rep", "4-gram-rep"]:
        df[metric] = quality_results.apply(lambda x: x[metric])

    # degenerate filter 2: complete sentence
    df["complete_sent"] = df["output_split"].apply(
        lambda x: is_complete_sentence(x, nlp)
    )

    non_empty_mask = df["output_split"] != ""
    df_non_empty = df[non_empty_mask]

    # Probe only for non-empty rows
    if not df_non_empty.empty:

        X_dict = prepare_features(
            list(df_non_empty["output_split"].values),
            probe_layers,
            layer_means,
            model,
            tokenizer,
        )

        # if np.isnan(X_dict["tense"]).any():
        #    print("X_dict['tense'] contains NaN values!")

        tense_preds = clf_dict["tense"].predict(X_dict["tense"])
        aspect_preds = clf_dict["aspect"].predict(X_dict["aspect"])
        fulltense_preds = clf_dict["full_tense"].predict(X_dict["full_tense"])

        df.loc[df_non_empty.index, "tense"] = [label_dict_tense[p] for p in tense_preds]
        df.loc[df_non_empty.index, "aspect"] = [
            label_dict_aspect[p] for p in aspect_preds
        ]
        df.loc[df_non_empty.index, "full_tense"] = [
            label_dict_full_tense[p] for p in fulltense_preds
        ]

        del X_dict
        empty_cache()

    # compute relative perplexity
    if unsteered_pp is not None:
        df["relative_perplexity"] = df["perplexity"] / unsteered_pp
    df.to_csv(output_path, index=False)
    return df


def load_clfs(model_name, probe_layers):
    tense_layer = probe_layers["tense"]
    aspect_layer = probe_layers["aspect"]
    fulltense_layer = probe_layers["full_tense"]

    # load probe models
    clf_tense = joblib.load(
        f"results/{model_name}/probing_models/tense/l{tense_layer}/probe_model_sum_norm.joblib"
    )
    clf_aspect = joblib.load(
        f"results/{model_name}/probing_models/aspect/l{aspect_layer}/probe_model_sum_norm.joblib"
    )
    clf_fulltense = joblib.load(
        f"results/{model_name}/probing_models/full_tense/l{fulltense_layer}/probe_model_sum_norm.joblib"
    )
    return {"tense": clf_tense, "aspect": clf_aspect, "full_tense": clf_fulltense}


def prepare_features(samples, probe_layers, layer_means, model, tokenizer):

    batch_size = 20

    summed_embeddings = {target: [] for target in probe_layers.keys()}

    for i in tqdm(range(0, len(samples), batch_size)):
        batch_prompts = samples[i : i + batch_size]
        batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(
            model.device
        )

        with torch.no_grad():

            outputs = model(**batch_inputs, output_hidden_states=True)

            # Mask out padded tokens
            attention_mask = (
                batch_inputs.attention_mask.bool()
            )  # Shape: (batch_size, seq_len)

            for target, target_layer_idx in probe_layers.items():
                layer_embeddings = outputs.hidden_states[
                    target_layer_idx
                ]  # Shape: (batch_size, seq_len, hidden_dim)

                # Collect unpadded embeddings and sum across valid tokens
                summed_per_sequence = [
                    layer_embeddings[i, attention_mask[i]].sum(dim=0)
                    / torch.sqrt(
                        torch.sum(attention_mask[i])
                    )  # Sum over token dimension and normalize
                    for i in range(layer_embeddings.shape[0])
                ]

                # Store results (move to CPU if needed)
                summed_embeddings[target].extend(
                    [emb.detach().cpu().numpy() for emb in summed_per_sequence]
                )

                """for emb in summed_embeddings[target]:
                    if np.isnan(emb).any():
                        print("emb with nan: ", np.where(np.isnan(emb)))"""

                del layer_embeddings
            del outputs, attention_mask, batch_inputs
            empty_cache()

    X_dict = {}
    for target, summed_emb in summed_embeddings.items():
        X_dict[target] = (
            np.array(summed_emb) - layer_means[target][probe_layers[target]]
        )

    return X_dict


def is_complete_sentence(sentence, nlp):
    doc = nlp(sentence)

    for sent in doc.sentences:
        verbs = []
        for word in sent.words:

            if word.upos in ["AUX", "VERB"]:
                verbs.append(word.text)

        if len(verbs) > 0:
            return True

    return False


def measure_text_diversity(text):
    text = text.strip("\n").strip()
    token_list = text.strip().split()
    ngram_list = [2, 3, 4]

    res_dict = {n: {"unique": 0, "total": 0} for n in ngram_list}

    for n in ngram_list:
        if len(token_list) < n:
            continue

        start_idx, end_idx = 0, n
        ngram_set = set()

        while end_idx <= len(token_list):
            one_ngram_list = token_list[start_idx:end_idx]
            assert len(one_ngram_list) == n

            one_ngram = " ".join(one_ngram_list)
            res_dict[n]["total"] += 1
            ngram_set.add(one_ngram)

            start_idx += 1
            end_idx += 1

        res_dict[n]["unique"] = len(ngram_set)

    # Calculate sequence repetition rates with error handling
    def calculate_repetition(n):
        if res_dict[n]["total"] == 0:
            return 0  # No n-grams could be created
        return round((1 - (res_dict[n]["unique"] / res_dict[n]["total"])), 2)

    def unigram_repetition(token_list):
        if not token_list:
            return 0
        token_counts = Counter(token_list)
        most_common_token_count = token_counts.most_common(1)[0][1]
        return round(most_common_token_count / len(token_list), 2)

    seq_1 = unigram_repetition(text)
    seq_2 = calculate_repetition(2)
    seq_3 = calculate_repetition(3)
    seq_4 = calculate_repetition(4)

    # Calculate diversity score, handling potential zero values
    diversity_components = [
        (1 - seq_2) if seq_2 > 0 else 1,
        (1 - seq_3) if seq_3 > 0 else 1,
        (1 - seq_4) if seq_4 > 0 else 1,
    ]

    diversity_score = 1
    for component in diversity_components:
        diversity_score *= component

    return {
        "1-gram-rep": seq_1,
        "2-gram-rep": seq_2,
        "3-gram-rep": seq_3,
        "4-gram-rep": seq_4,
        "diversity": diversity_score,
    }


def eval_layer_steering(
    layer_idx,
    layer_means,
    model,
    model_name,
    tokenizer,
    steering_keys,
    alpha_values,
    clf_dict,
    label_dict_tense,
    label_dict_aspect,
    label_dict_full_tense,
    unsteered_pp,
    probe_layers,
    file_suffix,
    nlp,
):

    # load csv_files
    root_dir = f"results/{model_name}/lda_steering"
    for alpha in tqdm(alpha_values):
        for sk in tqdm(steering_keys):
            file_path = f"{root_dir}/{sk}/{alpha}/{layer_idx}_results_{file_suffix}.csv"
            output_path = file_path.replace(".csv", "_processed.csv")
            if os.path.isfile(output_path):
                print(f"skipping: {file_path}")
                continue
            df = pd.read_csv(file_path)
            df = process_df(
                df,
                clf_dict,
                probe_layers,
                layer_means,
                model,
                tokenizer,
                label_dict_tense,
                label_dict_aspect,
                label_dict_full_tense,
                nlp,
                output_path,
                unsteered_pp,
            )
