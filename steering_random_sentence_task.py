import torch
import numpy as np
import random
from transformers import set_seed
from sentence_transformers import SentenceTransformer, util
from bert_score import BERTScorer
import pickle
import stanza
from tqdm import tqdm
import time
import pandas as pd
from collections import defaultdict
import os
import sys
import gc
from helper import (
    read_config,
    empty_cache,
    load_label_info,
    get_model,
    get_outputs,
    load_steering_hyperparams,
    load_steering_vectors,
)
from hooks import (
    make_add_constant_hook_dynamic,
    make_add_constant_hook_fixed,
    make_add_constant_hook_first_step_only,
    make_add_constant_hook_decreasing,
    make_add_constant_hook_every_second,
    get_layer_norm_hook,
)
from evaluate_generations import (
    load_clfs,
    prepare_features,
    measure_text_diversity,
    is_complete_sentence,
)


def get_unsteered_outputs(
    model,
    tokenizer,
    prompts,
    model_name,
    greedy,
    file_suffix,
    clf_dict,
    probe_layers,
    layer_means,
    nlp,
    label_dict_tense,
    label_dict_aspect,
    batch_size=1,
):

    unsteered_results = {}

    activation_norms = defaultdict(list)
    activation_norms_last = defaultdict(list)

    current_attention_mask = None

    # do a forward pass for the hooks to capture activations and compute layer norms
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i : i + batch_size]
        batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(
            model.device
        )
        current_attention_mask = batch_inputs.attention_mask

        # register hooks
        hooks = []
        for layer_idx in layer_indices:
            if model_name == "gpt2_137m":
                submodule = model.transformer.h[layer_idx]
            else:
                submodule = model.model.layers[layer_idx]
            hooks.append(
                submodule.register_forward_hook(
                    get_layer_norm_hook(
                        f"layer_{layer_idx}",
                        activation_norms,
                        activation_norms_last,
                        current_attention_mask,
                    )
                )
            )

        with torch.no_grad():
            _ = model(**batch_inputs)

        # Remove hooks after the forward pass
        for hook in hooks:
            hook.remove()

    avg_norms = {
        layer: sum(norms) / len(norms) for layer, norms in activation_norms.items()
    }

    avg_last_token_norms = {
        layer: sum(norms) / len(norms) for layer, norms in activation_norms_last.items()
    }

    base_output_dir = f"results/{model_name}/steering_experiments/random_task"
    os.makedirs(base_output_dir, exist_ok=True)

    with open(f"{base_output_dir}/layer_norm_mean.pkl", "wb") as f:
        pickle.dump(avg_norms, f)

    with open(f"{base_output_dir}/layer_norm_last.pkl", "wb") as f2:
        pickle.dump(avg_last_token_norms, f2)

    # generate unsteered output
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i : i + batch_size]
        batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(
            model.device
        )
        with torch.no_grad():
            input_lengths = [len(ids) for ids in batch_inputs.input_ids]
            outputs = get_outputs(model, tokenizer, batch_inputs, greedy)

            generated_sequences = outputs.sequences
            scores = outputs.scores  # logits for each step

            for j, prompt in enumerate(batch_prompts):
                input_length = input_lengths[j]
                new_tokens = generated_sequences[j][input_length:]
                sent = tokenizer.decode(new_tokens, skip_special_tokens=True)
                sent = sent.split("\n")[0].strip()

                # compute perplexity
                log_probs = []
                for step, token_id in enumerate(new_tokens):
                    token_log_probs = torch.nn.functional.log_softmax(
                        scores[step], dim=1
                    )
                    log_prob = token_log_probs[
                        j, token_id
                    ].item()  # get log probability of chosen token
                    log_probs.append(log_prob)

                avg_log_prob = sum(log_probs) / len(log_probs) if log_probs else 0
                perplexity = (
                    torch.exp(torch.tensor(-avg_log_prob)).item()
                    if log_probs
                    else float("inf")
                )

                unsteered_results[prompt] = {"text": sent, "perplexity": perplexity}

    # Save unsteered results separately
    base_output_dir = f"results/{model_name}/steering_experiments/random_task"
    os.makedirs(f"{base_output_dir}/unsteered", exist_ok=True)

    unsteered_df = pd.DataFrame(
        [
            {
                "prompt": prompt,
                "output": unsteered_results[prompt]["text"],
                "perplexity": unsteered_results[prompt]["perplexity"],
                "is_steered": False,
            }
            for prompt in prompts
        ]
    )

    # evaluate generations
    unsteered_df_processed = eval_generations(
        unsteered_df,
        None,
        None,
        None,
        clf_dict,
        probe_layers,
        layer_means,
        model,
        tokenizer,
        label_dict_tense,
        label_dict_aspect,
        nlp,
        None,
    )

    if greedy:
        unsteered_df_processed.to_csv(
            f"{base_output_dir}/unsteered/results_greedy.csv",
            index=False,
        )
        with open(f"{base_output_dir}/unsteered/results_greedy.pkl", "wb") as f:
            pickle.dump(unsteered_results, f)
    else:
        unsteered_df_processed.to_csv(
            f"{base_output_dir}/unsteered/results_{file_suffix}.csv",
            index=False,
        )
        with open(f"{base_output_dir}/unsteered/results_{file_suffix}.pkl", "wb") as f:
            pickle.dump(unsteered_results, f)

    return unsteered_df_processed


def process_with_parameter_variations(
    model,
    tokenizer,
    prompts,
    model_name,
    greedy,
    file_suffix,
    process_layers,
    steering_keys,
    target,
    alpha_values,
    clf_dict,
    probe_layers,
    layer_means,
    nlp,
    emb_model,
    bert_scorer,
    label_dict_tense,
    label_dict_aspect,
    unsteered_df_processed,
    unsteered_output_embs,
    batch_size=1,
):

    all_results = {}
    base_output_dir = f"results/{model_name}/steering_experiments/random_task"

    for layer_idx in process_layers:
        sv_per_feature = load_steering_vectors(
            layer_idx, steering_keys, model_name, model
        )
        for alpha in alpha_values:
            all_metrics = {}
            for sk in steering_keys:

                # Save to structured directory
                output_dir = f"{base_output_dir}/{sk}/{alpha}"
                os.makedirs(output_dir, exist_ok=True)

                out_file = f"{output_dir}/{layer_idx}_results_{file_suffix}.csv"
                out_file_pkl = f"{output_dir}/{layer_idx}_results_{file_suffix}.pkl"
                if os.path.isfile(out_file):
                    print(f"skipping: {out_file}")
                    continue

                variation_key = f"l{layer_idx}_{sk}_a{alpha}"
                all_results[variation_key] = {}

                if model_name == "gpt2_137m":
                    submodule = model.transformer.h[layer_idx]
                else:
                    submodule = model.model.layers[layer_idx]

                if file_suffix == "final_token_greedy":
                    hook = submodule.register_forward_hook(
                        make_add_constant_hook_dynamic(
                            sv_per_feature[sk], alpha, "final"
                        )
                    )
                elif file_suffix in [
                    "final_token_greedy_fixed",
                    "final_token_nongreedy_fixed",
                ]:
                    hook = submodule.register_forward_hook(
                        make_add_constant_hook_fixed(sv_per_feature[sk], alpha, "final")
                    )
                elif file_suffix == "first_step_only":
                    hook = submodule.register_forward_hook(
                        make_add_constant_hook_first_step_only(
                            sv_per_feature[sk], alpha
                        )
                    )
                elif file_suffix == "decreasing_intensity":
                    hook = submodule.register_forward_hook(
                        make_add_constant_hook_decreasing(
                            sv_per_feature[sk],
                            alpha,
                            decay_factor=0.9,
                        )
                    )
                elif file_suffix == "every_second_step":
                    hook = submodule.register_forward_hook(
                        make_add_constant_hook_every_second(sv_per_feature[sk], alpha)
                    )

                for i in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[i : i + batch_size]
                    batch_inputs = tokenizer(
                        batch_prompts, return_tensors="pt", padding=True
                    ).to(model.device)
                    with torch.no_grad():
                        input_lengths = [len(ids) for ids in batch_inputs.input_ids]
                        outputs = get_outputs(model, tokenizer, batch_inputs, greedy)

                        generated_sequences = outputs.sequences
                        scores = outputs.scores  # logits for each step

                        for j, prompt in enumerate(batch_prompts):
                            input_length = input_lengths[j]
                            new_tokens = generated_sequences[j][input_length:]
                            sent = tokenizer.decode(
                                new_tokens, skip_special_tokens=True
                            )
                            sent = sent.split("\n")[0].strip()

                            # compute perplexity
                            log_probs = []
                            for step, token_id in enumerate(new_tokens):
                                token_log_probs = torch.nn.functional.log_softmax(
                                    scores[step], dim=1
                                )
                                log_prob = token_log_probs[
                                    j, token_id
                                ].item()  # get log probability of chosen token
                                log_probs.append(log_prob)

                            avg_log_prob = (
                                sum(log_probs) / len(log_probs) if log_probs else 0
                            )
                            perplexity = (
                                torch.exp(torch.tensor(-avg_log_prob)).item()
                                if log_probs
                                else float("inf")
                            )

                            all_results[variation_key][prompt] = {
                                "text": sent,
                                "perplexity": perplexity,
                            }

                hook.remove()

                variation_results = all_results[variation_key]
                variation_df = pd.DataFrame(
                    [
                        {
                            "prompt": prompt,
                            "layer_idx": layer_idx,
                            "alpha": alpha,
                            "output": variation_results[prompt]["text"],
                            "perplexity": variation_results[prompt]["perplexity"],
                            "is_steered": True,
                            "steering_key": sk,
                        }
                        for prompt in prompts
                        if prompt in variation_results
                    ]
                )

                # evaluate generations
                processed_df, metrics = eval_generations(
                    variation_df,
                    unsteered_df_processed,
                    unsteered_output_embs,
                    sk,
                    clf_dict,
                    probe_layers,
                    layer_means,
                    model,
                    tokenizer,
                    label_dict_tense,
                    label_dict_aspect,
                    nlp,
                    emb_model,
                    bert_scorer,
                )

                # Save as CSV and pickle
                processed_df.to_csv(
                    out_file,
                    index=False,
                )

                with open(out_file_pkl, "wb") as f:
                    pickle.dump(variation_results, f)

                all_metrics[sk] = metrics
                print(f"Steering [{sk}] at L{layer_idx}, a={alpha}:\n{metrics}")

                del all_results[variation_key]
                gc.collect()
                torch.cuda.empty_cache()

            metrics = [
                "success_rate",
                "success_rate_strict",
                "selectivity",
                "selectivity_strict",
                "degenerate_rate",
                "topic_shift_strict",
            ]
            all_metrics["mean"] = {m: [] for m in metrics}
            for sk, m_dict in all_metrics.items():
                if sk not in ["random", "mean"]:
                    for metric in metrics:
                        all_metrics["mean"][metric].append(m_dict[metric])

            for m in metrics:
                all_metrics["mean"][m] = np.mean(np.array(all_metrics["mean"][m]))

            result_file = f"{base_output_dir}/all_metrics_{target}_L={layer_idx}_alpha={alpha}_{file_suffix}.pkl"
            with open(result_file, "wb") as f:
                pickle.dump(all_metrics, f)

    return


def compute_steering_metrics(
    df, unsteered_df, unsteered_output_embs, steering_key, emb_model, bert_scorer,
):
    """
    Compute steering-specific metrics by comparing steered vs unsteered outputs.
    """
    metrics = {}

    # Determine target and source columns based on steering key
    if steering_key in ["past", "present", "future"]:
        col = "tense"
        other_col = "aspect"
    elif steering_key in ["simple", "progressive", "perfect", "perfect_progressive"]:
        col = "aspect"
        other_col = "tense"
    else:
        raise ValueError(f"Unknown steering_key: {steering_key}")

    df["source_nontarget"] = unsteered_df[other_col].values
    df["unsteered_output"] = unsteered_df["output_split"].values

    # Filter to non-empty outputs only
    non_empty_mask = df["output_split"] != ""
    df_non_empty = df[non_empty_mask]

    if df_non_empty.empty:
        # Return default metrics if no valid outputs
        return df, {
            "degenerate_rate": 1.0,
            "success_rate": 0.0,
            "success_rate_strict": 0.0,
            "selectivity": 0.0,
            "selectivity_strict": 0.0,
            "topic_shift_strict": 0.0,
        }

    # Quality filters
    complete_sent_filter = df_non_empty["complete_sent"]
    diversity_filter = df_non_empty["diversity"] > 0.5
    repetition_filter = (df_non_empty["4-gram-rep"] < 0.2) & (
        df_non_empty["2-gram-rep"] < 0.3
    )
    unigram_filter = df_non_empty["1-gram-rep"] < 0.25
    wellformed_filter = (
        complete_sent_filter & diversity_filter & repetition_filter & unigram_filter
    )

    # Basic metrics
    metrics["degenerate_rate"] = (len(df) - len(df_non_empty[wellformed_filter])) / len(
        df
    )
    metrics["success_rate"] = len(
        df_non_empty[df_non_empty[col] == steering_key]
    ) / len(df)
    metrics["success_rate_strict"] = len(
        df_non_empty[(df_non_empty[col] == steering_key) & wellformed_filter]
    ) / len(df)
    metrics["selectivity"] = len(
        df_non_empty[df_non_empty[other_col] == df_non_empty["source_nontarget"]]
    ) / len(df)
    metrics["selectivity_strict"] = len(
        df_non_empty[
            (df_non_empty[col] == steering_key)
            & (df_non_empty[other_col] == df_non_empty["source_nontarget"])
            & wellformed_filter
        ]
    ) / len(df)

    # Compute cosine similarity only for non-empty outputs
    unsteered_output_embs_filtered = unsteered_output_embs[non_empty_mask]
    steered_output_embs = emb_model.encode(
        df_non_empty["output_split"].values, convert_to_tensor=True
    )

    sim = util.cos_sim(unsteered_output_embs_filtered, steered_output_embs).diagonal()
    df_non_empty.loc[:, "cosine_sim"] = sim.cpu().numpy()

    # Topic shift metric (only for successful steering cases)
    successful_steering_mask = df_non_empty[col] == steering_key
    df_success = df_non_empty[successful_steering_mask]
    if successful_steering_mask.any():
        metrics["topic_shift_strict"] = df_success[
            "cosine_sim"
        ].mean()
    else:
        metrics["topic_shift_strict"] = 0.0

    references = df_success["unsteered_output"].str.strip().str.strip('"').values.tolist()
    candidates = df_success["output_split"].str.strip().str.strip('"').values.tolist()
    if len(candidates) != 0:
        bert_score_precision, bert_score_recall, bert_score = bert_scorer.score(candidates, references)

        metrics["BERTScore_Pr"] = bert_score_precision.mean().item()
        metrics["BERTScore_Rec"] = bert_score_recall.mean().item()
        metrics["BERTScore"] = bert_score.mean().item()
    else:
        metrics["BERTScore_Pr"] = 0
        metrics["BERTScore_Rec"] = 0
        metrics["BERTScore"] = 0

    return df, metrics


def eval_generations_base(
    df,
    clf_dict,
    probe_layers,
    layer_means,
    model,
    tokenizer,
    label_dict_tense,
    label_dict_aspect,
    nlp,
):
    """
    Base evaluation function that handles text processing and classification.
    Always returns a processed dataframe.
    """
    # Initialize columns
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

    # Process text
    df.fillna("", inplace=True)
    df["output_split"] = df["output"].apply(lambda x: x.split(".")[0])

    # Quality metrics
    quality_results = df["output_split"].apply(lambda x: measure_text_diversity(x))
    for metric in ["diversity", "1-gram-rep", "2-gram-rep", "3-gram-rep", "4-gram-rep"]:
        df[metric] = quality_results.apply(lambda x: x[metric])

    # Complete sentence check
    df["complete_sent"] = df["output_split"].apply(
        lambda x: is_complete_sentence(x, nlp)
    )

    # Classification for non-empty outputs
    non_empty_mask = df["output_split"] != ""
    df_non_empty = df[non_empty_mask]

    if not df_non_empty.empty:
        X_dict = prepare_features(
            list(df_non_empty["output_split"].values),
            probe_layers,
            layer_means,
            model,
            tokenizer,
        )

        tense_preds = clf_dict["tense"].predict(X_dict["tense"])
        aspect_preds = clf_dict["aspect"].predict(X_dict["aspect"])

        df.loc[df_non_empty.index, "tense"] = [label_dict_tense[p] for p in tense_preds]
        df.loc[df_non_empty.index, "aspect"] = [
            label_dict_aspect[p] for p in aspect_preds
        ]

        del X_dict
        empty_cache()

    return df


def eval_generations(
    df,
    unsteered_df,
    unsteered_output_embs,
    steering_key,
    clf_dict,
    probe_layers,
    layer_means,
    model,
    tokenizer,
    label_dict_tense,
    label_dict_aspect,
    nlp,
    emb_model,
    bert_scorer=None,
):
    """
    Main evaluation function that handles both base evaluation and steering metrics.
    """
    # Always do base evaluation
    df = eval_generations_base(
        df,
        clf_dict,
        probe_layers,
        layer_means,
        model,
        tokenizer,
        label_dict_tense,
        label_dict_aspect,
        nlp,
    )

    # If unsteered comparison data is provided, compute steering metrics
    if (
        unsteered_df is not None
        and unsteered_output_embs is not None
        and steering_key is not None
    ):
        return compute_steering_metrics(
            df, unsteered_df, unsteered_output_embs, steering_key, emb_model, bert_scorer,
        )
    else:
        return df


if __name__ == "__main__":
    # fix random seeds for reproducibility
    seed_value = 10
    set_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.set_grad_enabled(False)

    model_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    file_suffix = sys.argv[3]

    # load dataset
    with open(f"data/open_ended_prompts.txt", "r") as f:
        prompts = [line.strip() for line in f.readlines()]

    # load model
    config = read_config(model_name)
    model, tokenizer, num_layers = get_model(config)
    layer_indices = list(range(num_layers))

    # load steering info
    label_info = load_label_info()
    label_dict_tense = label_info["label_dict_tense"]
    label_dict_aspect = label_info["label_dict_aspect"]
    steering_hyperparams = load_steering_hyperparams()

    # prepare evaluation through probing
    probe_layers = steering_hyperparams[model_name]["probe_layers"]
    clf_dict = load_clfs(model_name, probe_layers)
    layer_means = {}
    for probing_target in ["tense", "aspect", "full_tense"]:
        with open(
            f"results/{model_name}/probing_models/{probing_target}/train_ds_mean_for_centering_sum_norm.pkl",
            "rb",
        ) as f:
            layer_means[probing_target] = pickle.load(f)
            print(f"DS means for {probing_target}")

    # prepare evaluation through NLP tools
    nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse")
    emb_model = SentenceTransformer("all-mpnet-base-v2")

    bert_scorer = BERTScorer(lang="en", model_type="microsoft/deberta-xlarge-mnli", rescale_with_baseline=False)

    # set steering configuration
    greedy = True
    if "nongreedy" in file_suffix:
        greedy = False

    unsteered_df_processed = get_unsteered_outputs(
        model,
        tokenizer,
        prompts,
        model_name,
        greedy,
        file_suffix,
        clf_dict,
        probe_layers,
        layer_means,
        nlp,
        label_dict_tense,
        label_dict_aspect,
        batch_size=batch_size,
    )

    # get embeddings of unsteered outputs for computing topic shift later
    unsteered_outputs = unsteered_df_processed["output_split"].values
    unsteered_output_embs = emb_model.encode(unsteered_outputs, convert_to_tensor=True)

    # perform gridsearch for steering
    for target in ["tense", "aspect"]:
        steering_config = steering_hyperparams[model_name]["steering_config_random"][
            target
        ]
        steering_keys = list(label_info["label_lookup"][target].keys())
        alpha_values = steering_config["alphas"]
        process_layers = steering_config["layer_idx"]

        start_time = time.time()
        process_with_parameter_variations(
            model,
            tokenizer,
            prompts,
            model_name,
            greedy=greedy,
            file_suffix=file_suffix,
            process_layers=process_layers,
            steering_keys=steering_keys,
            target=target,
            alpha_values=alpha_values,
            clf_dict=clf_dict,
            probe_layers=probe_layers,
            layer_means=layer_means,
            nlp=nlp,
            emb_model=emb_model,
            bert_scorer=bert_scorer,
            label_dict_tense=label_dict_tense,
            label_dict_aspect=label_dict_aspect,
            unsteered_df_processed=unsteered_df_processed,
            unsteered_output_embs=unsteered_output_embs,
            batch_size=batch_size,
        )
        print(f"Parameter sweep completed in {time.time() - start_time:.2f} seconds")
