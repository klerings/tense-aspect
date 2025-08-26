import torch
import numpy as np
import random
from transformers import set_seed
from sentence_transformers import SentenceTransformer, util
from bert_score import BERTScorer
import pickle
from tqdm import tqdm
import pandas as pd
import os
import sys
import stanza
from collections import defaultdict
from helper import (
    read_config,
    empty_cache,
    load_label_info,
    get_model,
    load_steering_hyperparams,
    get_outputs,
    load_steering_vectors,
)
from probing import load_data
from evaluate_generations import (
    load_clfs,
    prepare_features,
    measure_text_diversity,
    is_complete_sentence,
)
from hooks import (
    make_add_constant_hook_dynamic,
    make_add_constant_hook_fixed,
    make_add_constant_hook_first_step_only,
    make_add_constant_hook_decreasing,
    make_add_constant_hook_every_second,
    make_subtract_tense_specific_hook,
    make_subtract_proj_tense_specific_hook,
    get_layer_norm_hook,
)

from create_few_shot_data import (
    get_repetition_dataset,
    get_translation_dataset_tense,
    get_translation_dataset_aspect,
)


def get_unsteered_outputs(
    batches, tokenizer, model, layer_indices, task, prompt_style, greedy=True
):
    results = []

    activation_norms = defaultdict(list)
    activation_norms_last = defaultdict(list)

    current_attention_mask = None

    # do a forward pass for the hooks to capture activations and compute layer norms
    for b in batches:
        if task == "repetition":
            (batch_samples, batch_prompts, batch_inputs, _) = b
        elif task == "translation":
            (batch_samples, batch_prompts, batch_inputs, _, _, _) = b

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

    base_output_dir = (
        f"results/{model_name}/steering_experiments/{task}_task_{prompt_style}"
    )
    os.makedirs(base_output_dir, exist_ok=True)

    with open(f"{base_output_dir}/layer_norm_mean.pkl", "wb") as f:
        pickle.dump(avg_norms, f)

    with open(f"{base_output_dir}/layer_norm_last.pkl", "wb") as f2:
        pickle.dump(avg_last_token_norms, f2)

    # generate unsteered outputs
    for b in batches:
        if task == "repetition":
            (batch_samples, batch_prompts, batch_inputs, _) = b
        elif task == "translation":
            (batch_samples, batch_prompts, batch_inputs, _, _, batch_solutions) = b

        with torch.no_grad():
            input_lengths = [len(ids) for ids in batch_inputs.input_ids]
        outputs = get_outputs(model, tokenizer, batch_inputs, greedy)

        # Extract only the newly generated tokens
        answers = []
        for i, length in enumerate(input_lengths):
            new_tokens = outputs.sequences[i, length:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Only look at first next sentence
            answer = answer.split("\n")[0]
            if prompt_style == "b":
                answer = answer.split(".")[0] + "."
            answers.append(answer.strip())
        for i in range(len(answers)):
            result = {
                "sent": batch_samples[i].sent,
                "few-shot prompt": batch_prompts[i],
                "output": answers[i],
                "source_tense": batch_samples[i].tense,
                "source_aspect": batch_samples[i].aspect,
                "feature_name": None,
            }
            if task == "translation":
                result["solution"] = batch_solutions[i]
            results.append(result)
    unsteered_df = pd.DataFrame(results)

    return unsteered_df


def get_steered_outputs(
    batches,
    tokenizer,
    model,
    model_name,
    layer_idx,
    alpha,
    feature_name,
    sv_per_feature,
    file_suffix,
    task,
    prompt_style,
    greedy=True,
):

    results = []

    if model_name == "gpt2_137m":
        submodule = model.transformer.h[layer_idx]
    else:
        submodule = model.model.layers[layer_idx]

    if file_suffix == "final_token_greedy":
        hook = submodule.register_forward_hook(
            make_add_constant_hook_dynamic(sv_per_feature[feature_name], alpha, "final")
        )
    elif file_suffix in ["final_token_greedy_fixed", "final_token_nongreedy_fixed"]:
        hook = submodule.register_forward_hook(
            make_add_constant_hook_fixed(sv_per_feature[feature_name], alpha, "final")
        )
    elif file_suffix == "first_token_greedy":
        hook = submodule.register_forward_hook(
            make_add_constant_hook_dynamic(sv_per_feature[feature_name], alpha, "first")
        )
    elif file_suffix == "first_token_greedy_fixed":
        hook = submodule.register_forward_hook(
            make_add_constant_hook_fixed(sv_per_feature[feature_name], alpha, "first")
        )
    elif file_suffix == "first_step_only":
        hook = submodule.register_forward_hook(
            make_add_constant_hook_first_step_only(sv_per_feature[feature_name], alpha)
        )
    elif file_suffix == "decreasing_intensity":
        # You can adjust the decay factor as needed
        hook = submodule.register_forward_hook(
            make_add_constant_hook_decreasing(
                sv_per_feature[feature_name], alpha, decay_factor=0.9
            )
        )
    elif file_suffix == "every_second_step":
        hook = submodule.register_forward_hook(
            make_add_constant_hook_every_second(sv_per_feature[feature_name], alpha)
        )
    elif file_suffix in ["subtract_and_add", "subtract_and_add_nongreedy"]:
        hook_fn, set_batch_tenses = make_subtract_tense_specific_hook(
            sv_per_feature[feature_name], sv_per_feature, feature_name, alpha, "final"
        )
        hook = submodule.register_forward_hook(hook_fn)
    elif file_suffix == "subtract_proj":
        hook_fn, set_batch_tenses = make_subtract_proj_tense_specific_hook(
            sv_per_feature[feature_name], sv_per_feature, feature_name, alpha, "final"
        )
        hook = submodule.register_forward_hook(hook_fn)
    else:
        raise Exception(f"file suffix {file_suffix} not implemented")

    for b in batches:
        if task == "repetition":
            (batch_samples, batch_prompts, batch_inputs, batch_tenses) = b
        elif task == "translation":
            (
                batch_samples,
                batch_prompts,
                batch_inputs,
                batch_tenses,
                batch_targets,
                batch_solutions,
            ) = b
        with torch.no_grad():
            input_lengths = [len(ids) for ids in batch_inputs.input_ids]

            if file_suffix in [
                "subtract_and_add",
                "subtract_proj",
                "subtract_and_add_nongreedy",
            ]:
                set_batch_tenses(batch_tenses)

            outputs = get_outputs(model, tokenizer, batch_inputs, greedy)

        # Extract only the newly generated tokens
        answers = []
        for i, length in enumerate(input_lengths):
            new_tokens = outputs.sequences[i, length:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Only look at first next sentence
            answer = answer.split("\n")[0]
            answer = answer.split(".")[0] + "."
            if prompt_style == "a":
                answer = answer.split("\\")[0]
            answers.append(answer.strip())
        for i in range(len(answers)):
            result = {
                "sent": batch_samples[i].sent,
                "few-shot prompt": batch_prompts[i],
                "output": answers[i],
                "source_tense": batch_samples[i].tense,
                "source_aspect": batch_samples[i].aspect,
                "feature_name": feature_name,
            }
            if task == "translation":
                result["steer_target"] = batch_targets[i]
                result["solution"] = batch_solutions[i]
            results.append(result)
    hook.remove()
    steered_df = pd.DataFrame(results)
    return steered_df


def compute_steering_metrics(
    df,
    target,
    task,
    label_dict_tense,
    label_dict_aspect,
    correct_indices,
    unsteered_output_embs,
    unsteered_outputs,
    emb_model, 
    bert_scorer
):
    """
    Compute steering-specific metrics when target is provided.
    """
    metrics = {}

    df["unsteered_output"] = unsteered_outputs

    # Determine which column we're steering and map source columns
    if target in label_dict_tense.values():
        col = "tense"
        other_col = "aspect"
        df[f"source_{col}_new"] = df[f"source_{col}"].map(label_dict_tense)
        df[f"source_{other_col}_new"] = df[f"source_{other_col}"].map(label_dict_aspect)
    elif target in label_dict_aspect.values():
        col = "aspect"
        other_col = "tense"
        df[f"source_{col}_new"] = df[f"source_{col}"].map(label_dict_aspect)
        df[f"source_{other_col}_new"] = df[f"source_{other_col}"].map(label_dict_tense)
    else:
        raise Exception(f"{target} not defined")

    # only consider samples that are correct when unsteered (use indices)
    filtered_df1 = df[correct_indices["correct"]]

    if task == "repetition":
        # only consider samples where source concept != target concept
        filtered_df2 = filtered_df1[
            filtered_df1[f"source_{col}_new"] != filtered_df1["feature_name"]
        ]
    elif task == "translation":
        # only consider samples where target concept == "steer_target"
        filtered_df2 = filtered_df1[
            filtered_df1[f"steer_target"] == filtered_df1["feature_name"]
        ]
    else:
        raise Exception(f"Unknown task: {task}")

    if len(filtered_df2) == 0:
        metrics["degenerate_rate"] = 1
        metrics["success_rate"] = 0
        metrics["success_rate_strict"] = 0
        metrics["selectivity"] = 0
        metrics["selectivity_strict"] = 0
        metrics["topic_shift_strict"] = 0
        return metrics

    # quality filters
    complete_sent_filter = filtered_df2["complete_sent"]
    diversity_filter = filtered_df2["diversity"] > 0.5
    repetition_filter = (filtered_df2["4-gram-rep"] < 0.2) & (
        filtered_df2["2-gram-rep"] < 0.3
    )
    unigram_filter = filtered_df2["1-gram-rep"] < 0.25
    wellformed_filter = (
        complete_sent_filter & diversity_filter & repetition_filter & unigram_filter
    )

    metrics["degenerate_rate"] = (
        len(filtered_df2) - len(filtered_df2[wellformed_filter])
    ) / len(filtered_df2)

    metrics["success_rate"] = len(
        filtered_df2[filtered_df2["feature_name"] == filtered_df2[col]]
    ) / len(filtered_df2)

    metrics["success_rate_strict"] = len(
        filtered_df2[
            (filtered_df2["feature_name"] == filtered_df2[col]) & wellformed_filter
        ]
    ) / len(filtered_df2)

    metrics["selectivity"] = len(
        filtered_df2[
            (filtered_df2[f"source_{other_col}_new"] == filtered_df2[other_col])
        ]
    ) / len(filtered_df2)

    metrics["selectivity_strict"] = len(
        filtered_df2[
            (filtered_df2["feature_name"] == filtered_df2[col])
            & (filtered_df2[f"source_{other_col}_new"] == filtered_df2[other_col])
            & wellformed_filter
        ]
    ) / len(filtered_df2)

    # compute cosine similarity for topic shift
    filtered_indices = filtered_df2.index
    unsteered_output_embs_filtered = unsteered_output_embs[filtered_indices]
    steered_output_embs_filtered = emb_model.encode(
        filtered_df2["output"].values, convert_to_tensor=True
    )
    sim = util.cos_sim(
        unsteered_output_embs_filtered, steered_output_embs_filtered
    ).diagonal()
    filtered_df2[f"cosine_sim"] = sim.cpu().numpy()

    successful_steering_mask = filtered_df2["feature_name"] == filtered_df2[col]
    df_success = filtered_df2[successful_steering_mask]
    if successful_steering_mask.any():
        metrics["topic_shift_strict"] = df_success[
            "cosine_sim"
        ].mean()
    else:
        metrics["topic_shift_strict"] = 0.0
    
    references = df_success["unsteered_output"].str.strip().str.strip('"').values.tolist()
    candidates = df_success["output"].str.strip().str.strip('"').values.tolist()
    if len(candidates) != 0:
        bert_score_precision, bert_score_recall, bert_score = bert_scorer.score(candidates, references)

        metrics["BERTScore_Pr"] = bert_score_precision.mean().item()
        metrics["BERTScore_Rec"] = bert_score_recall.mean().item()
        metrics["BERTScore"] = bert_score.mean().item()
    else:
        metrics["BERTScore_Pr"] = 0
        metrics["BERTScore_Rec"] = 0
        metrics["BERTScore"] = 0

    return metrics


def eval_generations_base(
    df, clf_dict, nlp, label_dict_tense, label_dict_aspect, probe_layers
):
    """
    Base evaluation function that computes quality metrics and linguistic predictions.
    """
    # Initialize columns
    for col_name in [
        "tense",
        "aspect",
        "diversity",
        "4-gram-rep",
        "2-gram-rep",
        "3-gram-rep",
        "1-gram-rep",
    ]:
        df[col_name] = "unknown"
    df["complete_sent"] = False

    # Compute quality metrics
    quality_results = df["output"].apply(lambda x: measure_text_diversity(x))
    for metric in ["diversity", "1-gram-rep", "2-gram-rep", "3-gram-rep", "4-gram-rep"]:
        df[metric] = quality_results.apply(lambda x: x[metric])

    # Check sentence completeness
    df["complete_sent"] = df["output"].apply(lambda x: is_complete_sentence(x, nlp))

    # Make linguistic predictions for non-empty outputs
    non_empty_mask = df["output"] != ""
    df_non_empty = df[non_empty_mask]

    if not df_non_empty.empty:
        X_dict = prepare_features(
            list(df_non_empty["output"].values),
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
    task,
    clf_dict,
    nlp,
    label_dict_tense,
    label_dict_aspect,
    probe_layers,
    target=None,
    correct_indices=None,
    unsteered_output_embs=None,
    unsteered_outputs=None,
    emb_model=None, 
    bert_scorer=None,
):
    """
    Main evaluation function that handles both steering and non-steering cases.
    """
    # Always run base evaluation
    df = eval_generations_base(
        df, clf_dict, nlp, label_dict_tense, label_dict_aspect, probe_layers
    )

    metrics = {}

    if target is None:
        # Case 1: Non-steering evaluation - compute task accuracy
        if task == "repetition":
            correct_col_name = "sent"
        elif task == "translation":
            correct_col_name = "solution"
        else:
            raise Exception(f"Unknown task: {task}")

        metrics["acc"] = len(df[df[correct_col_name] == df["output"]]) / len(df)
        df["correct"] = df[correct_col_name] == df["output"]
    else:
        # Case 2: Steering evaluation - compute steering metrics
        if correct_indices is None or unsteered_output_embs is None:
            raise ValueError(
                "correct_indices and unsteered_output_embs must be provided when target is specified"
            )

        metrics = compute_steering_metrics(
            df,
            target,
            task,
            label_dict_tense,
            label_dict_aspect,
            correct_indices,
            unsteered_output_embs,
            unsteered_outputs,
            emb_model,
            bert_scorer,
        )

    return df, metrics


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
    task = sys.argv[4]
    prompt_style = sys.argv[5]
    target = sys.argv[6]

    # load dataset
    test_samples = load_data(sources=["bigbench"])

    # load model
    config = read_config(model_name)
    model, tokenizer, num_layers = get_model(config)
    layer_indices = list(range(num_layers))

    # load steering labels
    label_info = load_label_info()
    label_dict_tense = label_info["label_dict_tense"]
    label_dict_aspect = label_info["label_dict_aspect"]

    # set steering configuration
    greedy = True
    if "nongreedy" in file_suffix:
        greedy = False
    steering_hyperparams = load_steering_hyperparams()
    steering_config = steering_hyperparams[model_name]["steering_config_few_shot"][
        target
    ]
    steering_keys = list(label_info["label_lookup"][target].keys())

    # prepare evaluation through NLP tools
    nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse")
    emb_model = SentenceTransformer("all-mpnet-base-v2")

    bert_scorer = BERTScorer(lang="en", model_type="microsoft/deberta-xlarge-mnli", rescale_with_baseline=False)

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

    # prepare output dir
    base_output_dir = (
        f"results/{model_name}/steering_experiments/{task}_task_{prompt_style}"
    )
    os.makedirs(f"{base_output_dir}/unsteered", exist_ok=True)

    # generate outputs without steering
    if task == "repetition":
        batches = get_repetition_dataset(
            test_samples,
            tokenizer,
            model,
            label_info,
            batch_size=batch_size,
            prompt_style=prompt_style,
        )
    elif task == "translation":
        if target == "tense":
            batches = get_translation_dataset_tense(
                test_samples,
                tokenizer,
                model,
                label_info,
                batch_size=batch_size,
                prompt_style=prompt_style,
            )
        elif target == "aspect":
            batches = get_translation_dataset_aspect(
                test_samples,
                tokenizer,
                model,
                label_info,
                batch_size=batch_size,
                prompt_style=prompt_style,
            )
    else:
        raise Exception(f"{task} is not a valid task.")

    unsteered_df = get_unsteered_outputs(
        batches, tokenizer, model, layer_indices, task, prompt_style, greedy
    )

    # evaluate unsteered task performance
    unsteered_df_processed, unsteered_metrics = eval_generations(
        unsteered_df,
        task,
        clf_dict,
        nlp,
        label_dict_tense,
        label_dict_aspect,
        probe_layers,
    )
    if greedy:
        unsteered_df_processed.to_csv(
            f"{base_output_dir}/unsteered/results_greedy.csv",
            index=False,
        )
        with open(f"{base_output_dir}/unsteered/results_greedy.pkl", "wb") as f1:
            pickle.dump(unsteered_df_processed, f1)
    else:
        unsteered_df_processed.to_csv(
            f"{base_output_dir}/unsteered/results_{file_suffix}.csv",
            index=False,
        )
        with open(f"{base_output_dir}/unsteered/results_{file_suffix}.pkl", "wb") as f2:
            pickle.dump(unsteered_df_processed, f2)

    # get embeddings of unsteered outputs for computing topic shift later
    unsteered_outputs = unsteered_df_processed["output"].values
    unsteered_output_embs = emb_model.encode(unsteered_outputs, convert_to_tensor=True)

    # perform gridsearch for steering
    for layer_idx in steering_config["layer_idx"]:
        sv_per_feature = load_steering_vectors(
            layer_idx, steering_keys, model_name, model
        )
        for alpha in tqdm(steering_config["alphas"]):
            all_metrics = {}

            for sk in steering_keys:
                result_dir = f"{base_output_dir}/{sk}/{alpha}"
                os.makedirs(result_dir, exist_ok=True)

                steered_df = get_steered_outputs(
                    batches,
                    tokenizer,
                    model,
                    model_name,
                    layer_idx,
                    alpha,
                    sk,
                    sv_per_feature,
                    file_suffix,
                    task,
                    prompt_style,
                    greedy,
                )

                # evaluate outputs
                correct_indices = unsteered_df_processed
                steered_df_processed, steered_metrics = eval_generations(
                    steered_df,
                    task,
                    clf_dict,
                    nlp,
                    label_dict_tense,
                    label_dict_aspect,
                    probe_layers,
                    sk,
                    correct_indices,
                    unsteered_output_embs,
                    unsteered_outputs,
                    emb_model, 
                    bert_scorer,
                )

                steered_df_processed.to_csv(
                    f"{base_output_dir}/{sk}/{alpha}/results_L={layer_idx}_{file_suffix}.csv",
                    index=False,
                )
                with open(
                    f"{base_output_dir}/{sk}/{alpha}/results_L={layer_idx}_{file_suffix}.pkl",
                    "wb",
                ) as f3:
                    pickle.dump(steered_df_processed, f3)
                all_metrics[sk] = steered_metrics
                print(f"Steering [{sk}] at L{layer_idx}, a={alpha}:\n{steered_metrics}")

            # compute mean for target (without random)
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

            all_metrics["unsteered_task_acc"] = unsteered_metrics["acc"]

            out_file = f"{base_output_dir}/all_metrics_{target}_L={layer_idx}_alpha={alpha}_{file_suffix}.pkl"
            with open(out_file, "wb") as f:
                pickle.dump(all_metrics, f)
        print(f"finished: {target}")
