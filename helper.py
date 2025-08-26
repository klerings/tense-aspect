import torch
import gc
import gc
import torch
import json
import pickle
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_config(model_name):
    with open(f"configs/{model_name}.json") as f:
        config = json.load(f)
    return config


def get_nnsight_model(config):
    model_name = config["model_name"]
    try:
        model = LanguageModel(model_name, device_map="cuda")
        blocks = getattr(model.config, config["n_layer_param"], None)
    except Exception:
        print(f"model name not covered: {model_name}")
    return model, blocks


def get_model(config):
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    num_layers = getattr(model.config, config["n_layer_param"])
    return model, tokenizer, num_layers


def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()


def load_label_info():
    with open("label_info.json", "r") as f:
        label_info = json.load(f)
    label_info_parsed = {
        "label_mapper": label_info["label_mapper"],
        "label_lookup": label_info["label_lookup"],
    }
    for dname, d in label_info.items():
        if dname not in ["label_mapper", "label_lookup"]:
            d_new = {int(k): v for k, v in d.items()}
            label_info_parsed[dname] = d_new
    return label_info_parsed


def load_steering_hyperparams():
    with open("hyperparams_steering.json", "r") as f:
        steering_params = json.load(f)
    return steering_params


def get_outputs(model, tokenizer, batch_inputs, greedy=True):
    if greedy:
        outputs = model.generate(
            **batch_inputs,
            max_new_tokens=100,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    else:
        outputs = model.generate(
            **batch_inputs,
            max_new_tokens=100,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    return outputs


def load_steering_vectors(layer_idx, steering_keys, model_name, model):
    try:
        with open(
            f"results/{model_name}/feature_vectors/l{layer_idx}/target_dirs_full.pkl",
            "rb",
        ) as f:
            target_dirs_full = pickle.load(f)
        sv_per_feature = {
            feature_name: (sv["lda"] / torch.norm(sv["lda"])).to(model.device)
            for feature_name, sv in target_dirs_full.items()
            if feature_name in steering_keys
        }

        # add random direction
        random_vec = torch.randn(model.config.hidden_size)
        sv_per_feature["random"] = (random_vec / torch.norm(random_vec)).to(
            model.device
        )
    except FileNotFoundError:
        print(
            f"Warning: Steering vectors for layer {layer_idx} not found. Skipping this layer."
        )

    return sv_per_feature
