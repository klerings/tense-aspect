import pickle
import json
import sys
import numpy as np


def identify_best_probes(model_name, agg):
    probes_per_target = {}
    for target in ["tense", "aspect", "full_tense"]:
        with open(f"results/{model_name}/full_layer_probe_{target}.pkl", "rb") as f:
            probe_results = pickle.load(f)[agg]
            best_layer = int(np.argmax(probe_results))
            best_probe = probe_results[best_layer]
            probes_per_target[target] = best_layer
            print(f"{target}: L{best_layer} -> {best_probe}")
    return probes_per_target


def update_steering_config(model_name, probes_per_target):
    filename = "hyperparams_steering.json"
    with open(filename, "r") as f:
        config = json.load(f)
        print(config)
    with open(filename, "w") as f2:
        if model_name in config:
            raise Exception(f"Settings for {model_name} already set")
        else:
            config[model_name] = {"probe_layers": probes_per_target}
            print(config)
            json.dump(config, f2, indent=4)


if __name__ == "__main__":
    model_name = sys.argv[1]
    probes_per_target = identify_best_probes(model_name, "sum_norm")
    update_steering_config(model_name, probes_per_target)
