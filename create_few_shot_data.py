import random
from itertools import permutations, cycle
import pandas as pd


def get_repetition_dataset(
    test_samples, tokenizer, model, label_info, batch_size=16, prompt_style="a"
):
    all_test_samples = []
    for source, samples in test_samples.items():
        sents_per_source = [s.sent for s in samples]
        all_test_samples.extend(sents_per_source)
        # print(source, len(sents_per_source))

    batches = []

    label_mapper = {}
    for label_numeric in label_info["label_mapper"].keys():
        tense = label_info["label_dict_tense"][int(label_numeric.split("_")[0])]
        aspect = label_info["label_dict_aspect"][int(label_numeric.split("_")[1])]
        label_mapper[label_numeric] = (tense, aspect)

    for source, sample_list in test_samples.items():
        for i in range(0, len(sample_list), batch_size):
            batch_samples = sample_list[i : i + batch_size]
            batch_prompts = []
            batch_tenses = []
            for test_sample in batch_samples:

                # randomly sample two examples
                few_shot_examples = []
                while len(few_shot_examples) == 0:
                    r_samples = random.sample(all_test_samples, 2)
                    no_duplicate = True
                    for s in r_samples:
                        if s == test_sample.sent:
                            no_duplicate = False
                            break
                    if no_duplicate:
                        few_shot_examples = r_samples

                # create few-shot prompt
                if prompt_style == "a":
                    few_shot_prompt = "\n\n".join(
                        [f"{example} \\\\ {example}" for example in few_shot_examples]
                    )

                    prompt = f"""{few_shot_prompt}\n\n{test_sample.sent} \\\\"""
                elif prompt_style == "b":
                    few_shot_prompt = "\n\n".join(
                        [
                            f"Example {i}:\nInput: {example}\nOutput: {example}"
                            for i, example in enumerate(few_shot_examples, start=1)
                        ]
                    )

                    prompt = f"""Task: Repeat the input text as shown in the examples.\n\n{few_shot_prompt}\n\nNow it's your turn:\n\nInput: {test_sample.sent}\nOutput:"""

                batch_prompts.append(prompt)
                batch_tenses.append(label_mapper[source])

            # for p in batch_prompts:
            #    print(f"{p}\n-----------")
            batch_inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True
            ).to(model.device)

            batches.append((batch_samples, batch_prompts, batch_inputs, batch_tenses))
    return batches


def get_translation_dataset_aspect(test_samples, tokenizer, model, prompt_style="a"):

    filter_df = pd.read_csv(
        "data/raw_data/bigbench_translated_complete.csv", delimiter=";"
    )
    allowed_sents = []
    for i, row in filter_df.iterrows():
        if row.isnull().any():
            continue
        allowed_sents.append(row["Original"])

    batch_size = 16

    batches = []

    label_mapper = {
        "1_2": ("future", "simple"),
        "1_3": ("future", "progressive"),
        "1_4": ("future", "perfect"),
        "1_5": ("future", "perfect_progressive"),
        "0_2": ("present", "simple"),
        "0_3": ("present", "progressive"),
        "0_4": ("present", "perfect"),
        "0_5": ("present", "perfect_progressive"),
        "-1_2": ("past", "simple"),
        "-1_3": ("past", "progressive"),
        "-1_4": ("past", "perfect"),
        "-1_5": ("past", "perfect_progressive"),
    }

    aspects = ["simple", "progressive", "perfect", "perfect_progressive"]
    # Create a dictionary mapping source -> list of (target, steer) permutations
    aspect_combinations = {
        source: list(permutations([a for a in aspects if a != source], 2))
        for source in aspects
    }

    # Convert each list to a cycle iterator to ensure even usage
    aspect_iterators = {k: cycle(v) for k, v in aspect_combinations.items()}

    for source, sample_list in test_samples.items():
        sample_list_filtered = [s for s in sample_list if s.sent in allowed_sents]
        for i in range(0, len(sample_list_filtered), batch_size):
            batch_samples = sample_list_filtered[i : i + batch_size]
            batch_prompts = []
            batch_tenses = []
            batch_steer_targets = []
            batch_solutions = []
            for test_sample in batch_samples:

                source_aspect = label_mapper[source][1]

                # Get (target, steer) pair in a round-robin fashion
                target_aspect, steer_aspect = next(aspect_iterators[source_aspect])

                sent_dict = {
                    "simple": ["I walk through the park.", "Paul visits the school."],
                    "progressive": [
                        "I am walking through the park.",
                        "Paul is visiting the school.",
                    ],
                    "perfect": [
                        "I have walked through the park.",
                        "Paul has visited the school.",
                    ],
                    "perfect_progressive": [
                        "I have been walking through the park.",
                        "Paul has been visiting the school.",
                    ],
                }

                inputs = sent_dict[source_aspect]
                outputs = sent_dict[target_aspect]

                if prompt_style == "a":
                    few_shot_prompt = f"{inputs[0]} \\\\ {outputs[0]}\n\n{inputs[1]} \\\\ {outputs[1]}\n\n{test_sample.sent} \\\\"

                elif prompt_style == "b":
                    few_shot_prompt = f"Task: Modify the input text as shown in the examples.\n\nExample 1:\nInput: {inputs[0]}\nOutput: {outputs[0]}\n\nExample 2:\nInput: {inputs[1]}\nOutput: {outputs[1]}\n\nNow it's your turn:\n\nInput: {test_sample.sent}\nOutput:"

                sol_col = target_aspect
                steer = steer_aspect

                batch_prompts.append(few_shot_prompt)
                batch_tenses.append(label_mapper[source])
                batch_steer_targets.append(steer)
                batch_solutions.append(
                    filter_df[filter_df["Original"] == test_sample.sent][
                        sol_col
                    ].values[0]
                )

            # for p in batch_prompts:
            #    print(f"{p}\n-----------")
            batch_inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True
            ).to(model.device)

            batches.append(
                (
                    batch_samples,
                    batch_prompts,
                    batch_inputs,
                    batch_tenses,
                    batch_steer_targets,
                    batch_solutions,
                )
            )
    return batches


def get_translation_dataset_tense(test_samples, tokenizer, model, prompt_style="a"):

    filter_df = pd.read_csv("data/raw_data/bigbench_translated.csv", delimiter=";")
    allowed_sents = filter_df["Original"].values

    batch_size = 16

    batches = []

    label_mapper = {
        "1_2": ("future", "simple"),
        "1_3": ("future", "progressive"),
        "1_4": ("future", "perfect"),
        "1_5": ("future", "perfect_progressive"),
        "0_2": ("present", "simple"),
        "0_3": ("present", "progressive"),
        "0_4": ("present", "perfect"),
        "0_5": ("present", "perfect_progressive"),
        "-1_2": ("past", "simple"),
        "-1_3": ("past", "progressive"),
        "-1_4": ("past", "perfect"),
        "-1_5": ("past", "perfect_progressive"),
    }

    for source, sample_list in test_samples.items():
        sample_list_filtered = [s for s in sample_list if s.sent in allowed_sents]
        for i in range(0, len(sample_list_filtered), batch_size):
            batch_samples = sample_list_filtered[i : i + batch_size]
            batch_prompts = []
            batch_tenses = []
            batch_steer_targets = []
            batch_solutions = []
            for test_sample in batch_samples:

                source_tense = label_mapper[source]

                # create few-shot prompt
                template_type = random.randint(0, 1)

                if prompt_style == "a":
                    if source_tense[0] == "future":
                        if template_type == 0:
                            few_shot_prompt = f"I will be at the park. \\\\ I am at the park.\n\nPaul will go to school. \\\\ Paul goes to school.\n\n{test_sample.sent} \\\\"
                            steer = "past"
                            sol_col = "Present"
                        else:
                            few_shot_prompt = f"I will be at the park. \\\\ I was at the park.\n\nPaul will go to school. \\\\ Paul went to school.\n\n{test_sample.sent} \\\\"
                            steer = "present"
                            sol_col = "Past"
                    elif source_tense[0] == "present":
                        if template_type == 0:
                            few_shot_prompt = f"I am at the park. \\\\ I will be at the park.\n\nPaul goes to school. \\\\ Paul will go to school.\n\n{test_sample.sent} \\\\"
                            steer = "past"
                            sol_col = "Future"
                        else:
                            few_shot_prompt = f"I am at the park. \\\\ I was at the park.\n\nPaul goes to school. \\\\ Paul went to school.\n\n{test_sample.sent} \\\\"
                            steer = "future"
                            sol_col = "Past"
                    elif source_tense[0] == "past":
                        if template_type == 0:
                            few_shot_prompt = f"I was at the park. \\\\ I will be at the park.\n\nPaul went to school. \\\\ Paul will go to school.\n\n{test_sample.sent} \\\\"
                            steer = "present"
                            sol_col = "Future"
                        else:
                            few_shot_prompt = f"I was at the park. \\\\ I am at the park.\n\nPaul went to school. \\\\ Paul goes to school.\n\n{test_sample.sent} \\\\"
                            steer = "future"
                            sol_col = "Present"
                elif prompt_style == "b":
                    if source_tense[0] == "future":
                        if template_type == 0:
                            few_shot_prompt = f"Task: Modify the input text as shown in the examples.\n\nExample 1:\nInput: I will be at the park.\nOutput: I am at the park.\n\nExample 2:\nInput: Paul will go to school.\nOutput: Paul goes to school.\n\nNow it's your turn:\n\nInput: {test_sample.sent}\nOutput:"
                            steer = "past"
                            sol_col = "Present"
                        else:
                            few_shot_prompt = f"Task: Modify the input text as shown in the examples.\n\nExample 1:\nInput: I will be at the park.\nOutput: I was at the park.\n\nExample 2:\nInput: Paul will go to school.\nOutput: Paul went to school.\n\nNow it's your turn:\n\nInput: {test_sample.sent}\nOutput:"
                            steer = "present"
                            sol_col = "Past"
                    elif source_tense[0] == "present":
                        if template_type == 0:
                            few_shot_prompt = f"Task: Modify the input text as shown in the examples.\n\nExample 1:\nInput: I am at the park.\nOutput: I will be at the park.\n\nExample 2:\nInput: Paul goes to school.\nOutput: Paul will go to school.\n\nNow it's your turn:\n\nInput: {test_sample.sent}\nOutput:"
                            steer = "past"
                            sol_col = "Future"
                        else:
                            few_shot_prompt = f"Task: Modify the input text as shown in the examples.\n\nExample 1:\nInput: I am at the park.\nOutput: I was at the park.\n\nExample 2:\nInput: Paul goes to school.\nOutput: Paul went to school.\n\nNow it's your turn:\n\nInput: {test_sample.sent}\nOutput:"
                            steer = "future"
                            sol_col = "Past"
                    elif source_tense[0] == "past":
                        if template_type == 0:
                            few_shot_prompt = f"Task: Modify the input text as shown in the examples.\n\nExample 1:\nInput: I was at the park.\nOutput: I will be at the park.\n\nExample 2:\nInput: Paul went to school.\nOutput: Paul will go to school.\n\nNow it's your turn:\n\nInput: {test_sample.sent}\nOutput:"
                            steer = "present"
                            sol_col = "Future"
                        else:
                            few_shot_prompt = f"Task: Modify the input text as shown in the examples.\n\nExample 1:\nInput: I was at the park.\nOutput: I am at the park.\n\nExample 2:\nInput: Paul went to school.\nOutput: Paul goes to school.\n\nNow it's your turn:\n\nInput: {test_sample.sent}\nOutput:"
                            steer = "future"
                            sol_col = "Present"
                # print(few_shot_prompt)
                batch_prompts.append(few_shot_prompt)
                batch_tenses.append(label_mapper[source])
                batch_steer_targets.append(steer)
                batch_solutions.append(
                    filter_df[filter_df["Original"] == test_sample.sent][
                        sol_col
                    ].values[0]
                )

            # for p in batch_prompts:
            #    print(f"{p}\n-----------")
            batch_inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True
            ).to(model.device)

            batches.append(
                (
                    batch_samples,
                    batch_prompts,
                    batch_inputs,
                    batch_tenses,
                    batch_steer_targets,
                    batch_solutions,
                )
            )
    return batches
