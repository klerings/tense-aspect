import json
import re


def parse_tense_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    results = []

    for example in data.get("examples", []):
        input_text = example.get("input", "")
        target_text = example.get("target", "")

        # extract target tense using regex
        # looking for text in all caps after "Change tense to "
        tense_match = re.search(r"Change tense to ([A-Z\s]+):", input_text)

        if tense_match:
            target_tense = tense_match.group(1).strip()

            # get original sentence (everything before "Change tense to")
            original_sentence = input_text.split("Change tense to")[0].strip()

            results.append(
                {
                    "original_sentence": original_sentence,
                    "target_tense": target_tense,
                    "target_sentence": target_text,
                }
            )

    return results


if __name__ == "__main__":
    file_path = "data/raw_data/bigbench_tense_task.json"

    parsed_data = parse_tense_data(file_path)

    bigbench_to_tense_labels = {
        "SIMPLE PRESENT": "simple_present",
        "PRESENT CONTINUOUS": "present_progressive",
        "PRESENT PERFECT": "present_perfect",
        "PRESENT PERFECT CONTINUOUS": "present_perfect_progressive",
        "SIMPLE PAST": "simple_past",
        "PAST CONTINUOUS": "past_progressive",
        "PAST PERFECT": "past_perfect",
        "PAST PERFECT CONTINUOUS": "past_perfect_progressive",
        "SIMPLE FUTURE": "simple_future",
        "FUTURE CONTINUOUS": "future_progressive",
        "FUTURE PERFECT": "future_perfect",
        "FUTURE PERFECT CONTINUOUS": "future_perfect_progressive",
    }

    samples = {}

    for item in parsed_data:
        tense = bigbench_to_tense_labels[item["target_tense"]]
        if tense in samples:
            samples[tense].append(item["target_sentence"])
        else:
            samples[tense] = [item["target_sentence"]]

    for tense, list in samples.items():
        with open(f"data/raw_data/bigbench/{tense}.txt", "w+") as f:
            for sent in list:
                f.write(sent)
                f.write("\n")
        print(f"{tense}: {len(list)}")
