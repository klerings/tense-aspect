import nltk

nltk.download("treebank")
nltk.download("propbank")

import spacy

from nltk.corpus import propbank
from nltk.corpus import treebank
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm


def extract_sentences_with_annotations():
    sentences_with_annotations = {}

    pb_instances = propbank.instances()
    detokenizer = TreebankWordDetokenizer()

    for inst in tqdm(pb_instances):
        tree_id = inst.fileid  # Corresponding Penn Treebank file ID
        sentence_id = int(inst.sentnum)  # Sentence number in the file

        try:
            sentence_tokens = treebank.sents(tree_id)[sentence_id]
            sentence = detokenizer.detokenize(sentence_tokens)

            # Extract verb annotation
            # predicate = inst.predicate.select(tree)
            # arguments = inst.arguments  # List of argument structures

            infl = inst.inflection

            # Store extracted data
            anno = {
                "sentence": sentence,
                # "verb": predicate,
                # "arguments": arguments,
                "tense": infl.tense,
                "aspect": infl.aspect,
                "person": infl.person,
                "voice": infl.voice,
                "form": infl.form,
            }
            if sentence in sentences_with_annotations:
                sentences_with_annotations[sentence].append(anno)
            else:
                sentences_with_annotations[sentence] = [anno]
        except OSError:
            # no tree available
            continue

    return sentences_with_annotations


if __name__ == "__main__":
    data = extract_sentences_with_annotations()

    sents_per_tense_aspect = {
        "(n,-)": [],
        "(n,p)": [],
        "(n,o)": [],
        "(n,b)": [],
        "(p,-)": [],
        "(p,p)": [],
        "(p,o)": [],
        "(p,b)": [],
        "(f,-)": [],
        "(f,p)": [],
        "(f,o)": [],
        "(f,b)": [],
    }

    all_annos = sum([len(annos) for annos in data.values()])
    print(f"all annotations: {all_annos}")
    valid = 0

    nlp = spacy.load("en_core_web_lg")

    for sent, anno_list in data.items():
        # this doesnt catch sentences that have multiple verbs but just one annotation
        if len(anno_list) > 1:
            continue

        # this should catch sentences that have multiple verb phrases
        doc = nlp(sent)
        verb_phrases = [word for word in doc if word.pos_ == "VERB"]
        if len(verb_phrases) > 1:
            continue

        anno = anno_list[0]
        key = f"({anno['tense']},{anno['aspect']})"
        if key in sents_per_tense_aspect:
            sents_per_tense_aspect[key].append(anno)
            valid += 1

    print(f"all annotations with one verb in sent: {valid}")
    for key, list in sents_per_tense_aspect.items():
        print(f"{key}: {len(list)}")

    propbank_to_tense_labels = {
        "(n,-)": "simple_present",
        "(n,p)": "present_perfect",
        "(n,o)": "present_progressive",
        "(n,b)": "present_perfect_progressive",
        "(p,-)": "simple_past",
        "(p,p)": "past_perfect",
        "(p,o)": "past_progressive",
        "(p,b)": "past_perfect_progressive",
        "(f,-)": "simple_future",
        "(f,p)": "future_perfect",
        "(f,o)": "future_progressive",
        "(f,b)": "future_perfect_progressive",
    }

    for key, list in sents_per_tense_aspect.items():
        with open(
            f"data/raw_data/propbank/{propbank_to_tense_labels[key]}.txt", "w+"
        ) as f:
            for sent in list:
                f.write(sent["sentence"])
                f.write("\n")
