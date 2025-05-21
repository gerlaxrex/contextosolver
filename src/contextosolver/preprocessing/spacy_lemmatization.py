import spacy
import json

from contextosolver import DATA_PATH

if __name__ == "__main__":
    with open(DATA_PATH / "index.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    nlp = spacy.load("en_core_web_sm")

    lemmas = []

    for word in data:
        doc = nlp(word)
        lemmas.append(doc[0].lemma_)

    unique_lemmas = list(set(lemmas))

    with open(DATA_PATH / "unique_lemmas.json", "w", encoding="utf-8") as f:
        json.dump(list(sorted(unique_lemmas)), f)

    print(f"Reduced wordlist of {(1 - len(unique_lemmas) / len(data)) * 100:.2f}%")
