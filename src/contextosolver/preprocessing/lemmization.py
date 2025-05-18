from typing import List

import nltk
from multiprocessing import Pool

from src.contextosolver import DATA_PATH

nltk.download("wordnet")


def lemmatize_mp(text_list: List[str], cores=4):
    with Pool(processes=cores) as pool:
        wnl = nltk.WordNetLemmatizer()
        result = pool.map(wnl.lemmatize, text_list)
    return result


if __name__ == "__main__":
    import json

    with open(DATA_PATH / "index.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    lemmed_text = lemmatize_mp(data)
    unique_lemmed_text = set(lemmed_text)

    with open(DATA_PATH / "unique_lemmed_text.json", "w", encoding="utf-8") as f:
        json.dump(list(sorted(unique_lemmed_text)), f)

    print(f"Reduced wordlist of {(1 - len(unique_lemmed_text) / len(data)) * 100:.2f}%")
