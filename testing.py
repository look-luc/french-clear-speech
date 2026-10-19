import random

from datasets import load_dataset


def get_french_sentences():
    data = load_dataset("OpenLLM-France/wikimedia", "fr", streaming=True)
    random_amount = 10
    random_subset = random.sample(list(data["train"]["text"]), k=random_amount)
    print("="*100)
    print(random_subset)
    print("="*100)


if __name__ == "__main__":
    get_french_sentences()
