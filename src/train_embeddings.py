import glob
from pathlib import Path
from typing import Iterable

from gensim.utils import tokenize
from glovpy import GloVe


def stream_sentences(files: list[str]) -> Iterable[list[str]]:
    for file in files:
        with open(file) as in_file:
            for line in in_file:
                yield list(tokenize(line, lower=True, deacc=True))


def main():
    print("Collecting training data.")
    files = glob.glob("dat/*.txt")
    sentences = list(stream_sentences(files))
    print("Training Word embeddings.")
    model = GloVe(vector_size=50)
    model.train(sentences)
    print("Saving embeddings.")
    out_dir = Path("model/")
    out_dir.mkdir(exist_ok=True)
    out_path = str(out_dir.joinpath("sermons_glove_50.kv"))
    model.wv.save(out_path)
    print("DONE")


if __name__ == "__main__":
    main()
