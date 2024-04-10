from pathlib import Path
from typing import Iterable

from embedding_explorer.app import get_dash_app
from embedding_explorer.blueprints.explorer import create_explorer
from gensim.models import KeyedVectors
from gensim.utils import tokenize
from glovpy import GloVe
from radicli import Arg, Radicli

from glove_semantic_explorer.templates import (
    COMPOSE_TEMPLATE,
    DOCKERFILE_TEMPLATE,
    MAIN_FILE_TEMPLATE,
)

cli = Radicli()


def stream_sentences(files: list[str]) -> Iterable[list[str]]:
    for file in files:
        with open(file) as in_file:
            for line in in_file:
                yield list(tokenize(line, lower=True, deacc=True))


@cli.command(
    "train_model",
    data_folder=Arg(
        help="Folder containing .txt files to train a GloVe model on."
    ),
    out_path=Arg(
        "--out_file",
        "-o",
        help="Path to the output file in keyed vector format.",
    ),
)
def train_model(data_folder: str, out_path: str = "model/glove.kv") -> None:
    print("Collecting training data.")
    data_folder = Path(data_folder)
    files = data_folder.glob("*.txt")
    sentences = list(stream_sentences(files))
    print("Training Word embeddings.")
    model = GloVe(vector_size=50)
    model.train(sentences)
    print("Saving embeddings.")
    out_path = Path(out_path)
    out_dir = out_path.parent
    out_dir.mkdir(exist_ok=True)
    model.wv.save(str(out_path))
    print("DONE")


@cli.command(
    "run_explorer",
    model_path=Arg(
        "--model_file",
        "-m",
        help="Path to the model file in keyed vector format.",
    ),
    port=Arg("--port", "-p", help="Port to run the app on."),
)
def run_explorer(model_path: str = "model/glove.kv", port: int = 8080) -> None:
    kv = KeyedVectors.load(model_path)
    blueprint = create_explorer(corpus=kv.index_to_key, embeddings=kv.vectors)
    app = get_dash_app(blueprint=blueprint, name=__name__, use_pages=False)
    app.run_server(debug=False, port=port, host="0.0.0.0")


@cli.command(
    "generate_docker",
    model_path=Arg(
        "--model_file",
        "-m",
        help="Path to the model file in keyed vector format.",
    ),
    port=Arg("--port", "-p", help="Port to run the app on."),
    url_base_pathname=Arg(
        "--url_base_pathname",
        "-u",
        help="Base path name of the app at the port.",
    ),
    out_dir=Arg(
        "--out_dir",
        "-o",
        help="Folder to output the container information to.",
    ),
)
def generate_docker(
    model_path: str = "model/glove.kv",
    port: int = 8080,
    url_base_pathname: str = "/",
    out_dir: str = "deployment/",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    print("Generating files for deployment.")
    with out_dir.joinpath("main.py").open("w") as main_file:
        main_file.write(MAIN_FILE_TEMPLATE)
    with out_dir.joinpath("Dockerfile").open("w") as dockerfile:
        dockerfile.write(DOCKERFILE_TEMPLATE)
    with out_dir.joinpath("compose.yaml").open("w") as dockerfile:
        dockerfile.write(
            COMPOSE_TEMPLATE.format(
                port=port,
                model_path=Path(model_path).absolute(),
                url_base_pathname=url_base_pathname,
            )
        )
    print("DONE")
