# glove-semantic-explorer

Set up embedding explorer on a corpus with GloVe word embeddings with an easy-to-use CLI.

## Installation

You can install the CLI from PyPI:

> We recommend that you use a Linux/Unix system, preferably Debian when using this tool.
> Windows and MacOS could still work, but we do not guarrantee this.

```bash
pip install glove-semantic-explorer
```


## Usage

### 1. Train a model

You will need a corpus in the format of a bunch of `.txt` files in a folder.
Every line in a file should represent one sentence/passage.

To train a GloVe model on the corpus, run:

```bash
python3 -m glove_semantic_explorer train_model dat/ -o model/glove.kv
```

This will output a keyed vectors file to  `model/glove.kv`.

### 2. Run the Explorer

To run the explorer on the trained model locally, run:

```bash
python3 -m glove_semantic_explorer run_explorer -m model/glove.kv --port 8080
```

This will start embedding-explorer on the trained embedding model on port 8080.

### 3. Deploy!

You can deploy the application using `docker compose`.
The way this can be done with our CLI is by auto-generating a `Dockerfile`, a `compose.yaml` and a `main.py` file, that contains all the code for running the server.

To output this into a folder called `deployment/`, run the following command:

```bash
python3 -m glove_semantic_explorer generate_docker "your_project_name" -m model/glove.kv -p 8080 -o deployment/
```

> Beware that the model file only gets mounted to the container, and thus should not be removed, moved or renamed.

To deploy the app with docker compose run the following:

```bash
cd deployment/
sudo docker compose up
```

The app will then run on port 8080.
