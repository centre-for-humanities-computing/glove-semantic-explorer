from embedding_explorer.app import get_dash_app
from embedding_explorer.blueprints.explorer import create_explorer
from gensim.models import KeyedVectors

kv = KeyedVectors.load("model/sermons_glove_50.kv")

blueprint = create_explorer(corpus=kv.index_to_key, embeddings=kv.vectors)
app = get_dash_app(blueprint=blueprint, name=__name__, use_pages=False)

server = app.server

if __name__ == "__main__":
    app.run_server(debug=False, port=8080, host="0.0.0.0")
