MAIN_FILE_TEMPLATE = """
from embedding_explorer.app import get_dash_app
from embedding_explorer.blueprints.explorer import create_explorer
from gensim.models import KeyedVectors

kv = KeyedVectors.load("/model.kv")

blueprint = create_explorer(corpus=kv.index_to_key, embeddings=kv.vectors)
app = get_dash_app(blueprint=blueprint, name=__name__, use_pages=False)

server = app.server

if __name__ == "__main__":
    app.run_server(debug=False, port=8080, host="0.0.0.0")
"""

DOCKERFILE_TEMPLATE = """FROM python:3.9-slim-bullseye

RUN apt update
RUN apt install -y build-essential

RUN pip install gunicorn==20.1.0
RUN pip install typing-extensions
RUN pip install embedding_explorer==0.5.2
RUN pip install gensim==4.2.0
RUN pip install Pillow==9.5.0
RUN pip install scipy==1.12.0

COPY main.py main.py

EXPOSE 8080
CMD gunicorn --timeout 0 -b 0.0.0.0:8080 --worker-tmp-dir /dev/shm --workers=2 --threads=4 --worker-class=gthread main:server
"""

COMPOSE_TEMPLATE = """
services:
  server:
    image: sermon_embeddings/server
    build: .
    ports: 
      - "{port}:8080"
    deploy:
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 5
    volumes:
      - type: bind
        source: {model_path}
        target: /model.kv
    environment:
      DASH_URL_BASE_PATHNAME: "{url_base_pathname}"
"""
