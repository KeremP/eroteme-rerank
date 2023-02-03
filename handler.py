import os
import uuid
import json
from typing import Optional, Union, List, Dict, Any, Tuple
from dotenv import load_dotenv
import logging
import pprint

pp = pprint.PrettyPrinter(indent=4, compact=True)
load_dotenv()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from urllib.request import urlopen
from bs4 import BeautifulSoup
import redis
from redis import UsernamePasswordCredentialProvider
import numpy as np

import hashlib
import faiss
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore

# OPENAI CONFIG
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_KEY

# REDIS CONFIG
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_DB   = 0
USE_REDIS = bool(int(os.getenv("USE_REDIS")))

# DEBUG MODE
DEBUG = bool(int(os.getenv("DEBUG")))

class CustomFAISS(FAISS):
    """
    Extension of langchain `FAISS` class to use cached embeddings from a Redis instance

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def decode_cached_embeddings(cache_result):
        return json.loads(cache_result)

    @classmethod
    def from_precomputed(cls, embedding, texts: list, metadatas: List[dict], dims: int, redis: redis.Redis, doc_hash: str, **kwargs) -> FAISS:
        embeddings = []
        docs = []
        for i in range(len(doc_hash)):
            hash = doc_hash[i]
            embed_text = texts[i]
            metadata = metadatas[i]
            if redis.exists(hash) == 0 or redis is None:
                tmp = embedding.embed_documents(embed_text)
                embeddings+=tmp
                if redis is not None:
                    try:
                        redis.set(hash, json.dumps(tmp))
                    except Exception as e:
                        logger.warning(f"Unable to cache embeddings in Redis: {e}")
            else:
                cached_embedding = redis.get(hash)
                embeddings+=CustomFAISS.decode_cached_embeddings(cached_embedding)
            docs += [Document(page_content=text, metadata=metadata[i]) for i,text in enumerate(embed_text)]

        index = faiss.IndexFlatL2(dims)
        index.add(np.array(embeddings, dtype=np.float32))
        index_to_id = {i: str(uuid.uuid4()) for i in range(len(docs))}
        docstore = InMemoryDocstore(
            {index_to_id[i]: doc for i, doc in enumerate(docs)}
        )
        return cls(embedding.embed_query, index, docstore, index_to_id)

def init_redis(host: str, port: int, username: Optional[str] = None, password: Optional[str] = None, ssl: bool = True) -> Union[redis.Redis, None]:
    """
    Initialize Redis instance @host:port

    Optionally pass `username` and `password` for auth, and `ssl` if needed

    By default `decode_responses` is set to False
    """
    if username and password:
        cred_provider = UsernamePasswordCredentialProvider(username, password)
    else:
        cred_provider = None
    
    client = redis.Redis(host, port, credential_provider=cred_provider, decode_responses=False, ssl=ssl)
    try:
        pong = client.ping()
    except:
        pong = None
    
    if pong:
        logger.info("Redis initialized")
        return client
    else:
        raise Exception("Unable to connect to Redis server")

def hash_result(url):
    return hashlib.md5(
        bytes(url, 'utf-8')
    ).hexdigest()

def extract_text_from_url(url: str) -> List[str]:

    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")
    
    text = soup.find_all("p")

    out = [t.getText().strip("\n") for t in text if t.getText() != "\n"]
    return out

def build_context(html: str):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    context = text_splitter.split_text(html)
    metadata = [{'source':i,'content':context[i]} for i in range(len(context))]
    return context, metadata


def build_docstore(context, metadata, doc_hash, redis, dims=1536):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
    docsearch = CustomFAISS.from_precomputed(embeddings, context, metadatas=metadata, dims=dims, redis=redis, doc_hash=doc_hash)
    return docsearch



def lambda_handler(event, context):
    if USE_REDIS:
        logger.info("Using REDIS")
        if DEBUG: ssl = False 
        else: ssl = True
        REDIS_CLIENT: redis.Redis = init_redis(REDIS_HOST, REDIS_PORT, REDIS_DB, ssl=ssl)
    else:
        REDIS_CLIENT = None

    urls = event.get('urls')
    query = event.get('query')

    contexts = []
    metadatas = []
    metadata_cache = {}
    id_html_map = {}
    url_hashes = []

    for url in urls:
        extracted_html = extract_text_from_url(url)
        extracted_html = "\n\n".join(extracted_html)
        id_html_map.update({url:extracted_html})
        url_hashes.append(hash_result(url))
        context, metadata = build_context(extracted_html)
        contexts.append(context)
        metadata_cache.update({url:metadata})
        metadatas.append([{"source":f"{url}<:>{i}"} for i in range(len(metadata))])

    docstore = build_docstore(contexts, metadatas, url_hashes, REDIS_CLIENT)

    similar_sections = docstore.similarity_search_with_score(
        query
    )
    
    results = [
        doc[0].page_content for doc in similar_sections
    ]
    meta_keys = [
        doc[0].metadata['source'].split("<:>") for doc in similar_sections
    ]
    sources = [
        metadata_cache[k[0]][int(k[1])] for k in meta_keys
    ]
    resp = {
        'results':results,
        'sources':sources
    }

    return resp