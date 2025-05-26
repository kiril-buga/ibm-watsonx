import logging
import os

from dotenv import load_dotenv
from langchain.indexes import SQLRecordManager
from langchain_core.indexing import index
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import BM25BuiltInFunction, Milvus


# from local_loader import get_document_text
# from remote_loader import download_file
# from splitter import split_documents

# def create_vector_db(docs, embeddings=None, collection_name=""):
#     if not docs:
#         logging.warning("Empty docs passed in to create vector database")
#     if not embeddings:
#         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
#
#     vector_store_saved = Milvus.from_documents(
#         [Document(page_content="foo!")],
#         embeddings,
#         collection_name="langchain_example",
#         connection_args={"uri": URI},
#     )
#
#     doc_store = QdrantVectorStore.from_documents(
#         docs, embedding=embeddings,
#         sparse_embedding=sparse_embeddings,
#         url=os.getenv("QDRANT_URL"),
#         api_key=os.getenv("QDRANT_API_KEY"),
#         collection_name=collection_name,
#         retrieval_mode=RetrievalMode.HYBRID,
#         force_recreate=True
#     )
#
#     return doc_store


def load_vector_db(embeddings=None, collection_name=""):
    if not embeddings:
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    doc_store = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={
            "uri": f"http://{os.getenv('MILVUS_HOST')}:{os.getenv('MILVUS_PORT')}",
            "host": os.environ["MILVUS_HOST"],
            "port": os.environ["MILVUS_PORT"],  # str or int both accepted
            "user": "ibmlhapikey",
            "password": os.environ["MILVUS_PASSWORD"],  # ← your API key
            "secure": True,
            "db_name": "default",
        },
        # index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 1024}},
        consistency_level="Strong",
        # drop_old=False,  # set to True if seeking to drop the collection with that name if it exists
    )

    return doc_store

if __name__=="__main__":
    print(load_vector_db())