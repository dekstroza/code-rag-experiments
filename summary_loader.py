import os
import sys
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import utils as chromautils


# Define constants
DB_PATH = "vectorstores/db/"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 200
JAVA_PARSER_THRESHOLD = 10

def persist_vector_db(docs):
    model_kwargs = {'device': 'mps','trust_remote_code': True}
    encode_kwargs = {'normalize_embeddings': False}
    embedding = HuggingFaceEmbeddings( model_name=MODEL_NAME, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=DB_PATH,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True,),
    )
    vectorstore.persist()

def load_json(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data


def create_vector_db(source_folder):
    # Construct the path to the input folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sources_dir = os.path.join(base_dir, 'summary_sources')
    source_path = os.path.join(sources_dir, source_folder)

    # Check if the input folder exists
    if not os.path.exists(source_path):
        print(f"The folder '{source_folder}' does not exist in the 'sources' directory.")
        return

    # Load and process the documents
    loader = GenericLoader.from_filesystem(
        source_path,
        glob="**/*",
        suffixes=[".txt"],
    )
    docs = loader.load()
    print("Loaded : ", len(docs), "documents")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    result = splitter.split_documents(docs)
    print(result)
    persist_vector_db(result)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <source_folder>")
        sys.exit(1)

    source_folder = sys.argv[1]
    create_vector_db(source_folder)

