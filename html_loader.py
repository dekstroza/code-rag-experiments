import os
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import DirectoryLoader
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
# Define constants
DB_PATH = "vectorstores/db/"
MODEL_NAME = "all-MiniLM-L6-v2.gguf2.f16.gguf"
#MODEL_NAME = "nomic-embed-text-v1.f16.gguf"
GPT4ALL_KWARGS = {'allow_download': 'True'}

def create_vector_db(source_folder):
    # Construct the path to the input folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sources_dir = os.path.join(base_dir, 'source_code')
    source_path = os.path.join(sources_dir, source_folder)

    # Check if the input folder exists
    if not os.path.exists(source_path):
        print(f"The folder '{source_folder}' does not exist in the 'sources' directory.")
        return

    # Load and process the documents
    loader = DirectoryLoader(source_path, glob="**/*.html", show_progress=True, use_multithreading=True, loader_cls=BSHTMLLoader)
    docs = loader.load()


    html_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.HTML, chunk_size=10000, chunk_overlap=200)
    result = html_splitter.split_documents(docs)
    model_kwargs = {'device': 'mps','trust_remote_code': True}
    encode_kwargs = {'normalize_embeddings': False}
    embedding = HuggingFaceEmbeddings( model_name="BAAI/bge-large-en-v1.5", model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    vectorstore = Chroma.from_documents(
        documents=result,
        embedding=embedding,
        persist_directory=DB_PATH,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True,),
    )
    vectorstore.persist()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <source_folder>")
        sys.exit(1)

    source_folder = sys.argv[1]
    create_vector_db(source_folder)

