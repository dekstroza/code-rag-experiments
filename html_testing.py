import os
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import DirectoryLoader

# Define constants

def pretty_print_docs(documents):
 for doc in documents:
    print(doc.metadata)
    print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print(doc.page_content)


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


    html_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.HTML, chunk_size=10000, chunk_overlap=500)
    result = html_splitter.split_documents(docs)
    pretty_print_docs(result)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <source_folder>")
        sys.exit(1)

    source_folder = sys.argv[1]
    create_vector_db(source_folder)

