from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def load_documents_into_database(model_name: str, documents_path: str) -> Chroma:
    """
    Loads documents from the specified directory into the Chroma database
    after splitting the text into chunks.

    Args:
        model_name (str): The name of the embedding model.
        documents_path (str): The path to the directory containing documents.

    Returns:
        Chroma: The Chroma database with loaded documents.
    """
    print("Loading documents")
    raw_documents = load_documents(documents_path)
    documents = TEXT_SPLITTER.split_documents(raw_documents)

    print("Creating embeddings and loading documents into Chroma")
    embedding_function = OllamaEmbeddings(model=model_name)
    db = Chroma.from_documents(
        documents,
        embedding_function=embedding_function,  # Use the embedding function
    )
    return db

def check_if_embeddings_exist(model_name: str) -> bool:
    """
    Checks if embeddings already exist in the Chroma database.

    Args:
        model_name (str): The name of the embedding model.

    Returns:
        bool: True if embeddings exist, False otherwise.
    """
    try:
        # Initialize Chroma with the existing embedding function
        embedding_function = OllamaEmbeddings(model=model_name)
        db = Chroma(embedding_function=embedding_function)

        # Attempt to retrieve a small number of documents to check for existence
        existing_documents = db.get(limit=1)

        if existing_documents:
            print("Embeddings found in the database.")
            return True
        else:
            print("No embeddings found in the database.")
            return False
    except Exception as e:
        print(f"Error while checking for embeddings: {e}")
        return False

def load_documents(path: str) -> List[Document]:
    """
    Loads documents from the specified directory path.

    This function supports loading of PDF, Markdown, and HTML documents by utilizing
    different loaders for each file type. It checks if the provided path exists and
    raises a FileNotFoundError if it does not. It then iterates over the supported
    file types and uses the corresponding loader to load the documents into a list.

    Args:
        path (str): The path to the directory containing documents to load.

    Returns:
        List[Document]: A list of loaded documents.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    loaders = {
        ".pdf": DirectoryLoader(
            path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
        ),
        ".md": DirectoryLoader(
            path,
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=True,
        ),
    }

    docs = []
    for file_type, loader in loaders.items():
        print(f"Loading {file_type} files")
        docs.extend(loader.load())
    return docs
