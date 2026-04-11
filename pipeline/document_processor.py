"""Document processing pipeline for DeepTutor.

Handles PDF ingestion, text extraction, chunking, and vector store creation
for use in the science QA pipeline.
"""

import os
import hashlib
from typing import List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


# Default chunking parameters
# Increased chunk size from 1000 to 1500 for better context retention in
# scientific papers, which tend to have longer, denser paragraphs.
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 150


def compute_file_hash(file_path: str) -> str:
    """Compute MD5 hash of a file for cache invalidation."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF file and return a list of LangChain Document objects.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of Document objects, one per page.

    Raises:
        FileNotFoundError: If the PDF does not exist at the given path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """Split documents into smaller chunks suitable for embedding.

    Args:
        documents: Raw documents loaded from a PDF.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def build_vectorstore(
    chunks: List[Document],
    embeddings: Optional[OpenAIEmbeddings] = None,
) -> FAISS:
    """Build a FAISS vector store from document chunks.

    Args:
        chunks: Chunked Document objects to embed and index.
        embeddings: Optional pre-instantiated embeddings model.
                    Defaults to OpenAIEmbeddings if not provided.

    Returns:
        A FAISS vector store ready for similarity search.
    """
    if embeddings is None:
        embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def process_pdf(
    file_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embeddings: Optional[OpenAIEmbeddings] = None,
) -> FAISS:
    """End-to-end pipeline: load
