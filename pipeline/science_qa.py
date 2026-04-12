"""Science QA pipeline for DeepTutor.

Handles document-based question answering using RAG (Retrieval-Augmented Generation)
with support for multiple LLM backends.
"""

import os
import logging
from typing import Optional

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLLM
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)

# System prompt template for scientific paper Q&A
QA_PROMPT_TEMPLATE = """
You are an expert tutor helping students understand scientific papers and academic content.
Use the following context from the document to answer the question accurately and clearly.
If the answer cannot be found in the context, say so honestly rather than making up information.

Context:
{context}

Question: {question}

Answer: Provide a detailed, educational response that helps the student understand the concept.
"""

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def build_qa_chain(
    llm: BaseLLM,
    vectorstore: VectorStore,
    # Bumped top_k from 4 to 5 -- in my testing, 5 chunks gives noticeably better
    # answers for longer papers without a meaningful speed hit on my machine.
    top_k: int = 5,
    chain_type: str = "stuff",
) -> RetrievalQA:
    """Build a RetrievalQA chain for answering questions about documents.

    Args:
        llm: The language model to use for generation.
        vectorstore: The vector store containing document embeddings.
        top_k: Number of relevant chunks to retrieve.
        chain_type: LangChain chain type ("stuff", "map_reduce", "refine").

    Returns:
        A configured RetrievalQA chain.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    chain_type_kwargs = {"prompt": QA_PROMPT} if chain_type == "stuff" else {}

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )

    logger.info("QA chain built with chain_type=%s, top_k=%d", chain_type, top_k)
    return qa_chain


def run_qa(
    qa_chain: RetrievalQA,
    question: str,
) -> dict:
    """Run a question through the QA chain and return structured results.

    Args:
        qa_chain: The configured RetrievalQA chain.
        question: The user's question string.

    Returns:
        A dict with keys:
            - "answer": The generated answer text.
            - "sources": List of source document metadata dicts.
            - "error": Optional error message if something went wrong.
    """
    if not question or not question.strip():
        return {"answer": "", "sources": [], "error": "Empty question provided."}

    try:
        result = qa_chain.invoke({"query": question})
        answer = result.get("result", "").strip()
        source_docs = result.get("source_documents", [])

        sources = [
