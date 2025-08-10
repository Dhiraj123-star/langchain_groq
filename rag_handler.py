"""
RAG (Retrieval-Augmented Generation) Handler Module
Handles document processing, embedding, and retrieval for Q&A functionality
Uses OpenAI embeddings for high-quality semantic search
"""

import os
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    Docx2txtLoader,
    CSVLoader
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class RAGHandler:
    """Handles RAG functionality for document Q&A"""
    
    def __init__(self, groq_api_key: str, openai_api_key: str = None, 
                model_name: str = "llama-3.1-8b-instant", 
                temperature: float = 0.7, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize RAG handler
        
        Args:
            groq_api_key: Groq API key for generation
            openai_api_key: OpenAI API key for embeddings
            model_name: Model for generation
            temperature: Response temperature
            embedding_model: OpenAI embedding model to use
        """
        self.groq_api_key = groq_api_key
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.embedding_model = embedding_model
        
        # Initialize components
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.documents = []
        
        self._setup_components()
    
    def _setup_components(self):
        """Setup LLM and embeddings"""
        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=self.model_name,
            temperature=self.temperature
        )
        
        # Initialize OpenAI embeddings
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for embeddings")
        
        self.embeddings = OpenAIEmbeddings(
            api_key=self.openai_api_key,
            model=self.embedding_model
        )
    
    def load_documents(self, uploaded_files: List) -> Dict[str, Any]:
        """
        Load and process uploaded documents
        
        Args:
            uploaded_files: List of uploaded files from Streamlit
            
        Returns:
            Dictionary with processing results
        """
        results = {
            "success": [],
            "errors": [],
            "total_chunks": 0
        }
        
        all_documents = []
        
        for uploaded_file in uploaded_files:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=Path(uploaded_file.name).suffix
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name
                
                # Load document based on file type
                documents = self._load_single_document(tmp_path, uploaded_file.name)
                
                if documents:
                    all_documents.extend(documents)
                    results["success"].append(uploaded_file.name)
                else:
                    results["errors"].append(f"{uploaded_file.name}: No content extracted")
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
            except Exception as e:
                results["errors"].append(f"{uploaded_file.name}: {str(e)}")
        
        if all_documents:
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            chunks = text_splitter.split_documents(all_documents)
            results["total_chunks"] = len(chunks)
            
            # Create vector store
            self._create_vectorstore(chunks)
            self.documents = chunks
        
        return results
    
    def _load_single_document(self, file_path: str, file_name: str) -> List:
        """Load a single document based on its type"""
        file_extension = Path(file_name).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_extension == '.csv':
                loader = CSVLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata['source'] = file_name
            
            return documents
            
        except Exception as e:
            raise Exception(f"Failed to load {file_name}: {str(e)}")
    
    def _create_vectorstore(self, chunks: List):
        """Create FAISS vector store from document chunks"""
        try:
            self.vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Retrieve top 4 most similar chunks
            )
            
            # Create RAG chain
            self._create_rag_chain()
            
        except Exception as e:
            raise Exception(f"Failed to create vector store: {str(e)}")
    
    def _create_rag_chain(self):
        """Create the RAG chain for question answering"""
        # RAG prompt template
        rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided context. 
Use the following pieces of context to answer the question at the end.

If you don't know the answer based on the context, just say that you don't know, 
don't try to make up an answer.

Context: {context}

Question: {question}

Answer: """)
        
        # Create RAG chain
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question about the loaded documents
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.rag_chain:
            return {
                "answer": "No documents loaded. Please upload documents first.",
                "sources": [],
                "error": "No vector store available"
            }
        
        try:
            # Get relevant documents
            relevant_docs = self.retriever.invoke(question)
            
            # Generate answer
            answer = self.rag_chain.invoke(question)
            
            # Extract sources
            sources = []
            for doc in relevant_docs:
                source_info = {
                    "source": doc.metadata.get('source', 'Unknown'),
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "page": doc.metadata.get('page', 'N/A')
                }
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources,
                "error": None
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded documents"""
        if not self.documents:
            return {"total_chunks": 0, "sources": []}
        
        sources = list(set([doc.metadata.get('source', 'Unknown') for doc in self.documents]))
        
        return {
            "total_chunks": len(self.documents),
            "sources": sources,
            "embedding_model": self.embedding_model
        }
    
    def clear_documents(self):
        """Clear all loaded documents and reset vector store"""
        self.documents = []
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
    
    def update_model_settings(self, model_name: str = None, temperature: float = None,
                            embedding_model: str = None):
        """Update model settings and recreate components"""
        updated_llm = False
        updated_embeddings = False
        
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            updated_llm = True
        
        if temperature is not None and temperature != self.temperature:
            self.temperature = temperature
            updated_llm = True
        
        if embedding_model and embedding_model != self.embedding_model:
            self.embedding_model = embedding_model
            updated_embeddings = True
        
        if updated_llm:
            # Recreate LLM
            self.llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name=self.model_name,
                temperature=self.temperature
            )
        
        if updated_embeddings:
            # Recreate embeddings
            self.embeddings = OpenAIEmbeddings(
                api_key=self.openai_api_key,
                model=self.embedding_model
            )
            
            # If we have documents, need to recreate vector store
            if self.documents:
                self._create_vectorstore(self.documents)
        
        # Recreate RAG chain if we have documents and LLM was updated
        if (updated_llm or updated_embeddings) and self.retriever:
            self._create_rag_chain()


def create_rag_handler(groq_api_key: str, openai_api_key: str = None,
                        model_name: str = "llama-3.1-8b-instant", 
                        temperature: float = 0.7, 
                        embedding_model: str = "text-embedding-3-small") -> RAGHandler:
    """
    Factory function to create RAG handler
    
    Args:
        groq_api_key: Groq API key for generation
        openai_api_key: OpenAI API key for embeddings
        model_name: Model for generation
        temperature: Response temperature
        embedding_model: OpenAI embedding model to use
        
    Returns:
        Initialized RAGHandler instance
    """
    return RAGHandler(
        groq_api_key=groq_api_key,
        openai_api_key=openai_api_key,
        model_name=model_name,
        temperature=temperature,
        embedding_model=embedding_model
    )