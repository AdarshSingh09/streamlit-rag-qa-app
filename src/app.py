import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
import torch
import io
import tempfile
import os
import shutil

load_dotenv()

torch.classes.__path__ = []


st.set_page_config("Retrieval Augmented Generation")
st.title("Retrieval Augmented Generation")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)


def create_vector_store(all_splits):
    vector_store = FAISS.from_documents(all_splits,embedding=embeddings)
    return vector_store

def get_response(user_question):
    
    prompt = PromptTemplate(template="""
                            * You are a knowledgable researcher.
                            * Your job is to use the below provided context to answer user's question with proper explaination in descriptive way.
                            * Do not makeup anything.
                            * if the provided context is not sufficient just say I don't know.
                            Context : {context}
                            Question : {question}
                            Answer : 
                            """,
                            input_variables=['context', 'question'])
    
    retriever = st.session_state.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(user_question)
    
    retrieved_docs = retriever.invoke(user_question)
    
    return {"answer":response, "sources":retrieved_docs}

# session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_processed" not in  st.session_state:
    st.session_state.pdf_processed = False
if "temp_dir_path" not in st.session_state: 
    st.session_state.temp_dir_path = None

with st.sidebar:
    st.header("Upload PDF Documents")
    files = st.file_uploader(label="Upload Files", type="pdf",accept_multiple_files=True,label_visibility="collapsed")
    
    # Files uploaded
    if files is not None and len(files) > 0:
        # Processing Documents
        if st.button("Process Documents 🔁"):
            with st.spinner(f"Processing {len(files)} Documents", show_time=True):
                documents = []
                temp_dir = None
                temp_dir = tempfile.mkdtemp()
                st.session_state.temp_dir_path = temp_dir
                print("Temperory Directory created!")
                
                for i, uploaded_file in enumerate(files):
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    with open(file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                            
                    # st.success(f"Saved: {uploaded_file.name} to {file_path}")
                    
                    loader = PyMuPDFLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                
                # Chunking
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                all_splits = text_splitter.split_documents(documents)
                st.write(f"Number of Chunks created: {len(all_splits)}")

                st.session_state.vector_store = create_vector_store(all_splits)
                
                st.write("Vectore store created")
                st.success("✅ Documents processed! Ask your questions.")
                st.session_state.pdf_processed = True

# Deleting the temporary files
if st.session_state.pdf_processed:
    st.markdown("---") 
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2: 
        if st.button("Clear Processed Data and Files 🗑️", use_container_width=True):
            if st.session_state.temp_dir_path and os.path.exists(st.session_state.temp_dir_path):
                try:
                    shutil.rmtree(st.session_state.temp_dir_path)
                    st.success(f"Temporary directory '{st.session_state.temp_dir_path}' and its contents cleared!")
                    st.session_state.temp_dir_path = None
                    st.session_state.pdf_processed = False
                    st.session_state.vector_store = None
                    st.rerun() 
                except Exception as e:
                    st.error(f"Error clearing temporary directory: {e}")
            else:
                st.info("No temporary directory to clear.")
    st.markdown("---") 
                
if st.session_state.pdf_processed != True:
    st.info("Please First upload & process your pdf files")
else:           
    user_question = st.chat_input("Ask a question about your documents...") 
    
    if user_question: 
        with st.spinner("Gemini is thinking and retrieving sources... 🤔"): 
            response = get_response(user_question)

        sources = response["sources"]
        
        with st.chat_message("human"):
            st.write(user_question)
            
        with st.chat_message("ai"):
            
            st.write(response["answer"])
            st.divider()

            # Display the sources in expanders
            st.subheader("📚 Sources")
            for doc in response["sources"]:
                source_file = doc.metadata.get('source', 'N/A')
                page_num = doc.metadata.get('page', 'N/A')
                
                with st.expander(f"Source: {source_file} - Page: {page_num}"):
                    st.text_area("", value=doc.page_content, height=150, disabled=True)