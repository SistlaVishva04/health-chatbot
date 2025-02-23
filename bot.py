import streamlit as st
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_groq import ChatGroq

def initialize_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",  # Replace with your API key
        model_name="llama-3.3-70b-versatile"
    )
    return llm

def create_vector_db():
    loader = PyPDFLoader("mental_health_Document.pdf")  # Ensure the file exists
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    vector_db.persist()
    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """You are a compassionate mental health chatbot. Respond thoughtfully to the following question:
    {context}
    User: {question}
    Chatbot: """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Streamlit App
def main():
    st.title("🧠 Mental Health Chatbot")
    st.write("A compassionate AI-powered chatbot to support mental well-being.")
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize LLM
    llm = initialize_llm()
    db_path = "./chroma_db"
    
    if not os.path.exists(db_path):
        vector_db = create_vector_db()
    else:
        embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    qa_chain = setup_qa_chain(vector_db, llm)
    
    # Display chat history
    for chat in st.session_state.chat_history:
        st.write(f"**You:** {chat['question']}")
        st.write(f"**Hope:** {chat['response']}")

    # User Input
    user_input = st.text_input("You:", key="user_input")
    if st.button("Ask"): 
        if user_input.lower() == "exit":
            st.write("Hope: Take care of yourself, goodbye! 💙")
        else:
            response = qa_chain.run(user_input)
            st.session_state.chat_history.append({"question": user_input, "response": response})  # Store chat history
            st.rerun()  # Corrected method for rerunning the script

if __name__ == "__main__":
    main()
