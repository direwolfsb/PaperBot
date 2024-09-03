import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from bs4 import BeautifulSoup
import os

# Load environment variables from the .env file
load_dotenv()

# Load the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Initialize embeddings using the OpenAI API key
embeddings = OpenAIEmbeddings(openai_api_key=api_key)


# Function to get a response based on user input
def get_response(user_input):
    # Create a retriever chain using the stored vector store
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    
    # Create a conversational retrieval-augmented generation (RAG) chain
    conversational_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    # Invoke the RAG chain with chat history and user input to get a response
    response = conversational_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })
    
    return response["answer"]

# Function to create a vector store from a website URL
def get_vectorstore_from_url(url):
    # Load the document from the given URL
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # Split the document into chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=20)
    chunks = text_splitter.split_documents(document)
    
    # Create a vector store from the chunks using the embeddings
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

# Function to create a context-aware retriever chain
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=api_key)  # Initialize the language model
    
    # Use the vector store as a retriever
    retriever = vector_store.as_retriever()
    
    # Define the prompt template for retrieving relevant information
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation if you don't know the answer and apologize")
    ])
    
    # Create a retriever chain using the language model, retriever, and prompt
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

# Function to create a conversational RAG chain
def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(openai_api_key=api_key)  # Initialize the language model
    
    # Define the prompt template for answering user questions based on retrieved context
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    # Create a chain to combine retrieved documents into a response
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create and return the full retrieval-augmented generation chain
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Streamlit app setup
st.set_page_config(page_title="Paperbot", page_icon="🤖")
st.title("Chat with websites 🤖")

# Sidebar for user input settings
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

# If no website URL is provided, display an info message
if website_url is None or website_url == "":
    st.info("Please enter a website URL")
else:
    # Initialize chat history if not already done
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a Paperbot. How can I help you?"),
        ]

    # Initialize vector store if not already done
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)    

    # Capture user input from the chat box
    user_input = st.chat_input("Ask me anything")
    
    if user_input is not None and user_input != "":
        # Get the response based on user input and chat history
        response = get_response(user_input)
       
        # Update the chat history with the user input and bot response
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response))
    
    # Display the chat history in the Streamlit interface
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
