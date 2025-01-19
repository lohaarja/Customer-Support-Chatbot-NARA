import streamlit as st
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from difflib import SequenceMatcher
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3
import threading
import queue
import os
import re

# Load environment variables and configure Google API
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=GOOGLE_API_KEY)

# Initialize speech components
@st.cache_resource
def initialize_speech_components():
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()
    return recognizer, engine

def speak(engine, text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()

def listen(recognizer):
    """Listen for user input and convert speech to text."""
    try:
        with sr.Microphone() as source:
            st.write("Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            st.write("Listening... Please speak now.")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            st.write("Processing audio...")
            
            try:
                text = recognizer.recognize_google(audio)
                st.write(f"Recognized text: {text}")
                return text
            except sr.UnknownValueError:
                st.error("Google Speech Recognition could not understand the audio")
                return None
            except sr.RequestError as e:
                st.error(f"Could not request results from Google Speech Recognition service: {e}")
                return None
                
    except Exception as e:
        st.error(f"Error with microphone: {str(e)}")
        return None

# Initialize Gemini chat
@st.cache_resource
def initialize_gemini_chat():
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        chat = model.start_chat(history=[])
        return chat
    except Exception as e:
        st.error(f"Error initializing Gemini chat: {str(e)}")
        return None

# Cache the models
@st.cache_resource
def initialize_models():
    try:
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if gpt2_tokenizer.pad_token is None:
            gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
        return gpt2_model, gpt2_tokenizer
    except Exception as e:
        st.error(f"Error initializing GPT-2 models: {str(e)}")
        return None, None

# Keep all the existing functions unchanged
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, just say, "I don't have enough information to answer that question." Please don't provide incorrect information.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Keep all existing intent classification code
healthcare_keywords = {
    "symptoms": r"fever|pain|headache|cough|nausea|diabetes|hypertension|infection|fatigue|dizziness|vomiting",
    "medicines": r"paracetamol|ibuprofen|aspirin|amoxicillin|acetaminophen|lisinopril|metformin|omeprazole|sertraline|amlodipine"
}

banking_keywords = {
    "accounts": r"account|savings|current|balance|statement|bank",
    "loans": r"loan|interest|EMI|mortgage|repayment",
    "cards": r"credit card|debit card|PIN|limit|charges",
    "general": r"branch|ATM|IFSC|transfer|deposit|withdraw"
}

def classify_intent(user_input):
    # Check for document-related queries
    doc_keywords = r"document|pdf|file|text|read|extract|analyze"
    if re.search(doc_keywords, user_input, re.IGNORECASE):
        return "document"
    
    # Check for stock-related queries
    stock_keywords = r"stock|share|market|trading|invest|portfolio|dividend|nasdaq|nyse"
    if re.search(stock_keywords, user_input, re.IGNORECASE):
        return "stocks"
    
    # Check healthcare and banking intents
    for category, pattern in healthcare_keywords.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            return "healthcare"
    for category, pattern in banking_keywords.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            return "banking"
    
    return "general"

def get_stock_response():
    return """Here are some ways to track your stocks:
1. Use stock tracking apps like Yahoo Finance or Google Finance
2. Set up price alerts on your trading platform
3. Monitor your portfolio through your broker's website or app
4. Use financial websites for real-time quotes and charts
5. Consider subscribing to stock market news services

Would you like specific recommendations for any of these methods?"""

def get_healthcare_response(query):
    if "fever" in query.lower():
        return """For fever, here are some recommendations:
1. Take rest and stay hydrated
2. Consider taking over-the-counter medications like paracetamol
3. Monitor your temperature regularly
4. Seek medical attention if fever persists over 3 days or exceeds 103°F (39.4°C)
5. Use a cool compress if needed

Would you like more specific information about any of these points?"""
    return None

def display_chat_history(messages):
    for message in messages[-20:]:  # Show last 20 messages
        if message['role'] == 'user':
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin: 10px;">
                    <div style="background-color: #2e7d32; color: white; border-radius: 15px; 
                         padding: 10px; max-width: 70%; font-size: 16px;">
                        {message['text']}
                    </div>
                    <span style="font-size: 20px; margin-left: 10px;">👤</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="display: flex; align-items: flex-start; margin: 10px;">
                    <span style="font-size: 20px; margin-right: 10px;">🤖</span>
                    <div style="background-color: #1a237e; color: white; border-radius: 15px; 
                         padding: 10px; max-width: 70%; font-size: 16px;">
                        {message['text']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

def process_document_query(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        return response["output_text"]
    except Exception as e:
        return f"I encountered an error while processing your document query: {str(e)}"

def handle_chat_input(user_input, intent, chat_instance):
    try:
        if intent == "document":
            return process_document_query(user_input)
        elif intent == "stocks":
            return get_stock_response()
        elif intent == "healthcare":
            healthcare_response = get_healthcare_response(user_input)
            if healthcare_response:
                return healthcare_response
        
        # If no specific handler or healthcare response, use Gemini
        if chat_instance:
            response = chat_instance.send_message(user_input, stream=False)
            return response.text.strip()
        else:
            return "I'm having trouble connecting to my knowledge base. Please try again in a moment."
            
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
    
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def main():
    st.set_page_config(page_title="AI Assistant", layout="wide")
    st.title("Smart Customer Support Chatbot 🤖 with Voice Input")

    # Initialize speech components
    recognizer, engine = initialize_speech_components()

    # Initialize chat instance
    if 'chat_instance' not in st.session_state:
        st.session_state.chat_instance = initialize_gemini_chat()

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'voice_input_received' not in st.session_state:
        st.session_state.voice_input_received = None

    # Sidebar for document upload
    with st.sidebar:
        st.title("📚 Document Processing")
        pdf_docs = st.file_uploader(
            "Upload PDF documents for analysis",
            accept_multiple_files=True,
            type='pdf'
        )
        
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Documents processed successfully!")
                else:
                    st.warning("Please upload some documents first.")

    # Main chat interface
    if not st.session_state.logged_in:
        st.subheader("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if email == "kiit@gmail.com" and password == "kiit123":
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    else:
        display_chat_history(st.session_state.messages)
        
        # Voice input handling
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # If we have voice input, display it in the text input
            default_value = st.session_state.voice_input_received if st.session_state.voice_input_received else ""
            user_input = st.text_input("Type your message here...", value=default_value)
        
        with col2:
            # Add voice input button
            if st.button("🎤 Voice Input"):
                with st.spinner("Listening..."):
                    voice_input = listen(recognizer)
                    if voice_input:
                        st.session_state.voice_input_received = voice_input
                        st.rerun()
                    else:
                        st.error("Could not understand audio. Please try again.")
        
        if st.button("Send") and (user_input or st.session_state.voice_input_received):
            current_input = user_input or st.session_state.voice_input_received
            intent = classify_intent(current_input)
            response = handle_chat_input(current_input, intent, st.session_state.chat_instance)
            
            # Add messages to chat history
            st.session_state.messages.append({"role": "user", "text": current_input})
            st.session_state.messages.append({"role": "assistant", "text": response})
            
            # Convert response to speech
            threading.Thread(target=speak, args=(engine, response)).start()
            
            # Clear the voice input
            st.session_state.voice_input_received = None
            
            st.rerun()

if __name__ == "__main__":
    main()