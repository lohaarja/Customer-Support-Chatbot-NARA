import json
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from difflib import SequenceMatcher

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save the vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save or overwrite existing index

# Function to create a conversational chain for answering questions
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in 
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n 
    Context:\n {context}?\n 
    Question: \n{question}\n 

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and search for answers in PDFs or chat responses
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        print(response)
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"We couldn't answer your question this time. Error: {str(e)}")

# Load medicine dataset from JSON file
@st.cache_data
def load_medicine_dataset(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data["medicines"]

# Sentiment analysis function using VADER
def sentiment_analysis(user_input):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(user_input)
    if sentiment_score['compound'] >= 0.05:
        return "positive"
    elif sentiment_score['compound'] <= -0.05:
        return "negative"
    else:
        return "neutral"

# Function to classify user input into healthcare or banking categories
def classify_user_input(user_input):
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

    for category, pattern in healthcare_keywords.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            return "healthcare", category, re.search(pattern, user_input, re.IGNORECASE).group(0)
    
    for category, pattern in banking_keywords.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            return "banking", category, re.search(pattern, user_input, re.IGNORECASE).group(0)

    return "unknown", "unknown", ""

# Main function for handling chat and PDF uploads
def main():
    st.set_page_config("DOC Q/A")
    st.header("Chat with DOCQA")

    # User question input field
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        
        # Upload PDF files and process them
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, type='pdf')
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)  # This will overwrite any existing index

                st.success("Done")

if __name__ == "__main__":
    main()
