import json
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from difflib import SequenceMatcher
from dotenv import load_dotenv
import os

load_dotenv()

@st.cache_resource
def load_model_and_tokenizer():
    model = GPT2LMHeadModel.from_pretrained("gpt2")  
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

@st.cache_data
def load_medicine_dataset(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data["medicines"]

@st.cache_data
def load_banking_faqs(json_file):
    with open(json_file, "r", encoding="utf-8") as file:
        return json.load(file)

def sentiment_analysis(user_input):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(user_input)
    if sentiment_score['compound'] >= 0.05:
        return "positive"
    elif sentiment_score['compound'] <= -0.05:
        return "negative"
    else:
        return "neutral"

def get_medicine_by_symptom(symptom, medicines):
    matching_medicines = [
        medicine["name"]
        for medicine in medicines
        if "uses" in medicine and any(symptom.lower() in use.lower() for use in medicine["uses"])
    ]
    if matching_medicines:
        return f"The following medicines are commonly used for {symptom}: {', '.join(matching_medicines)}."
    return get_gemini_response(f"What medicines can be used for {symptom}?")

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

def classify_user_input(user_input):
    for category, pattern in healthcare_keywords.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            return "healthcare", category, re.search(pattern, user_input, re.IGNORECASE).group(0)
    for category, pattern in banking_keywords.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            return "banking", category, re.search(pattern, user_input, re.IGNORECASE).group(0)
    return "unknown", "unknown", ""

class BankingChatbot:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.eval()
        self.patterns = []
        self.responses = []

    def load_training_data(self, json_file):
        try:
            data = load_banking_faqs(json_file)
            for category in data.get("bank", []):
                for item in data["bank"][category]:
                    question, answer = item[0], item[1]
                    self.patterns.append(question)
                    self.responses.append(answer)
        except Exception as e:
            print(f"Error loading training data: {e}")

    def find_best_match(self, user_input):
        best_match = None
        highest_similarity = 0
        for i, pattern in enumerate(self.patterns):
            similarity = SequenceMatcher(None, user_input, pattern).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = self.responses[i]
        if best_match:
            return best_match
        return get_gemini_response(f"Can you answer this banking-related query: {user_input}?")
    
def display_chat_history(messages, max_messages=20):
    for message in messages[-max_messages:]:
        if message['role'] == 'user':
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin: 10px;">
                    <div style="
                        background-color: rgb(22, 105, 97); 
                        color: white; 
                        border-radius: 15px; 
                        padding: 10px; 
                        max-width: 70%; 
                        display: inline-block;
                        font-size: 16px;
                    ">
                        {message['text']}
                    </div>
                    <span style="font-size: 20px; margin-left: 10px;">🙂</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="display: flex; align-items: flex-start; margin: 10px;">
                    <span style="font-size: 20px; margin-right: 10px;">🤖</span>
                    <div style="
                        background-color: rgba(88, 83, 83, 0.37); 
                        color: white; 
                        border-radius: 15px; 
                        padding: 10px; 
                        max-width: 70%; 
                        display: inline-block;
                        font-size: 16px;
                    ">
                        {message['text']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

def main_chatbot():
    st.title("Smart Customer Support Chatbot")
    medicines = load_medicine_dataset("medicine-dataset-part1.json")
    banking_chatbot = BankingChatbot()
    banking_chatbot.load_training_data("bank_faqs[1].json")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    display_chat_history(st.session_state.messages)

    user_input = st.text_input("Type your query here...")
    if st.button("Send") and user_input:
        intent, category, keyword = classify_user_input(user_input)

        if intent == "healthcare":
            response = get_medicine_by_symptom(keyword, medicines) if category == "symptoms" else get_gemini_response(f"Provide information about {keyword}.")
        elif intent == "banking":
            response = banking_chatbot.find_best_match(user_input)
        else:
            response = get_gemini_response(user_input)

        st.session_state.messages.append({"role": "user", "text": user_input, "emoji": "🙂"})
        st.session_state.messages.append({"role": "bot", "text": response, "emoji": "🤖"})
        st.rerun()

def login_page():
    st.title("Login Page")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if email == "kiit@gmail.com" and password == "kiit123":
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid email or password.")

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if st.session_state.logged_in:
        main_chatbot()
    else:
        login_page()

@st.cache_resource
def initialize_gemini_chat():
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    return model.start_chat(history=[])

chat = initialize_gemini_chat()

def get_gemini_response(question):
    try:
        response = chat.send_message(question, stream=False)
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    main()