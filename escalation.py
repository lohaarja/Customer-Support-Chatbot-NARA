import json
import re
from difflib import SequenceMatcher
from typing import List, Tuple
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import urllib.parse

# Define BankingChatbot class first
class BankingChatbot:
    def _init(self):  # Fixed method name from _init to _init_
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model.eval()
        except Exception as e:
            st.error(f"Error initializing banking chatbot: {str(e)}")
            
        self.patterns = []
        self.responses = []

    def load_training_data(self, json_file: str) -> None:  # Added return type annotation
        try:
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)

            for category in data.get("bank", []):
                for item in data["bank"][category]:
                    self.patterns.append(item[0])
                    self.responses.append(item[1])
        except Exception as e:
            st.error(f"Error loading banking training data: {str(e)}")

    def generate_response(self, user_input: str) -> str:
        try:
            match = self.find_best_match(user_input)
            if match:
                return f"{match}\n\nFor further assistance, contact our expert at +6372315197"

            input_text = f"User: {user_input}\nBot:"
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            
            output = self.model.generate(
                input_ids,
                max_length=100,
                temperature=0.7,
                top_k=30,
                top_p=0.9,
                no_repeat_ngram_size=2,
                do_sample=True,
            )
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return f"{response}\n\nFor further assistance, contact our expert at +6372315197"
        except Exception as e:
            return f"I apologize, but I'm having trouble generating a response. Please contact our expert at +6372315197"

    def find_best_match(self, user_input: str) -> str | None:  # Added proper return type annotation
        best_match = None
        highest_similarity = 0
        for i, pattern in enumerate(self.patterns):
            similarity = SequenceMatcher(None, user_input.lower(), pattern.lower()).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = self.responses[i]
        return best_match if highest_similarity > 0.7 else None

def load_medicine_dataset(json_file: str) -> List[dict]:
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data["medicines"]
    except FileNotFoundError:
        st.error(f"Could not find {json_file}. Please ensure it exists in the correct directory.")
        return []
    except json.JSONDecodeError:
        st.error(f"Error reading {json_file}. Please ensure it's properly formatted JSON.")
        return []

def sentiment_analysis(user_input: str) -> str:
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(user_input)
    if sentiment_score['compound'] >= 0.05:
        return "positive"
    elif sentiment_score['compound'] <= -0.05:
        return "negative"
    else:
        return "neutral"

def needs_doctor_consultation(symptom: str) -> bool:
    severe_symptoms = [
        'fever', 'infection', 'diabetes', 'hypertension', 'pain',
        'vomiting', 'dizziness', 'fatigue'
    ]
    return any(s in symptom.lower() for s in severe_symptoms)

def get_medicine_by_symptom(symptom: str, medicines: List[dict]) -> Tuple[str, bool]:
    for medicine in medicines:
        if isinstance(medicine, dict) and "uses" in medicine:
            for use in medicine["uses"]:
                if symptom.lower() in use.lower():
                    return f"Yes, {medicine['name']} is commonly used for {symptom}.", needs_doctor_consultation(symptom)
    return "Sorry, I couldn't find a medicine for that symptom.", needs_doctor_consultation(symptom)

def empathetic_response(sentiment: str, user_input: str, response: str) -> str:
    if sentiment == "positive":
        return f"That's great you're asking about this! {response}"
    elif sentiment == "negative":
        return f"I'm really sorry you're feeling this way. {response}"
    else:
        return f"Thanks for your question! {response}"

def predict_intent_healthcare(user_input: str) -> Tuple[str, str]:
    symptoms = r"fever|pain|headache|cough|nausea|diabetes|hypertension|infection|fatigue|dizziness|vomiting"
    medicines = r"paracetamol|ibuprofen|aspirin|amoxicillin|acetaminophen|lisinopril|metformin|omeprazole|sertraline|amlodipine"

    symptom_match = re.search(symptoms, user_input, re.IGNORECASE)
    if symptom_match:
        return "symptom", symptom_match.group(0).lower()

    medicine_match = re.search(medicines, user_input, re.IGNORECASE)
    if medicine_match:
        return "medicine", medicine_match.group(0).lower()

    return "unknown", ""

def create_email_link(symptom: str) -> str:
    email_body = (
        f"Dear Doctor,\n\n"
        f"A patient has reported symptoms related to {symptom} and requires medical consultation.\n\n"
        f"Please review and provide medical advice.\n\n"
        f"Best regards,\nHealthcare Support System"
    )
    
    encoded_body = urllib.parse.quote(email_body)
    encoded_subject = urllib.parse.quote(f"Urgent Medical Consultation Required: {symptom}")
    
    return f"mailto:doctor@healthcare.com?subject={encoded_subject}&body={encoded_body}"

def process_message(user_input: str, medicines: List[dict], banking_chatbot: BankingChatbot) -> None:  # Added return type
    if not hasattr(st.session_state, 'messages'):  # Added safety check
        st.session_state.messages = []
        
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "text": user_input, "emoji": "😊"})

    # Process greetings
    if any(greeting in user_input.lower() for greeting in ["hello", "hi", "hey", "greetings"]):
        response = "Hello! How can I assist you today?"
        st.session_state.messages.append({"role": "bot", "text": response, "emoji": "🤖"})
        return

    # Process healthcare or banking query
    intent_type, detected_thing = predict_intent_healthcare(user_input)
    
    if intent_type == "symptom":
        response, needs_consultation = get_medicine_by_symptom(detected_thing, medicines)
        sentiment = sentiment_analysis(user_input)
        empathetic_reply = empathetic_response(sentiment, user_input, response)
        
        if needs_consultation:
            email_link = create_email_link(detected_thing)
            full_response = f"{empathetic_reply}\n\nBased on your symptoms, we recommend consulting a doctor. [Click here to request a medical consultation]({email_link})"
        else:
            full_response = empathetic_reply
            
        st.session_state.messages.append({"role": "bot", "text": full_response, "emoji": "🤖"})
    
    elif intent_type == "medicine":
        medicine_found = False  # Added flag to track if medicine was found
        for medicine in medicines:
            if detected_thing.lower() in medicine['name'].lower():
                response = f"{medicine['name']} is used for {', '.join(medicine['uses'])}. Side effects: {', '.join(medicine['side_effects'])}."
                sentiment = sentiment_analysis(user_input)
                empathetic_reply = empathetic_response(sentiment, user_input, response)
                st.session_state.messages.append({"role": "bot", "text": empathetic_reply, "emoji": "🤖"})
                medicine_found = True
                break
        
        if not medicine_found:  # Added response for when medicine is not found
            st.session_state.messages.append({"role": "bot", "text": "I'm sorry, I couldn't find information about that medicine.", "emoji": "🤖"})
    
    else:
        banking_response = banking_chatbot.generate_response(user_input)
        st.session_state.messages.append({"role": "bot", "text": banking_response, "emoji": "🤖"})

def main() -> None:  # Added return type annotation
    # Configure page before any other Streamlit commands
    st.set_page_config(
        page_title="Smart Customer Support Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("Smart Customer Support Chatbot")
    st.subheader("Ask your question below:")

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "bot",
            "text": "Hello! I'm your chatbot assistant. Feel free to ask about symptoms, medicines, or banking information.",
            'emoji': '🤖'
        })

    # Load data and initialize chatbot
    try:
        medicines = load_medicine_dataset("medicine-dataset-part1.json")
        banking_chatbot = BankingChatbot()
        banking_chatbot.load_training_data("bank_faqs.json")
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return

    # Display chat messages
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin: 5px;">
                    <div style="background-color: rgb(22, 105, 97); color: white; border-radius: 10px; padding: 10px; max-width: 70%;">
                        {message['text']}
                    </div>
                    <span style="font-size: 20px; margin-left: 10px;">{message['emoji']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="display: flex; align-items: flex-start; margin: 5px;">
                    <span style="font-size: 20px; margin-right: 10px;">{message['emoji']}</span>
                    <div style="background-color: rgba(88, 83, 83, 0.37); color: white; border-radius: 10px; padding: 10px; max-width: 60%;">
                        {message['text']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Handle user input
    def process_input() -> None:  # Added return type annotation
        if hasattr(st.session_state, 'user_input'):  # Added safety check
            user_input = st.session_state.user_input
            if user_input and user_input.strip():  # Added input validation
                process_message(user_input, medicines, banking_chatbot)

    # Input field and send button
    col1, col2 = st.columns([5, 1])
    with col1:
        st.text_input("Type here...", key="user_input", on_change=process_input)
    with col2:
        if st.button("Send"):
            process_input()

if __name__ == "_main":  # Fixed from _main to _main_
    main()