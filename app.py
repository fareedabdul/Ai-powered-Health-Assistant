import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the chatbot model
chatbot = pipeline("text-generation", model="distilgpt2")

def healthcare_chatbot(user_input):
    user_input = user_input.lower()
    
    # Predefined responses for common healthcare-related queries
    responses = {
        "covid": "COVID-19 is a highly contagious respiratory illness caused by the SARS-CoV-2 virus. Follow safety guidelines provided by health authorities.",
        "Bird flu": "Dont eat Chicken please",
        "symptom": "I recommend consulting a doctor for an accurate diagnosis.",
        "appointment": "Would you like to schedule an appointment with a healthcare professional?",
        "medication": "Make sure to follow your doctor’s prescription. If you have concerns, consult a medical expert.",
        "emergency": "In case of an emergency, please contact the nearest hospital or dial emergency services immediately.",
        "fever": "For fever, stay hydrated and rest. If it persists, consult a doctor.",
        "headache": "Headaches can be caused by stress or dehydration. Drink water and take rest. If severe, seek medical advice."
    }
    
    for keyword, response in responses.items():
        if keyword in user_input:
            return response
    
    # Generate response using the chatbot model for general queries
    response = chatbot(user_input, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

def main():
    st.set_page_config(page_title="Healthcare Chatbot", page_icon="💊", layout="centered")
    
    st.markdown(
        """
        <style>
        .main-container {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
        }
        .stTextInput, .stButton {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.markdown("<h1 style='text-align: center; color: #2E8B57;'>Healthcare Assistant Chatbot</h1>", unsafe_allow_html=True)
    st.write("🤖 Welcome to your AI-powered healthcare assistant! Ask me any health-related questions.")
    
    user_input = st.text_input("How can I assist you today?", placeholder="Type your question here...")
    
    if st.button("Ask Now", help="Click to get a response"):
        if user_input.strip():
            st.markdown(f"**👤 You:** {user_input}")
            with st.spinner("Processing your query..."):
                response = healthcare_chatbot(user_input)
            st.success("Response Generated!")
            st.markdown(f"**💬 Chatbot:** {response}")
        else:
            st.warning("Please enter a valid question.")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("⚠️ Disclaimer: This chatbot provides general health advice. Always consult a medical professional for serious concerns.")
    
if __name__ == "__main__":
    main()
