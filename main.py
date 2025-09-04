import streamlit as st
import pandas as pd
import cloudpickle
from tensorflow.keras.models import load_model #type: ignore
import cloudpickle

with open('Trained_Model', 'rb') as f:
    model = cloudpickle.load(f)


with open('tokenizer', 'rb') as f:
    tokenizer = cloudpickle.load(f)

trusted_df = pd.read_csv("links.csv")  
trusted_urls = set(trusted_df['Official Website'].str.lower())  

from keras.preprocessing.sequence import pad_sequences

def hybrid_predict(url):
    url_lower = url.lower()
    
    if url_lower in trusted_urls:
        return "Safe"
    
    seq = tokenizer.texts_to_sequences([url_lower])
    padded = pad_sequences(seq, maxlen=200)
    pred = model.predict(padded)[0][0]
    
    if pred > 0.5:
        return "Safe"
    else:
        return "UnSafe/ Fraud"

st.title("🔗 URL Safety Checker")
st.write("Paste a URL below and check if it is safe or phishing.")

user_url = st.text_input("Enter URL here:")

if st.button("Check URL"):
    if user_url:
        result = hybrid_predict(user_url)
        if(result == 'Safe'):
            st.success("Safe url (Legitimate)")
        else:
            st.warning("Not safe url (Fraud)")
    else:

        st.warning("Please enter a URL to check.")
