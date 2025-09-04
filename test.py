import pandas as pd
import cloudpickle

with open('Trained_Model', 'rb') as f:
    model = cloudpickle.load(f)

with open('tokenizer', 'rb') as f:
    tokenizer = cloudpickle.load(f)

# Load trusted links
trusted_df = pd.read_csv("links.csv")  # assuming one column with URLs
trusted_urls = set(trusted_df['Official Website'].str.lower())  # convert to lowercase for uniformity

from keras.preprocessing.sequence import pad_sequences

def hybrid_predict(url):
    url_lower = url.lower()
    
    # Step 1: Check if URL is in trusted list
    if url_lower in trusted_urls:
        return "Safe"
    
    # Step 2: If not, predict using the LSTM model
    seq = tokenizer.texts_to_sequences([url_lower])
    padded = pad_sequences(seq, maxlen=200)
    pred = model.predict(padded)[0][0]
    
    if pred > 0.5:
        return "Safe"
    else:
        return "UnSafe/ Fraud"
    

import pandas as pd
from keras.preprocessing.sequence import pad_sequences

# Load URLs
links_df = pd.read_csv("links.csv")  # assumes column name is 'url'
links_list = links_df['Official Website'].str.lower().tolist()  # lowercase for consistency
# Loop through all URLs and print predictions
phishing_count = 0
legitimate_count = 0
for url in links_list:
    result = hybrid_predict(url)
    if(result == "Phishing"):
        phishing_count += 1
    else:
        legitimate_count += 1
    print(f"URL: {url} --> Prediction: {result}")

print(f"Total UnSafe/ Fraud URLs: {phishing_count}")
print(f"Total Safe URLs: {legitimate_count}")