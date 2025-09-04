import pandas as pd

# Load dataset
data = pd.read_csv("url_dataset.csv")

# Separate phishing and legitimate
phishing = data[data['type'] == 'phishing']
legit = data[data['type'] == 'legitimate']

# Undersample legitimate to match phishing count
legit_sample = legit.sample(n=len(phishing), random_state=42)

# Combine back
balanced_data = pd.concat([phishing, legit_sample])

# Shuffle rows
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

print(balanced_data['type'].value_counts())

# Save balanced dataset
balanced_data.to_csv("balanced_url_dataset.csv", index=False)
