import pickle


with open('phishing.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)