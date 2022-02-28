#The main file
import pickle
import numpy as np
import joblib,os
import sys

def main():
    phish_model = open(r'C:\Users\soumi\OneDrive\Desktop\Summer-Intern-Project\Web-Extension-2\phishing.pkl','rb')
    phish_model_ls = joblib.load(phish_model)
    urlName=sys.argv[1] 
    predictUrl=[urlName]
    prediction=phish_model_ls.predict(predictUrl)
    if prediction==['good']:
        print('SECURE')
    elif prediction==['bad']:
        print('NOT SECURE')

if __name__ == "__main__":
    main()