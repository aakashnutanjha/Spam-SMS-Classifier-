import streamlit as st
import pickle 

import nltk
from nltk.corpus import stopwords
import string

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_txt(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    t = []
    for i in text:
        if i.isalnum():
            t.append(i)
    
    text = t[:]
    t.clear()
    
    stop_words = set(stopwords.words('english'))
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            t.append(i)
    text = t[:]
    t.clear()
    for i in text:
        t.append(ps.stem(i))
        
        
    return " ".join(t)

tfid= pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_txt(input_sms)

    vector_input= tfid.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not spam")
