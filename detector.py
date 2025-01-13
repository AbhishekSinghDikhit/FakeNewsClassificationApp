
import streamlit as st
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

vector_form = pickle.load(open('vectorizer.pkl', 'rb'))
load_model = pickle.load(open('DTC.pkl', 'rb'))

# def stemming(content):
#     con=re.sub('[^a-zA-Z]', ' ', content)
#     con=con.lower()
#     con=con.split()
#     con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
#     con=' '.join(con)
#     return con
def wordopt(text):
    text = text.lower()
    #remove URLs
    text = re.sub(r'https?://\S+|www\.\S+','',text)
    #remove HTML tags
    text = re.sub(r'<.*?>','',text)
    #remove punctuation
    text = re.sub(r'[^\w\s]','',text)
    #remove digits
    text = re.sub(r'\d','',text)
    #remove newline char
    text = re.sub(r'\n','',text)
    return text

def output_label(n):
  if n==0:
    return st.warning('The news is Fake')
  elif n==1:
    return st.success('The news is Real')

def fake_news(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test['text'].apply(wordopt)
    new_x_test = new_def_test['text']
    new_xv_test = vector_form.transform(new_x_test)  
    # news=stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return output_label(prediction[0])



if __name__ == '__main__':
    
    st.title('Fake News Classification app ')
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here", "",height=150)
    predict_btt = st.button("predict")
    if predict_btt:
        prediction_class=fake_news(sentence)
        print(prediction_class)
        if prediction_class == [0]:
            st.success('True')
        if prediction_class == [1]:
            st.warning('Fake')