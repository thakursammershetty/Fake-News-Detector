import streamlit as st
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
@st.cache_data
def load_data():
    news_df = pd.read_csv('train.csv')
    news_df = news_df.fillna(' ')
    news_df['content'] = news_df['author'] + ' ' + news_df['title']
    return news_df

news_df = load_data()
X = news_df.drop('label', axis=1)
y = news_df['label']

# Define stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming function to content column
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize data
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train,Y_train)

# Streamlit app
st.title('Fake News Detector')

# Description of the web app
st.markdown("""
Welcome to the Fake News Detector. This tool helps you verify the authenticity of news articles.
You can either input the news text directly or upload a file containing the news article.
""")

# Input text area
input_text = st.text_area('Enter news article text below:')

# File uploader
uploaded_file = st.file_uploader("Or upload a text file containing the news article", type="txt")

# Read the file if uploaded
if uploaded_file is not None:
    input_text = uploaded_file.read().decode('utf-8')

# Define the prediction function
def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

# Prediction button
if st.button('Check News'):
    if input_text:
        pred = prediction(input_text)
        if pred == 1:
            st.error('The News is Fake')
        else:
            st.success('The News Is Real')
    else:
        st.warning('Please enter or upload a news article for prediction.')

# Sidebar with additional information
st.sidebar.header('Tips for Using the Fake News Detector')
st.sidebar.markdown("""
- Make sure to enter or upload the full text of the news article for accurate results.
- The tool works best with articles written in English.
- For better accuracy, avoid using texts with heavy formatting or non-standard characters.
""")

st.sidebar.header('About the Model')
st.sidebar.markdown("""
- This model is trained using a large dataset of news articles.
- It uses machine learning algorithms to classify articles as real or fake.
- For more information, visit the [GitHub repository](https://github.com/your-repo).
""")
