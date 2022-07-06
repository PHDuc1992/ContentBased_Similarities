import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import warnings
from gensim import corpora, models, similarities
import jieba
import re

# 1. Read data
products_raw = pd.read_csv("ProductRaw.csv")
reviews_raw = pd.read_csv("ReviewRaw.csv")
products = pd.read_csv("Product_clean.csv")
reviews = pd.read_csv("Review_clean.csv")

#--------------
# GUI
st.title("Data Science Project")
st.write("##  Recommender System Product&Review")

# Upload file
uploaded_file = st.file_uploader("Choose a file", type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data.to_csv("Product_new.csv", index = False)

# 2. Data pre-processing
# Tokenize (split) the sentiment into words
product_information_token = [[text for text in x.split()] for x in products.product_infomation]
# Obtain the number of features based on dictionary: use corpora.Dictionary
dictionary=corpora.Dictionary(product_information_token)
# Numbers of features (word) in dictionary
feature_cnt=len(dictionary.token2id)
# Obtain corpus based on dictionary (dense matrix: ma tran thua)
corpus=[dictionary.doc2bow(text) for text in product_information_token]
# Use TF-IDF Model to process corpus, obtaining index
tfidf = models.TfidfModel(corpus)
# Tính toán sự tương tự trong ma trận thưa thớt
index = similarities.SparseMatrixSimilarity(tfidf[corpus],num_features = feature_cnt)




# 3. Build recommendation
def recommendation (view_product, dictionary, tfidf, index):
    # Convert search words into Sparse Vectors
    view_product = view_product.lower().split()
    kw_vector = dictionary.doc2bow(view_product)
    print("View product 's vector:")
    print(kw_vector)
    # Similarity calculation
    sim = index[tfidf[kw_vector]]
    
    # print result
    list_id = []
    list_score = []
    for i in range(len(sim)):
        list_id.append(i)
        list_score.append(sim[i])
    
    df_result = pd.DataFrame({'id': list_id,
                              'score': list_score})
    
    # 10 highest scores
    five_highest_score = df_result.sort_values(by='score', ascending=False).head(11)
    print("Five highest scores:")
    print(five_highest_score)
    print("Ids to list:")
    idToList = list(five_highest_score['id'])
    print(idToList)
    
    products_find = products[products.index.isin(idToList)]
    results = products_find[['item_id','name']]
    results = pd.concat([results, five_highest_score], axis=1).sort_values(by='score', ascending=False)
    return results

# GUI
menu = ["Business Objective", "Build Project", "Prediction"]
choice = st.sidebar.selectbox('Content', menu)
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ###### Recommendation is one of the most common natural language processing tasks for e-commercial company. With the advancements in machine learning and natural language processing techniques, it is now possible to give familiar product suggestion to customer.
    """)  
    st.write("""###### => Problem/ Requirement: Use Machine Learning algorithms in Python to give 5 suggestion product to customer when customer choose one product.""")
    st.image("Recommend product-pop-up.jpg")
elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("##### 1. Some data")
    st.write("##### 1.1 products_raw")
    st.dataframe(products_raw.head(3))
    st.write("##### 1.1 reviews_raw")
    st.dataframe(reviews_raw.head(3))

    st.write("##### 2. Visualize Products And Reviews")
    st.write("##### 2. Visualize Products_price")
    fig1 = products.price.plot(kind='hist', bins=20)
    st.pyplot(fig1.figure)
    st.write("####### Range of price is big")
    st.write("####### Most of price is <3.000K")

elif choice == 'Prediction':
    product_list = products['name'].values
    product_ID = st.selectbox( "Type or select a product from the dropdown", product_list )

    if st.button('Show Recommendation'):
        product_selection = products[products.name == product_ID]
        name_description_pre = product_selection['product_infomation'].to_string(index=False)
        results = recommendation(name_description_pre, dictionary, tfidf, index)
        results = results[results.item_id!=product_ID]
        results

