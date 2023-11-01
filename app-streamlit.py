import streamlit as st
import pandas as pd
import pinecone
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from datetime import datetime

model = SentenceTransformer('all-mpnet-base-v2')
pinecone.init(api_key="ff7232e8-2714-42f8-a392-8fa4c1615b0f", environment="us-west4-gcp-free")
index = pinecone.Index("20231014")
allowed_list = json.load(open("allowed_list.json"))

st.title("FactGrid Cuneiform Secondary Literature")
st.write("FactGrid Secondary Sources provides search functionality for a repository of ancient Near Eastern scholarly works, encompassing both serialized and monographic publications, scholarly journals, festschrifts, and dissertations, whether accessible online or offline. This initiative constitutes our endeavor to establish a mobile library dedicated to the ancient Near East, operating autonomously of institutional affiliations. Currently underway, the project aims to interconnect with primary sources and diverse assertions cited in the FactGrid Cuneiform Project.")

st.write("To use this resource, simply paste a passage of text in the input box and press 'Search'. The results furnish the top list of references and their corresponding similarity scores derived from the most similar matches within a page of documents and across different documents in our growing collection of secondary literature.")

search_input = st.text_area("Enter sentences in your document")
search_button = st.button("Search")

def search_similar(text, top_k=10):
    vector = np.float64(model.encode(text.replace("\n", "")))
    result = index.query(vector=list(vector),top_k=top_k,include_values=False)["matches"]
    for i in result:
        i["page"] = i["id"].split("_")[-1]
        i["id"] = i["id"][:-(len(i["page"])+1)]
    df = pd.DataFrame()
    df["file_name"] = [i["id"] for i in result]
    df["page"] = [i["page"] for i in result]
    df["similarity_score"] = [i["score"] for i in result]
    return df

def insert_text_vector(text, file_name, page_number, user_index):
    now_time = int(datetime.now().strftime("%Y%m%d%H%M%S"))
    vector = np.float64(model.encode(text.replace("\n", "")))
    vector_name = file_name + "_{}".format(page_number)
    index.upsert([(vector_name, list(vector), {"user": user_index, "insert_at": now_time})])

if search_button:
    search_results = search_similar(search_input)
    st.write(search_results)

st.write("Insert a New Doc")
new_doc_input = st.text_area("Enter a new document...", height=100)
user_name = st.text_input("Enter user_name here...insert feature only open to limited members...")
file_name = st.text_input("Enter the name of your file, this will be used as the identifier in your response...")
page_number = st.text_input("Enter the page number, start counting from 1...")
insert_doc_button = st.button("Insert")

if insert_doc_button:
    if user_name not in allowed_list:
        st.write(f"User_name: {user_name} does not have permission to insert, contact support.")
    elif not (new_doc_input and file_name and page_number):
        st.write("Please make sure insert_text, file_name, and page_number are not blank.")
    else:
        try:
            insert_text_vector(new_doc_input, file_name, page_number, allowed_list[str(user_name)])
            st.write(f"Successfully insert {file_name} page {page_number}")
        except Exception as e:
            st.write(f"Failed to insert {file_name} due to error {e}")

st.write("Support")
st.write("For maintenance/participation, please email kenanjeffjiang@gmail.com. If you find our search useful, give us a Thumb Up 👍:) to repo: https://github.com/Kenan-jeff-Jiang/FactGrid-Cuneiform-Lib")
st.write("To make your document safely collected in our library, please email ane.pdf.share@gmail.com")

# Note: Streamlit does not support upvoting functionality directly. You can use a different approach for upvoting if needed.

