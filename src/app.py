import os
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import pinecone
import requests
API_URL = "https://krdbb3bjmwqzgvq8.us-east-1.aws.endpoints.huggingface.cloud"
key = os.environ.get("HUGGING_FACE")
headers = {"Authorization": f"Bearer {key}"}

pinecone.init(api_key="ff7232e8-2714-42f8-a392-8fa4c1615b0f", environment="us-west4-gcp-free")
index = pinecone.Index("20231014")
import numpy as np
from dash import dash_table
import json
from datetime import datetime
import requests
allowed_list = json.load(open("allowed_list.json"))
button_style = {
    'background-color': 'lightblue',
    'color': 'white',
    'border': 'none',
    'border-radius': '5px',
    'padding': '10px 20px',
    'cursor': 'pointer',
    'font-size': '16px',
    'margin-top': '20px',
    'margin-left': '20px'
}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def search_similar(text, top_k=10):
    vector = query({"inputs": text.replace("\n", "")})["embeddings"]
    result = index.query(vector=vector,top_k=top_k,include_values=False)["matches"]
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
    vector = query({"inputs": text.replace("\n", "")})["embeddings"]
    vector_name = file_name + "_{}".format(page_number)
    index.upsert([(vector_name, vector, {"user": user_index, "insert_at": now_time})])

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define the layout of the app
app.layout = html.Div([
    html.H1("FactGrid Cuneiform Secondary Literature", style={'margin-left': '20px', 'margin-top': '20px'}),
    html.P("FactGrid Secondary Sources provides search functionality for a repository of ancient Near Eastern scholarly works, encompassing both serialized and monographic publications, scholarly journals, festschrifts, and dissertations, whether accessible online or offline. This initiative constitutes our endeavor to establish a mobile library dedicated to the ancient Near East, operating autonomously of institutional affiliations. Currently underway, the project aims to interconnect with primary sources and diverse assertions cited in the FactGrid Cuneiform Project.", style={'margin-left': '20px', 'margin-top': '10px'}),
    html.P("To use this resource, simply paste a passage of text in the input box and press <search>. The results furnish the top list of references and their corresponding similarity scores derived from the most similar matches within a page of documents and across different documents in our growing collection of secondary literature.", style={'margin-left': '20px', 'margin-top': '10px'}),
    dcc.Input(id='search-input', type='text', placeholder='Enter sentences in your document', style={'margin-left': '20px', 'margin-top': '10px', 'width': '80%', 'height': '50px'}),
    dbc.Button(id='search-button', n_clicks=0, children='Search', color="primary", outline=True, style={'margin-left': '20px'}),

    html.Div(id='search-output'),
    html.Div(
        dash_table.DataTable(id='search-table', style_cell={'textAlign': 'left'}),
        style={'display': 'flex', 'justify-content': 'center', 'margin-top': '20px'}
    ),
    html.Hr(style={'border-top': '3px solid #00ff00', 'margin-top': '20px'}),  # Colorful line
    html.H2("Insert a New Doc", style={'margin-left': '20px', 'margin-top': '20px'}),  # New section
    dcc.Input(id='new-doc-input', type='text', placeholder='Enter a new document...',
              style={'margin-left': '20px', 'margin-top': '20px', 'width': '95%', 'height': '50px', 'height': '100px'}),
    html.Div([dcc.Input(id='user_name', type='text', placeholder='Enter user_name here...insert feature only open to limited members...',
              style={'margin-left': '20px', 'margin-top': '20px', 'width': '30%', 'height': '50px'}),
              dcc.Input(id='file_name', type='text', placeholder='Enter the name of your file, this will be used as the identifier in your response...',
              style={'margin-left': '20px', 'margin-top': '20px', 'width': '30%', 'height': '50px'}),
              dcc.Input(id='page_number', type='text', placeholder='Enter the page number, start counting from 1...',
              style={'margin-left': '20px', 'margin-top': '20px', 'width': '30%', 'height': '50px'}),
            dbc.Button(id='insert-doc-button', n_clicks=0, children='Insert', color="primary", outline=True, style={'margin-left': '20px'})]),  # New section button
    html.Div(id='insert-doc-output', style={'margin-left': '20px', 'margin-top': '20px'}),  # Display the result of the insert-doc-button click
    html.Hr(style={'border-top': '3px solid #00ff00', 'margin-top': '20px'}),  # Colorful line
    html.H2("Support", style={'margin-left': '20px', 'margin-top': '20px'}),  # Text header for Support
    html.P("For maintenance/participation, please email kenanjeffjiang@gmail.com. If you find our search useful, give us a Thumb Up 👍:) to repo: https://github.com/Kenan-jeff-Jiang/FactGrid-Cuneiform-Lib", style={'margin-left': '20px', 'margin-top': '10px'}),
    html.P("To make your document safely collected in our library, please email ane.pdf.share@gmail.com", style={'margin-left': '20px', 'margin-top': '10px'}),
    # html.Div([html.Button(id='thumb-up-button', n_clicks=0, children='👍', style=button_style),
    #           html.Div(id='thumb-up-count', style={'margin-left': '20px', 'font-size': '20px', 'margin-top': '25px'})], 
    #           style={'display': 'flex', 'justify-content': 'flex-end', 'margin-top': '20px', 'margin-right': '20px'})
])

# Define the callback to update search results
@app.callback(
    [Output("search-table", "data"),
     Output('search-button', 'n_clicks')],
    [Input('search-button', 'n_clicks'),
    Input("search-input", "value")]
)
def update_search_results(n_clicks, search_query):
    print("request recieved")
    if search_query and n_clicks > 0:
        # Implement your search logic here
        search_results = search_similar(search_query).to_dict('records')
    else:
        search_results = []
    return search_results, 0

@app.callback(
    [Output("insert-doc-output", "children"),
     Output('insert-doc-button', 'n_clicks')],
    [Input('insert-doc-button', 'n_clicks'),
    Input("new-doc-input", "value"),
    Input("user_name", "value"),
    Input("file_name", "value"),
    Input("page_number", "value")]
)
def insert_vector(n_clicks, insert_text, user_name, file_name, page_number):
    # Replace this with your actual search function
    if user_name not in allowed_list and n_clicks > 0:
        return f"User_name: {user_name} do not have permission to insert, contact support.", 0
    elif (insert_text is None or file_name is None or page_number is None) and n_clicks > 0:
        return "Please make sure insert_text, file_name and page_number are not blank", 0
    elif n_clicks > 0:
        try:
            insert_text_vector(insert_text, file_name, page_number, allowed_list[str(user_name)])
            return f"Successfully insert {file_name} page {page_number}", 0
        except Exception as e:
            return f"Failed to insert {file_name} due to error {e}", 0
    return '', 0

if __name__ == "__main__":
    server.run(port=8080)