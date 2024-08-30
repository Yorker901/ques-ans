import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import requests

# Streamlit app title
st.title("PDF Summarization and Elasticsearch Query")

# File uploader for PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Read the PDF file
    reader = PdfReader(uploaded_file)
    st.write(f"Number of pages: {len(reader.pages)}")
    
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    st.write("Extracted Text:")
    st.text_area("PDF Text", text, height=200)
    
    # Load the sentence transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    emb = model.encode(text)
    
    st.write("Text Embedding:")
    st.write(emb)
    
    # Define the Cloud ID and credentials
    cloud_id = "ac668387facb455d9201540f7bcdccf3:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDM5OGQ1NGMzMzZlZTQ0MGM5MGVjM2VjYmIwYjc0MWRjJGQ4NDgxNTA2MWM1NDQwYjA4YmE3NTAxMGQ1YzM3MGJl"
    es = Elasticsearch(
        cloud_id=cloud_id,
        basic_auth=("Aarizkhan2580@gmail.com", "Arizkhan@901")  # Replace with your actual username and password
    )
    
    # Delete and recreate the index
    es.indices.delete(index="summarization_pdf", ignore_unavailable=True)
    es.indices.create(
        index="summarization_pdf",
        mappings={
            "properties": {
                "text": {
                    "type": "text"    
                },
                "username": {
                    "type": "text"
                },
                "text_embedding": {
                    "type": "dense_vector",
                    "dims": 384
                },
                "fId": {
                    "type": "text"
                },
                "fileName": {
                    "type": "text"
                },
                "pageNo": {
                    "type": "text"
                },
                "tables": {
                    "type": "keyword"
                }
            }
        }
    )
    
    # Index the document
    index = "summarization_pdf"
    doc = {"text_embedding": emb, "text": text}
    es.index(index=index, document=doc)
    
    st.write("Document indexed successfully.")
    
    # Query input
    query = st.text_input("Enter your query:", "When did the Microsoft systems cause trouble?")
    
    if query:
        query_emb = model.encode(query)
        
        search_query = {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'text_embedding') + 1.0",
                        "params": {
                            "query_vector": query_emb
                        }
                    }
                }
            },
            "size": 1
        }
        
        # Perform the search
        response = es.search(index=index, body=search_query)
        
        for hit in response['hits']['hits']:
            get_text = hit['_source']['text']
            st.write("Search Result:")
            st.write(get_text)
        
        # Use Mistral model to generate an answer
        def query_mistral(payload):
            API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
            API_TOKEN = "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"
            headers = {"Authorization": f"Bearer {API_TOKEN}"}
            response = requests.post(API_URL, headers=headers, json=payload)
            try:
                if response.status_code != 200:
                    st.error(f"Error: Received status code {response.status_code}")
                    return None
                data = response.json()
                generated_text = data[0]['generated_text']
                prompt_index = generated_text.find('[/INST]')
                if prompt_index != -1:
                    generated_text = generated_text[prompt_index + len('[/INST]'):]
                return generated_text
            except requests.exceptions.RequestException as e:
                st.error(f"Request exception: {e}")
                return None
            except ValueError as e:
                st.error(f"Value error: {e}")
                return None
        
        negResponse = "I'm unable to answer the question based on the information I have."
        prompt = f"[INST] You are a helpful Q&A assistant. Your task is to answer this question: {query}. Use only the information from this text: {get_text}. Provide the answer strictly in HTML format. If the answer is not contained in the text, reply with '{negResponse}'. Response [/INST]"
        max_new_tokens = 2000
        
        data = query_mistral({"parameters": {"max_new_tokens": max_new_tokens}, "inputs": prompt})
        
        st.write("Generated Answer:")
        st.write(data)

# To run the Streamlit app, you would execute `streamlit run your_script_name.py` in your terminal.
