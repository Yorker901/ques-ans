import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import requests

# Streamlit UI
st.title("PDF Summarization and Q&A")

# File uploader for the PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Question input
query = st.text_input("Enter your question:")

if uploaded_file is not None and query:
    # Read PDF
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Display number of pages
    st.write(f"Number of pages: {len(reader.pages)}")

    # Show extracted text
    if st.checkbox("Show extracted text"):
        st.write(text)

    # Sentence Transformer model for embedding
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    emb = model.encode(text)

    # Elasticsearch connection
    cloud_id = "ac668387facb455d9201540f7bcdccf3:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDM5OGQ1NGMzMzZlZTQ0MGM5MGVjM2VjYmIwYjc0MWRjJGQ4NDgxNTA2MWM1NDQwYjA4YmE3NTAxMGQ1YzM3MGJl"
    es = Elasticsearch(
        cloud_id=cloud_id,
        basic_auth=("your_username", "your_password")  # Replace with your actual username and password
    )

    # Index creation (if it doesn't exist)
    index = "summarization_pdf"
    es.indices.create(index=index, ignore=400)

    # Index the document
    doc = {
        "text_embedding": emb.tolist(),  # Convert numpy array to list for JSON serialization
        "text": text
    }
    es.index(index=index, body=doc)

    # Encode the query
    query_emb = model.encode(query)

    # Elasticsearch query for similarity search
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
        st.write("Answer based on the PDF content:")
        st.write(get_text)

    # Hugging Face model API call for more sophisticated answer generation
    def query_mistral(payload):
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
        API_TOKEN = "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"
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            st.error(f"Error: Received status code {response.status_code}")
            return None

        data = response.json()
        generated_text = data[0]['generated_text']
        prompt_index = generated_text.find('[/INST]')
        if prompt_index != -1:
            generated_text = generated_text[prompt_index + len('[/INST]'):]
        return generated_text

    # Prepare the prompt for the model
    negResponse = "I'm unable to answer the question based on the information I have."
    prompt = f"[INST] You are a helpful Q&A assistant. Your task is to answer this question: {query}. Use only the information from this text: {get_text}, Provide answer strictly in HTML format, If the answer is not contained in the text, reply with {negResponse}. Response [/INST]"
    
    if st.button("Generate Answer"):
        data = query_mistral({"parameters": {"max_new_tokens": 2000}, "inputs": prompt})
        st.write("Generated Answer:")
        st.write(data)

