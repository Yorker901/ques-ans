import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import requests
import os

# Streamlit app title
st.title("QuerySage")

# File upload allowing multiple files
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_text = ""

    # Read and combine text from all uploaded PDFs
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            all_text += page.extract_text()

    # Load sentence transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    emb = model.encode(all_text)

    # Elasticsearch connection (using API key stored in Streamlit secrets)
    es = Elasticsearch(
        cloud_id=st.secrets["ELASTIC_CLOUD_ID"],  # Stored in Streamlit secrets
        api_key=st.secrets["ELASTIC_API_KEY"]     # Stored in Streamlit secrets
    )

    # Create or update the Elasticsearch index
    index = "summarization_pdf"
    
    try:
        es.options(ignore_status=[400, 404]).indices.delete(index=index)
        es.indices.create(
            index=index,
            mappings={
                "properties": {
                    "text": {"type": "text"},
                    "text_embedding": {"type": "dense_vector", "dims": 384},
                }
            }
        )
    except Exception as e:
        st.error(f"Error handling Elasticsearch index: {e}")
    
    doc = {"text_embedding": emb, "text": all_text}
    es.index(index=index, document=doc)
    es.indices.refresh(index=index)

    # User query input
    query = st.text_input("Enter your question:")

    if query:
        query_emb = model.encode(query)

        # Search query
        search_query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'text_embedding') + 1.0",
                        "params": {"query_vector": query_emb}
                    }
                }
            },
            "size": 1
        }

        # Perform search
        try:
            response = es.search(index=index, body=search_query)
            if response['hits']['hits']:
                # Do not display relevant text on frontend
                get_text = response['hits']['hits'][0]['_source']['text']

                # Query Mistral AI model to get the answer
                def query_mistral(payload):
                    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
                    API_TOKEN = st.secrets["HUGGINGFACE_API_KEY"]  # Stored in Streamlit secrets
                    headers = {"Authorization": f"Bearer {API_TOKEN}"}
                    response = requests.post(API_URL, headers=headers, json=payload)
                    
                    try:
                        if response.status_code != 200:
                            return "Error: Unable to get response from the AI model."
                        data = response.json()
                        generated_text = data[0]['generated_text']
                        prompt_index = generated_text.find('[/INST]')
                        if prompt_index != -1:
                            generated_text = generated_text[prompt_index + len('[/INST]'):]
                        return generated_text
                    except requests.exceptions.RequestException as e:
                        return f"Request exception: {e}"
                    except ValueError as e:
                        return f"Value error: {e}"

                negResponse = "I'm unable to answer the question based on the information I have."
                prompt = f"[INST] You are a helpful Q&A assistant. Your task is to answer this question: {query}. Use only the information from this text: {get_text}. Provide the answer in normal text format. If the answer is not contained in the text, reply with {negResponse}. [/INST]"
                max_new_tokens = 2000

                # Get the answer
                data = query_mistral({"parameters": {"max_new_tokens": max_new_tokens}, "inputs": prompt})
                st.subheader("Answer:")
                st.write(data)
            else:
                st.error("No relevant text found for the query.")
        except Exception as e:
            st.error(f"Error during search or Mistral AI query: {e}")

else:
    st.info("Please upload PDF files to start.")
