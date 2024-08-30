import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import requests

# Streamlit app title
st.title("PDF Summarization and Q&A")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Read the PDF
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Display the extracted text
    st.subheader("Extracted Text:")
    st.write(text)

    # Load sentence transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    emb = model.encode(text)

    # Elasticsearch connection (using API key)
    es = Elasticsearch(
        cloud_id="ac668387facb455d9201540f7bcdccf3:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDM5OGQ1NGMzMzZlZTQ0MGM5MGVjM2VjYmIwYjc0MWRjJGQ4NDgxNTA2MWM1NDQwYjA4YmE3NTAxMGQ1YzM3MGJl",
        api_key="SxuCkvPJTsyqbpgYGWiitA"  # Replace with your actual API key
    )

    # Test connection
    try:
        if es.ping():
            st.success("Successfully connected to Elasticsearch")
        else:
            st.error("Failed to connect to Elasticsearch")
    except Exception as e:
        st.error(f"Error connecting to Elasticsearch: {e}")

    # Index the document in Elasticsearch
    index = "summarization_pdf"
    doc = {"text_embedding": emb, "text": text}

    try:
        es.indices.delete(index=index, ignore=[400, 404])
        es.indices.create(
            index=index,
            mappings={
                "properties": {
                    "text": {"type": "text"},
                    "text_embedding": {"type": "dense_vector", "dims": 384},
                }
            }
        )
        es.index(index=index, document=doc)
        st.success("Document indexed successfully!")
    except Exception as e:
        st.error(f"Error indexing document: {e}")

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
            for hit in response['hits']['hits']:
                get_text = hit['_source']['text']
                st.subheader("Relevant Text:")
                st.write(get_text)

            # Query Mistral AI model to get the answer
            def query_mistral(payload):
                API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
                API_TOKEN = "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"
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
            prompt = f"[INST] You are a helpful Q&A assistant. Your task is to answer this question: {query}. Use only the information from this text: {get_text}. Provide the answer strictly in HTML format. If the answer is not contained in the text, reply with {negResponse}. [/INST]"
            max_new_tokens = 2000

            # Get the answer
            data = query_mistral({"parameters": {"max_new_tokens": max_new_tokens}, "inputs": prompt})
            st.subheader("Answer:")
            st.write(data)
        except Exception as e:
            st.error(f"Error during search or Mistral AI query: {e}")

else:
    st.info("Please upload a PDF file to start.")
