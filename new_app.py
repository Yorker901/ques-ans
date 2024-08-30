import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import requests

# Streamlit app title
st.title("PDF Summarization and Q&A - Chat Interface")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Read the PDF
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Load sentence transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    emb = model.encode(text)

    # Elasticsearch connection (using API key)
    es = Elasticsearch(
        cloud_id="ac668387facb455d9201540f7bcdccf3:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDM5OGQ1NGMzMzZlZTQ0YzA5MGVjM2VjYmIwYjc0MWRjJGQ4NDgxNTA2MWM1NDQwYjA4YmE3NTAxMGQ1YzM3MGJl",
        api_key="bDMyb29aRUJmbEtlQzJiSDlEc0M6U3h1Q2t2UEpUc3lxYnBnWUdXaWl0QQ=="  # Replace with your actual API key
    )

    # Check if the index exists
    index = "summarization_pdf"
    if not es.indices.exists(index=index):
        # Create the index with the required mappings
        try:
            es.indices.create(
                index=index,
                mappings={
                    "properties": {
                        "text": {"type": "text"},
                        "text_embedding": {"type": "dense_vector", "dims": 384},
                    }
                }
            )
            st.success(f"Index '{index}' created successfully!")
        except Exception as e:
            st.error(f"Error creating index: {e}")
    else:
        st.info(f"Index '{index}' already exists.")

    # Try to index the document
    try:
        doc = {"text_embedding": emb, "text": text}
        response = es.index(index=index, document=doc)
        st.success(f"Document indexed successfully: {response['_id']}")
    except Exception as e:
        st.error(f"Error indexing document: {e}")

    # Rest of the code for chat interface...

# Continue the Streamlit app

# Input box for user queries
user_input = st.text_input("Ask a question about the PDF:")

if user_input:
    # Add the user's question to the chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Generate a response using Elasticsearch
    try:
        # Query Elasticsearch with user input
        query = {
            "size": 1,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'text_embedding') + 1.0",
                        "params": {"query_vector": model.encode(user_input).tolist()},
                    },
                }
            }
        }

        response = es.search(index=index, body=query)
        answer = response['hits']['hits'][0]['_source']['text']

        # Add the AI's response to the chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"Error querying Elasticsearch: {e}")
        st.session_state.chat_history.append({"role": "assistant", "content": "I couldn't retrieve the answer."})

# Display the chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Assistant:** {message['content']}")

# Optionally, clear the chat history
if st.button("Clear Chat"):
    st.session_state.chat_history = []
