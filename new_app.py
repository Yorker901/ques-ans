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

    # Test connection
    try:
        if not es.ping():
            st.error("Failed to connect to Elasticsearch")
    except Exception as e:
        st.error(f"Error connecting to Elasticsearch: {e}")

    # Index the document in Elasticsearch
    index = "summarization_pdf"
    
    try:
        # Update for deprecation warning
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
        st.error(f"Error indexing document: {e}")

    doc = {"text_embedding": emb, "text": text}
    es.index(index=index, document=doc)
    es.indices.refresh(index=index)

    # Chat-like interface
    st.subheader("Chat with the PDF")

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

    # User query input
    user_query = st.text_input("You:", key="query")

    if st.button("Send"):
        if user_query:
            # Append user's query to chat history
            st.session_state.chat_history.append({"sender": "user", "message": user_query})

            # Get the query embedding
            query_emb = model.encode(user_query)

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

            try:
                # Perform search
                response = es.search(index=index, body=search_query)
                if response['hits']['hits']:
                    get_text = response['hits']['hits'][0]['_source']['text']

                    # Create a prompt for the Mistral AI model
                    prompt = f"[INST] You are a helpful Q&A assistant. Your task is to answer this question: {user_query}. Use only the information from this text: {get_text}. Provide the answer in normal text format. If the answer is not contained in the text, reply with {negResponse}. [/INST]"
                    max_new_tokens = 2000

                    # Get the answer from Mistral AI
                    data = query_mistral({"parameters": {"max_new_tokens": max_new_tokens}, "inputs": prompt})

                    # Append AI's response to chat history
                    st.session_state.chat_history.append({"sender": "bot", "message": data})
                else:
                    st.session_state.chat_history.append({"sender": "bot", "message": "No relevant text found for your query."})

            except Exception as e:
                st.session_state.chat_history.append({"sender": "bot", "message": f"Error: {e}"})

    # Display chat history
    for chat in st.session_state.chat_history:
        if chat["sender"] == "user":
            st.markdown(f"**You:** {chat['message']}")
        else:
            st.markdown(f"**Bot:** {chat['message']}")

else:
    st.info("Please upload a PDF file to start.")
