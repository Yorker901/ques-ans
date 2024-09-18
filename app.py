import os
import tempfile
import streamlit as st
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import requests
from PyPDF2 import PdfReader
import re
from youtube_transcript_api import YouTubeTranscriptApi

# Set environment variable to avoid TOKENIZERS_PARALLELISM warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Elasticsearch connection (using API key stored in Streamlit secrets)
es = Elasticsearch(
    cloud_id=st.secrets["ELASTIC_CLOUD_ID"],  
    api_key=st.secrets["ELASTIC_API_KEY"]
)

# Function to interact with Mistral AI
def query_mistral(payload):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    API_TOKEN = st.secrets["HUGGINGFACE_API_KEY"]
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

# Indices for different data types
pdf_index = "summarization_pdf"
video_index = "video_transcripts"
youtube_index = "youtube_transcripts"

# Function to create Elasticsearch index with required mappings
def create_index(index_name, dimensions=384):
    try:
        es.indices.create(
            index=index_name,
            ignore=400,
            mappings={
                "properties": {
                    "text": {"type": "text"},
                    "text_embedding": {"type": "dense_vector", "dims": dimensions},
                }
            }
        )
    except Exception as e:
        st.error(f"Error creating Elasticsearch index {index_name}: {e}")

# Create indices
create_index(pdf_index)
create_index(video_index)
create_index(youtube_index)

# Function to extract text from video files
def extract_text_from_video(video_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
            temp_audio_path = temp_audio_file.name

        clip = VideoFileClip(video_file)
        audio = clip.audio
        audio.write_audiofile(temp_audio_path, codec='pcm_s16le')

        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_audio_path) as source:
            audio_content = recognizer.record(source)
        
        text = recognizer.recognize_google(audio_content)
        embeddings = model.encode(text)

        os.remove(temp_audio_path)

        doc = {"text": text, "text_embedding": embeddings.tolist()}
        es.index(index=video_index, document=doc)
        es.indices.refresh(index=video_index)

        return text
    except Exception as e:
        return f"Error extracting text from video: {str(e)}"

# Function to extract text from PDFs
def extract_text_from_pdfs(uploaded_files):
    all_text = ""
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            all_text += page.extract_text()
    return all_text

def get_youtube_transcript(video_url):
    match = re.search(r'v=([a-zA-Z0-9_-]+)', video_url)
    if match:
        video_id = match.group(1)
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = ' '.join([entry['text'] for entry in transcript])
            return transcript_text
        except Exception as e:
            return f"Error retrieving transcript: {str(e)}"
    else:
        return "Invalid YouTube URL format. Please provide a valid URL."


def delete_existing_entries(index_name):
    try:
        es.delete_by_query(index=index_name, body={"query": {"match_all": {}}})
        es.indices.refresh(index=index_name)
    except Exception as e:
        st.error(f"Error deleting previous entries from index {index_name}: {e}")

# Sidebar for app logo and name
st.sidebar.image("6c6337da-c7a2-4c83-b7ab-7ba39fad7d74_0.png", use_column_width=True)  # Add path to your logo image
st.sidebar.title("QuerySage")
# st.sidebar.markdown("Extract, Analyze, and Query from Multi-Sources.")

# Streamlit app code
st.title("Multi-Source Text Extraction")

source_option = st.selectbox("Choose your source", ["Upload PDFs", "Video File", "YouTube Video"])

if source_option == "Upload PDFs":
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        delete_existing_entries(pdf_index)  # Delete previous entries
        all_text = extract_text_from_pdfs(uploaded_files)
        emb = model.encode(all_text)

        doc = {"text": all_text, "text_embedding": emb.tolist()}
        es.index(index=pdf_index, document=doc)
        es.indices.refresh(index=pdf_index)

elif source_option == "Video File":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file:
        delete_existing_entries(video_index)  # Delete previous entries
        video_file_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        with open(video_file_path, "wb") as f:
            f.write(uploaded_file.read())

        text = extract_text_from_video(video_file_path)
        st.write("Video transcript extracted, previous entries deleted, and new text indexed.")

elif source_option == "YouTube Video":
    youtube_url = st.text_input("Enter YouTube Video URL:")
    if youtube_url:
        delete_existing_entries(youtube_index)  # Delete previous entries
        transcript_text = get_youtube_transcript(youtube_url)
        if transcript_text:
            emb = model.encode(transcript_text)

            doc = {"text": transcript_text, "text_embedding": emb.tolist()}
            es.index(index=youtube_index, document=doc)
            es.indices.refresh(index=youtube_index)

            st.write("YouTube transcript extracted, previous entries deleted, and new text indexed.")
        else:
            st.error("Failed to retrieve YouTube transcript.")

# User query input for searching indexed data
query = st.text_input("Enter your question based on the content:")

if query:
    query_emb = model.encode(query)

    # Automatically select index based on the user's earlier choice of source
    if source_option == "Upload PDFs":
        index = pdf_index
    elif source_option == "Video File":
        index = video_index
    elif source_option == "YouTube Video":
        index = youtube_index

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
        response = es.search(index=index, body=search_query)
        if response['hits']['hits']:
            get_text = response['hits']['hits'][0]['_source']['text']
            
            negResponse = "I'm unable to answer the question based on the information I have."
            prompt = f"[INST] You are a helpful Q&A assistant. Your task is to answer this question: {query}. Use only the information from this text: {get_text}. Provide the answer in normal text format. If the answer is not contained in the text, reply with {negResponse}. [/INST]"
            max_new_tokens = 2000

            data = query_mistral({"parameters": {"max_new_tokens": max_new_tokens}, "inputs": prompt})
            st.subheader("Answer:")
            st.write(data)
        else:
            st.error("No relevant text found for the query.")
    except Exception as e:
        st.error(f"Error during search or Mistral AI query: {e}")
