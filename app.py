import streamlit as st
from src.cleaner import TranscriptCleaner
from src.linker import SemanticLinker
import whisper

@st.cache_resource
def load_models():
    stt_model = whisper.load_model("base")
    cleaner = TranscriptCleaner(model_path="./models/t5_cleaner")
    linker = SemanticLinker(model_path="./models/bert_linker")
    return stt_model, cleaner, linker

stt, cleaner, linker = load_models()