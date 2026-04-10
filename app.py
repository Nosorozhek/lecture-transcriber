import os
import base64
import tempfile
import streamlit as st

from src.schema import (
    StatusEvent, MaterialsReadyEvent, RawSpeechEvent, 
    ProcessedChunkEvent, ModelRegistry
)
from src.inference_pipeline import load_models, run_lecture_pipeline

st.set_page_config(page_title="Lecture Transcriber", layout="wide", page_icon="🎓")

if "materials_db" not in st.session_state:
    st.session_state.materials_db = {}
if "ui_placeholders" not in st.session_state:
    st.session_state.ui_placeholders = {}

@st.cache_resource(show_spinner="Loading Models... This might take a few minutes...")
def get_models() -> ModelRegistry:
    return load_models()

models = get_models()

with st.sidebar:
    st.markdown("---")
    st.header("📂 Upload Files")
    audio_file = st.file_uploader("Recording (MP3/WAV)", type=["mp3", "wav", "m4a"])
    material_files = st.file_uploader("Materials (PDF/Code)", type=["pdf", "rs", "cu", "cuh", "py", "cpp", "c", "java"], accept_multiple_files=True)
    
    start_btn = st.button("🚀 Start Processing", use_container_width=True, type="primary")

st.title("🎓 Lecture Transcriber")

status_container = st.empty()
progress_bar = st.empty()

header_col1, header_col2 = st.columns(2)
with header_col1:
    st.subheader("🗣️ Transcribed Speech")
with header_col2:
    st.subheader("📚 Related Artifact")
st.markdown("---")

feed_container = st.container()

def detect_language(source_file: str) -> str:
    if not source_file:
        return "text"

    ext = source_file.split(".")[-1].lower()

    mapping = {
        "py": "python",
        "rs": "rust",
        "cpp": "cpp",
        "hpp": "cpp",
        "c": "c",
        "h": "c",
        "cu": "cuda",
        "cuh": "cuda",
        "java": "java",
        "md": "markdown",
        "txt": "text",
        "pdf": "text",
    }

    return mapping.get(ext, "text")

def render_artifact(material_id: str, container):
    material = st.session_state.materials_db.get(material_id)
    if not material:
        container.warning("Artifact not found.")
        return
    
    if material.type == "code":
        container.markdown(f"**File:** `{material.source_file}`")
        lang = detect_language(material.source_file)
        container.code(material.content, language=lang)
    else:
        if material.image_base64:
            image_bytes = base64.b64decode(material.image_base64)
            container.image(
                image_bytes,
                caption=f"{material.source_file} ({material.id})",
                width="stretch"
            )

        expander = container.expander("Show AI Description")
        expander.markdown(f"> {material.content}")

if start_btn:
    if not audio_file:
        st.error("Please provide a lecture recording.")
        st.stop()

    temp_dir = tempfile.mkdtemp()
    
    audio_path = os.path.join(temp_dir, audio_file.name)
    with open(audio_path, "wb") as f:
        f.write(audio_file.read())
        
    mat_paths = []
    for mf in material_files:
        p = os.path.join(temp_dir, mf.name)
        with open(p, "wb") as f:
            f.write(mf.read())
        mat_paths.append(p)

    st.session_state.ui_placeholders.clear()
    st.session_state.materials_db.clear()

    for event in run_lecture_pipeline(audio_path, mat_paths, models):
        if isinstance(event, StatusEvent):
            if event.stage == "complete":
                status_container.success(f"✅ {event.message}")
                progress_bar.empty()
            else:
                status_container.info(f"⏳ **Status:** {event.message}")
                if event.progress is not None:
                    progress_bar.progress(event.progress)
                else:
                    progress_bar.empty()

        elif isinstance(event, MaterialsReadyEvent):
            for m in event.materials:
                st.session_state.materials_db[m.id] = m

        elif isinstance(event, RawSpeechEvent):
            with feed_container:
                chunk_row = st.container()
                col_left, col_right = chunk_row.columns(2)
                text_slot = col_left.empty()
                art_slot = col_right.empty()
                
                st.session_state.ui_placeholders[event.chunk_index] = {
                    "text": text_slot,
                    "art": art_slot
                }

                text_slot.info(f"*Shortening...*\n\n{event.text}")
                art_slot.caption("Searching for artifacts...")

        elif isinstance(event, ProcessedChunkEvent):
            slots = st.session_state.ui_placeholders.get(event.chunk_index)
            if slots:
                slots["text"].success(event.cleaned_text)
                art_block = slots["art"].container()
                art_block.markdown(f"_Confidence Score: {event.similarity_score:.2f}_")
                render_artifact(event.matched_material_id, art_block)
                