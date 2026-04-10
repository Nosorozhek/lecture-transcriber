import os
import base64
import tempfile
import streamlit as st
import runpod
import requests

from src.schema import (
    StatusEvent, MaterialsReadyEvent, RawSpeechEvent, 
    ProcessedChunkEvent, ModelRegistry
)

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")
VOLUME_MOUNT_PATH = "/runpod-volume" 

if not RUNPOD_API_KEY or not ENDPOINT_ID:
    st.error("Internal Error: RUNPOD_API_KEY or ENDPOINT_ID not set")
    st.stop()

st.set_page_config(page_title="Lecture Transcriber", layout="wide", page_icon="🎓")

def dict_to_event(data: dict):
    etype = data.get("event_type")
    if etype == "StatusEvent":
        return StatusEvent(message=data["message"], stage=data["stage"], progress=data.get("progress"))
    if etype == "MaterialsReadyEvent":
        mats = [ParsedMaterial(**m) for m in data["materials"]]
        return MaterialsReadyEvent(materials=mats)
    if etype == "RawSpeechEvent":
        return RawSpeechEvent(chunk_index=data["chunk_index"], text=data["text"])
    if etype == "ProcessedChunkEvent":
        return ProcessedChunkEvent(
            chunk_index=data["chunk_index"],
            cleaned_text=data["cleaned_text"],
            matched_material_id=data["matched_material_id"],
            similarity_score=data["similarity_score"]
        )
    return None

if "materials_db" not in st.session_state:
    st.session_state.materials_db = {}
if "ui_placeholders" not in st.session_state:
    st.session_state.ui_placeholders = {}


with st.sidebar:
    st.markdown("---")
    st.header("📂 Upload Files")
    audio_file = st.file_uploader("Recording (MP3/WAV)", type=["mp3"])
    material_files = st.file_uploader("Materials (PDF/Code)", accept_multiple_files=True)
    
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

    runpod.api_key = RUNPOD_API_KEY
    endpoint = runpod.Endpoint(ENDPOINT_ID)

    import uuid
    job_id = str(uuid.uuid4())[:8]
    work_dir = os.path.join(VOLUME_MOUNT_PATH, "jobs", job_id)
    os.makedirs(work_dir, exist_ok=True)
    
    audio_path = os.path.join(work_dir, audio_file.name)
    with open(audio_path, "wb") as f:
        f.write(audio_file.read())
        
    mat_paths = []
    for mf in material_files:
        p = os.path.join(work_dir, mf.name)
        with open(p, "wb") as f:
            f.write(mf.read())
        mat_paths.append(p)

    st.session_state.ui_placeholders.clear()
    st.session_state.materials_db.clear()

    payload = {
        "audio_path": audio_path,
        "material_paths": mat_paths,
        "chunk_threshold_chars": 600
    }

    try:
        run_req = endpoint.run(payload)
        for response in run_req.stream():
            if response.get("event_type") == "Error" or "error" in response:
                st.error(f"Backend Error: {response.get('message') or response.get('error')}")
                break
            event = dict_to_event(response)
            if not event: continue
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

    except requests.exceptions.HTTPError as err:
        st.error(f"HTTP 400 Bad Request. \n\nDetails: {err.response.text}")
    except Exception as e:
        st.error(f"Endpoint Error: {e}")  