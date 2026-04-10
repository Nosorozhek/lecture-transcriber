import os
import io
import runpod
from typing import Iterator, Any

from src.schema import ModelRegistry
from src.inference_pipeline import load_models, run_lecture_pipeline

os.environ["HF_HOME"] = "/runpod-volume/huggingface_cache"

USERNAME="Nosorozhek"

MODELS = None

def get_models():
    global MODELS
    if MODELS is not None:
        return MODELS
    
    if not os.path.exists("/runpod-volume"):
        raise Exception("CRITICAL: /runpod-volume not found!")

    from src.inference_pipeline import load_models
    MODELS = load_models(
        whisper_model_size="medium",
        t5_model_path="Nosorozhek/rut5-cleaner-tuned",
        e5_linker_path="Nosorozhek/e5-linker-tuned",
        vlm_model_name="Qwen/Qwen2-VL-7B-Instruct",
        hf_token=os.getenv("HF_TOKEN")
    )
    return MODELS

def handler(job) -> Iterator[dict]:
    """
    request scheme is following
    {
        "audio_path": "/workspace/my_audio.mp3",
        "material_paths": ["/workspace/slides.pdf", "/workspace/code.cpp"]
    }
    """
    job_input = job["input"]
    audio_path = job_input.get("audio_path")
    material_paths = job_input.get("material_paths", [])
    chunk_threshold = job_input.get("chunk_threshold_chars", 600)

    if not audio_path or not os.path.exists(audio_path):
        yield {"error": f"Audio path {audio_path} not found"}
        return

    try:
        MODELS = get_models()
        for event in run_lecture_pipeline(
            audio_path=audio_path,
            material_paths=material_paths,
            models=MODELS,
            chunk_threshold_chars=chunk_threshold
        ):
            if hasattr(event, "__dict__"):
                event_dict = event.__dict__.copy()
                event_dict["event_type"] = event.__class__.__name__
                if event_dict["event_type"] == "MaterialsReadyEvent":
                    event_dict["materials"] = [m.__dict__ for m in event.materials]
                yield event_dict
            else:
                yield {"event_type": "Unknown", "data": str(event)}
                
    except Exception as e:
        yield {"event_type": "Error", "message": str(e)}

runpod.serverless.start({
    "handler": handler,
    "return_generator": True
})
