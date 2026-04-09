from dataclasses import dataclass
from typing import List, Any, Union, Literal, Optional

@dataclass
class ParsedMaterial:
    id: str
    content: str
    type: Literal["slide", "code"]
    source_file: str

@dataclass
class ModelRegistry:
    whisper_model: Any
    t5_tokenizer: Any
    t5_model: Any
    e5_linker: Any
    vlm_model: Any
    vlm_processor: Any
    device: str

@dataclass
class StatusEvent:
    message: str
    stage: Literal["initializing", "parsing", "transcribing", "linking", "complete"]
    progress: Optional[float] = None

@dataclass
class MaterialsReadyEvent:
    materials: List[ParsedMaterial]

@dataclass
class RawSpeechEvent:
    chunk_index: int
    text: str

@dataclass
class ProcessedChunkEvent:
    chunk_index: int
    cleaned_text: str
    matched_material_id: str
    similarity_score: float

PipelineEvent = Union[StatusEvent, MaterialsReadyEvent, RawSpeechEvent, ProcessedChunkEvent]
