import os
import io
import base64
import torch
from tqdm import tqdm
from PIL import Image
from functools import lru_cache
from typing import List, Any, Iterator

from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from src.schema import *

@lru_cache(maxsize=1)
def load_models(
    whisper_model_size: str = "large-v3",
    t5_model_path: str = "Nosorozhek/rut5-cleaner-tuned",
    e5_linker_path: str = "Nosorozhek/e5-linker-tuned",
    vlm_model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
    hf_token: str | None = None
) -> ModelRegistry:
    print("Loading models into memory...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    whisper_model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)

    t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_path, token=hf_token)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_path, token=hf_token).to(device)
    t5_model.eval()

    e5_linker = SentenceTransformer(e5_linker_path, token=hf_token, device=device)

    vlm_model: Qwen2VLForConditionalGeneration = Qwen2VLForConditionalGeneration.from_pretrained(
        vlm_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0", 
        token=hf_token
    )
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_name, token=hf_token)
    vlm_model.eval()
    
    return ModelRegistry(whisper_model, t5_tokenizer, t5_model, e5_linker, vlm_model, vlm_processor, device)

def get_slide_description_prompt():
    return f"""### РОЛЬ
Ты -- лектор в университете на кафедре Computer Science и Machine Learning. 
Твоя задача: посмотреть на слайд и произнести его вслух так, как это сделал бы живой человек на лекции.

### ИНСТРУКЦИИ
1. Сделай описание материала со слайда, опираясь НА КАРТИНКУ (графики, схемы).
2. Если на слайде много информации, в речи оставь произвольную часть, но НЕ МЕНЕЕ половины информации слайда должно быть покрыто.
3. Описывай код максимально подробно, каждую часть алгоритма.
4. Ответ должен состоять из 6 ПРЕДЛОЖЕНИЙ.

### ОГРАНИЧЕНИЯ (Negative Constraints)
- НЕ добавляй вводных фраз ("На слайде написано...", "На слайде обссуждается...").
- НЕ добавляй примечаний от редактора.
- НЕ меняй смысл высказывания.
- НЕ приводи отвлеченных примеров.
- НЕ заключай ответ в кавычки.
"""

def parse_pdf_to_slides(pdf_path: str, vlm_model: Any, vlm_processor: Any) -> Iterator[Union[StatusEvent, ParsedMaterial]]:    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    filename = os.path.basename(pdf_path)
    
    for i, page in enumerate(tqdm(doc)):
        yield StatusEvent(
            message=f"Parsing {filename} (Slide {i+1} of {total_pages})...", 
            stage="parsing", 
            progress=float(i / total_pages)
        )

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("png")
        img_b64 = base64.b64encode(img_data).decode("utf-8")
        image = Image.open(io.BytesIO(img_data))
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text", 
                        "text": get_slide_description_prompt(),
                    },
                ],
            }
        ]

        text = vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = vlm_processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(vlm_model.device)

        with torch.no_grad():
            generated_ids = vlm_model.generate(**inputs, max_new_tokens=500)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            vlm_description = vlm_processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]

        yield ParsedMaterial(
            id=f"{filename}_Slide_{i+1}", 
            content=vlm_description,
            type="slide",
            source_file=filename,
            image_base64=img_b64,
        )
    doc.close()

def parse_code_to_blocks(file_path: str) -> List[ParsedMaterial]:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    ext = file_path.split('.')[-1].lower()
    
    lang_map = {
        'py': Language.PYTHON, 'cpp': Language.CPP, 'c': Language.C,
        'h': Language.CPP, 'hpp': Language.CPP, 'cu': Language.CPP,
        'cuh': Language.CPP, 'java': Language.JAVA, 'go': Language.GO,
        'rs': Language.RUST,
    }
    
    lang = lang_map.get(ext)
    
    if lang:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=1500, chunk_overlap=150
        )
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        
    blocks = splitter.split_text(content)
        
    formatted_blocks = []
    for i, b in enumerate(blocks):
        if len(b.strip()) > 20:
            formatted_blocks.append(ParsedMaterial(
                id=f"{os.path.basename(file_path)}_block_{i+1}",
                content=b.strip(),
                type="code",
                source_file=os.path.basename(file_path)
            ))
            
    return formatted_blocks

def parse_materials(file_paths: List[str], vlm_model: Any, vlm_processor: Any) -> Iterator[Union[StatusEvent, ParsedMaterial]]:
    for path in file_paths:
        filename = os.path.basename(path)
        if path.lower().endswith(".pdf"):
            yield from parse_pdf_to_slides(path, vlm_model, vlm_processor)
            yield StatusEvent(message=f"Finished parsing {filename}", stage="parsing", progress=1.0)
        else:
            yield StatusEvent(message=f"Parsing code file: {filename}...", stage="parsing", progress=0.5)
            blocks = parse_code_to_blocks(path)
            for block in blocks:
                yield block
            yield StatusEvent(message=f"Finished parsing {filename}", stage="parsing", progress=1.0)

class SequentialLinker:
    def __init__(
        self,
        model, 
        materials: List[ParsedMaterial], 
        lookahead_window: int = 3,       
        confidence_threshold: float = 0.3,
        step_penalty: float = 0.02,       
        switch_artifact_penalty: float = 0.1,
    ):
        self.model = model
        
        self.lookahead_window = lookahead_window
        self.confidence_threshold = confidence_threshold
        self.step_penalty = step_penalty
        self.switch_artifact_penalty = switch_artifact_penalty
        
        self.artifacts: dict[str, list[ParsedMaterial]] = {}
        for m in materials:
            art_name = self._get_artifact_name(m.id)
            if art_name not in self.artifacts:
                self.artifacts[art_name] = []
            self.artifacts[art_name].append(m)
            
        self.state = {}
        self.embeddings = {}
        self.last_active_artifact = None 
        
        for art_name, mats in self.artifacts.items():
            self.state[art_name] = 0
            
            formatted_passages = ["passage: " + m.content for m in mats]
            self.embeddings[art_name] = self.model.encode(formatted_passages, convert_to_tensor=True)

    def _get_artifact_name(self, doc_id: str) -> str:
        if "_Slide_" in doc_id:
            return doc_id.split("_Slide_")[0]
        elif "_block_" in doc_id:
            return doc_id.split("_block_")[0]
        return "unknown_artifact"
        
    def predict(self, spoken_text: str):
        if not self.embeddings:
            return {"matched_id": "None", "matched_content": "None", "score": 0.0}

        query_emb = self.model.encode("query: " + spoken_text, convert_to_tensor=True)
        
        global_best_score = -999.0
        winning_artifact = None
        winning_local_idx = 0
        
        for art_name, embs in self.embeddings.items():
            curr_idx = self.state[art_name]
            
            if curr_idx >= len(embs):
                curr_idx = len(embs) - 1
                
            end_idx = min(curr_idx + self.lookahead_window + 1, len(embs))
            window_embs = embs[curr_idx:end_idx]
            
            scores = util.cos_sim(query_emb, window_embs)[0] 
            for i in range(len(scores)):
                scores[i] -= (i * self.step_penalty)
                
            if self.last_active_artifact and self.last_active_artifact != art_name:
                scores -= self.switch_artifact_penalty
                
            local_best_offset = int(torch.argmax(scores).item())
            local_best_score = scores[local_best_offset].item()
            
            if local_best_score > global_best_score:
                global_best_score = local_best_score
                winning_artifact = art_name
                winning_local_idx = curr_idx + local_best_offset
                
        if winning_artifact and global_best_score > self.confidence_threshold:
            self.state[winning_artifact] = winning_local_idx
            self.last_active_artifact = winning_artifact
            
        if not winning_artifact:
            return {"matched_id": "None", "matched_content": "None", "score": 0.0}
            
        matched_material = self.artifacts[winning_artifact][winning_local_idx]
        
        raw_score = util.cos_sim(query_emb, self.embeddings[winning_artifact][winning_local_idx])[0][0].item()
        
        return {
            "matched_id": matched_material.id,
            "matched_content": matched_material.content,
            "score": round(raw_score, 4),
            "penalty_score": round(global_best_score, 4),
            "source_file": winning_artifact
        }


def run_lecture_pipeline(
    audio_path: str,
    material_paths: List[str],
    models: ModelRegistry,
    chunk_threshold_chars: int = 600 
) -> Iterator[PipelineEvent]:
    
    yield StatusEvent("Initializing materials parsers...", "parsing", 0.0)

    materials = []
    for item in parse_materials(material_paths, models.vlm_model, models.vlm_processor):
        if isinstance(item, StatusEvent):
            yield item
        elif isinstance(item, ParsedMaterial):
            materials.append(item)

    yield MaterialsReadyEvent(materials)
    
    yield StatusEvent("Indexing materials for search...", "linking")
    linker = SequentialLinker(models.e5_linker, materials)

    yield StatusEvent("Starting transcription...", "transcribing")
    segments, _ = models.whisper_model.transcribe(audio_path, beam_size=5, language="ru")
    
    chunk_index = 0
    current_buffer = []
    current_char_count = 0

    for seg in segments:
        current_buffer.append(seg.text)
        current_char_count += len(seg.text)
        
        if current_char_count >= chunk_threshold_chars:
            raw_text = " ".join(current_buffer).strip()
            
            yield RawSpeechEvent(chunk_index, raw_text)
            
            input_text = "clean: " + raw_text
            inputs = models.t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(models.device)
            with torch.no_grad():
                out = models.t5_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=512,
                    num_beams=5,  
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=True
                )
            cleaned_text = models.t5_tokenizer.decode(out[0], skip_special_tokens=True)

            match_info = linker.predict(cleaned_text)
            
            yield ProcessedChunkEvent(
                chunk_index=chunk_index,
                cleaned_text=cleaned_text,
                matched_material_id=match_info["matched_id"],
                similarity_score=match_info["score"]
            )
            
            current_buffer = []
            current_char_count = 0
            chunk_index += 1

    yield StatusEvent("Lecture processing complete!", "complete")
