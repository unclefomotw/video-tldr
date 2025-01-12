import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from video_tldr._internal.log import setup_file_logger

logger = setup_file_logger(log_level=logging.INFO)

DEFAULT_MODEL = "openai/whisper-large-v3"


# ref: https://huggingface.co/openai/whisper-large-v3
class WhisperWrapper:
    def __init__(self, model_id: str = DEFAULT_MODEL,
                 device: Optional[str] = None,
                 output_dir: Path | str = "out"):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon
            else:
                device = "cpu"
            self.device = device

        torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, use_safetensors=True
        )
        self.model.to(self.device)

        logger.info(f"Using device: {device}")
        logger.info(f"Loading model {model_id} with output directory {output_dir}")

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def transcribe(self, audio_path, lang: Optional[str] = None):
        generate_kwargs = {}
        if lang is not None:
            generate_kwargs["language"] = lang

        logger.info(f"Transcribing {audio_path} to text...")

        result = self.pipe(audio_path, return_timestamps=True, generate_kwargs=generate_kwargs)
        output_file = self.output_dir / (Path(audio_path).stem + ".txt")

        whole_text = ""
        for chunk in result["chunks"]: # type: ignore
            if chunk['text']: # type: ignore
                whole_text += str(chunk['text']) # type: ignore
                whole_text += "\n"

        with open(output_file, "w") as f:
            f.write(whole_text)
