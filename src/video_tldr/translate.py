# For now it only supports Ollama serve as inference

from typing import Optional

from video_tldr.ollama import OllamaServer

ollama_server = OllamaServer(default_model="llama3.1", default_num_ctx=32768)


def naive_translate(text: str, lang: str, model_name: Optional[str] = None) -> str:
    system_message = "You are an expert linguist specializing in translation."

    user_prompt = f"Translate the following text to {lang}:\n\n{text}"

    return ollama_server.chat(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ],
        model=model_name
    )
