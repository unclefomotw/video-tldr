from typing import Optional

import requests


class OllamaServer:
    def __init__(self, host: str = "localhost",
                 port: int = 11434,
                 default_model: Optional[str] = None,
                 default_num_ctx: Optional[int] = None
                 ):
        self.host = host
        self.port = port
        self.default_model = default_model
        self.default_num_ctx = default_num_ctx

    def chat(self, messages: list[dict],
             model: Optional[str] = None,
             num_ctx: Optional[int] = None) -> str:
        if not model:
            if not self.default_model:
                raise ValueError("No default model specified and no model provided")
            model = self.default_model

        if not num_ctx:
            num_ctx = self.default_num_ctx

        url = f"http://{self.host}:{self.port}/api/chat"
        data = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if num_ctx is not None:
            data["options"] = {"num_ctx": num_ctx}

        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            raise Exception(f"Failed to chat with Ollama server: {response.text}")
