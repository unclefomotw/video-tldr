# Video Summarizer

This repository contains Python scripts to interact with an Ollama server for chat functionality and to summarize text using a language model. It also includes a wrapper for the Whisper ASR (Automatic Speech Recognition) model.

## Features

- **Whisper ASR Wrapper**: Transcribe audio files into text using the Whisper model.
- **Ollama Server Interaction**: Interact with an Ollama server to send chat messages and receive responses.
- **Text Summarization**: Summarize input text using a language model in chunks up to a specified size.
- **Translation**: Translate text to help you further understand the summarized content.

## Installation

### Ollama

Ollama https://ollama.com/ is required.  Run `ollama serve` to start the server.

`ollama pull <model>` in advance.  By default "phi4" and "llama3.1" are used,
but can be overridden in `WhisperWrapper` class and `summarize_text` function.

### This python library
```bash
python -m venv venv
source venv/bin/activate

pip install git+https://github.com/unclefomotw/video-tldr.git
```

## Usage

### Whisper ASR

```python
from video_tldr.whisper_wrapper import WhisperWrapper

# Initialize the Whisper wrapper
whisper = WhisperWrapper(output_dir="path/to/output/directory")

# Transcribe an audio file
audio_path = "path/to/your/audiofile.mp3"
transcription = whisper.transcribe(audio_path)

# Transcription is also under path/to/output/directory/audiofile.txt
print(transcription)
```

### Text Summarization

It is suggested that the text is made of lines; each represents a sentence in a reasonable length.

```python
from video_tldr.summarize import summarize_text

# Summarize a text
text = """Your long text here."""
summary = summarize_text(text, model_name="your-model")
print(summary)
```

### Translation

```python
from video_tldr.translate import naive_translate

# Example text to be translated
text_to_translate = "Hello, how are you today? This is a test translation."

# Language to which the text should be translated
target_language = "es"  # Spanish in this case

# Call the naive_translate function with the example text and target language
translated_text = naive_translate(text_to_translate, lang=target_language)

# Print the translated text
print("Translated Text:", translated_text)
```


## License

[MIT](https://choosealicense.com/licenses/mit/)