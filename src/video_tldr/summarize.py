# For now it only supports Ollama serve as inference

from typing import Optional

from video_tldr.ollama import OllamaServer
from video_tldr.util import split_lines_to_chunks

CHUNK_SIZE = 2048

ollama_server = OllamaServer(default_model="phi4", default_num_ctx=8192)


def summarize_whole(text: str, model_name: Optional[str]) -> str:

    system_message = """Your job is to summarize a transcript into a short, 3-minute readable paragraph
in 300 words.

Follow the principles when you do the summarization:
* Treat the article as an objective, third person perspetive, even when the article seems to talk to you.
* The summary you write should be from a third person perspective, describing what the transcript is about
* No critics.  Just summarize emotionlessly.
* Summarize what the article mentions, without adding any other additional information."""

    user_prompt = f"Summarize the following transcript:\n\n{text}"

    return ollama_server.chat(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ],
        model=model_name
    )


def summarize_with_context(chunks: list[str], model_name: Optional[str]) -> str:

    system_message = """Your job is to summarize a long paragraph into a short snippet in 100 words
given the context surrounding the paragraph.

Follow the principles when you do the summarization:
* Treat the article as an objective, third person perspetive, even when the article seems to talk to you.
* The summary you write should be from a third person perspective, describing what the transcript is about
* No critics.  Just summarize emotionlessly.
* Summarize what the article mentions, without adding any other additional information."""

    user_prompt_tmpl = """Summarize only the PART of the text.  The input spec is as follows:

You will be given the paragraph to summarize AND its context enclosed by XML tags <CONTEXT> and </CONTEXT>.
Summarize only the part within <SUMMARIZE_THIS> and </SUMMARIZE_THIS> tags.  You can use the rest of the text
as context, but do not summarize any of the other text.  Do not output anything other than the summarization
of the indicated part of the text.

<CONTEXT>
{context}
</CONTEXT>

To reiterate, you should summarize only this part of the text, shown here again between <SUMMARIZE_THIS> and </SUMMARIZE_THIS>:
<SUMMARIZE_THIS>
{chunk_to_summarize}
</SUMMARIZE_THIS>

Output only the summary of the portion you are asked to summarize, and nothing else."""

    chunk_summaries = []
    for i in range(len(chunks)):
        prev_chunk = chunks[i - 1] if i > 0 else ""
        next_chunk = chunks[i + 1] if i < len(chunks) - 1 else ""
        context = f"{prev_chunk}\n<SUMMARIZE_THIS>\n{chunks[i]}\n</SUMMARIZE_THIS>\n{next_chunk}"
        user_prompt = user_prompt_tmpl.format(context=context,
                                              chunk_to_summarize=chunks[i])
        chunk_summary = ollama_server.chat(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            model=model_name
        )
        chunk_summaries.append(chunk_summary)
    return "\n\n".join(chunk_summaries)


def summarize_text(text: str,
                   model_name: Optional[str] = None,
                   max_chunk_size: int = CHUNK_SIZE) -> str:
    """
    Summarize the given text using a language model.

    Args:
        text (str): The input text to be summarized.  Assumed to have proper line breaks.
        model_name (str): The name of the language model to use for summarization.
        max_chunk_size (int, optional): The maximum number of words allowed per chunk. Defaults to CHUNK_SIZE.

    Returns:
        str: The summarized text.
    """

    chunks = split_lines_to_chunks(text, max_chunk_size)

    if len(chunks) == 0:
        return ""
    elif len(chunks) == 1:
        print("Single chunk detected. Summarizing whole text...")
        return summarize_whole(chunks[0], model_name)
    else:
        print("Multiple chunks detected. Summarizing each chunk and then combining results...")
        return summarize_with_context(chunks, model_name)
