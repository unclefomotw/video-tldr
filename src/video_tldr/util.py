def count_words(text: str) -> int:
    """
    Count "words" in this text.

    Words are bounded by whitespace or newlines, which are not counted as words.
    """
    if not text:
        return 0

    words = []
    current_word = []

    for char in text:
        # Skip whitespace - it marks word boundaries
        if char.isspace():
            if current_word:
                words.append(''.join(current_word))
                current_word = []
            continue

        # Non-ASCII character becomes its own word
        if ord(char) > 127:
            if current_word:
                words.append(''.join(current_word))
                current_word = []
            words.append(char)
            continue

        # Special symbol
        if not char.isalpha() and not char.isdigit():
            if current_word:
                words.append(''.join(current_word))
                current_word = []
            words.append(char)
            continue

        # English word - collect consecutive letters or numbers
        if char.isalpha() or char.isdigit():
            current_word.append(char)

    # Don't forget the last word if it exists
    if current_word:
        words.append(''.join(current_word))

    return len(words)


def split_lines_to_chunks(text, max_length: int = 1024) -> list[str]:
    """
    Split text into chunks of at most `max_length` characters.

    This function will split the text into chunks where each chunk has around `max_length`
    characters. It splits the text in the unit of lines and ensures that no chunk exceeds
    the specified maximum length unless the line itself is longer than `max_length`.
    """

    lines = text.splitlines()
    chunks = []
    current_chunk = []
    current_length = 0

    for line in lines:
        if not line:
            continue

        line_length = count_words(line)

        # If adding this line exceeds the max_length
        if current_length + line_length > max_length:
            # Finalize the current chunk and start a new one
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_length = 0

        # Add the line to the current chunk
        current_chunk.append(line)
        current_length += line_length

        # Handle special case where a single line exceeds max_length
        if line_length > max_length:
            chunks.append(line)  # Add the long line as its own chunk
            current_chunk = []
            current_length = 0

    # Add the last chunk if it contains any lines
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks
