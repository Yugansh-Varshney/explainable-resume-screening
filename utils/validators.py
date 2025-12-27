import re

STOPWORDS = {
    "hi", "hello", "hey", "bro", "what", "whats", "up", "ok", "okay",
    "test", "testing", "lol"
}

def is_valid_text(text: str, min_words: int = 15) -> bool:
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())

    meaningful_words = [
        w for w in words if w not in STOPWORDS
    ]

    return len(meaningful_words) >= min_words
