import re

def is_formula(text: str) -> bool:
    """
    A more aggressive heuristic to detect if a sentence is likely a formula.
    """
    # Rule 1: Contains an equals sign and is relatively short.
    if '=' in text and len(text.split()) < 15:
        return True
    
    # Rule 2: High density of math-related or special characters.
    math_chars = ['=', '+', '*', '/', '∑', '∫', '∂', '√', '(', ')', '[', ']', '{', '}', '𝜖', '𝛾', '𝜋', '𝜃', '𝜆', '𝝃', '𝜼', '𝛼', '𝜁']
    char_count = len(text)
    if char_count < 10:
        return False
        
    math_char_count = sum(1 for char in text if char in math_chars)
    # If more than 20% of characters are math-related, classify as formula.
    if (math_char_count / char_count) > 0.20:
        return True
        
    return False

def is_caption(text: str) -> bool:
    """Heuristic to detect if a sentence is a figure or table caption."""
    return re.search(r'^\s*(figure|table)\s*\d+', text, re.IGNORECASE) is not None

def is_garbled(text: str) -> bool:
    """Heuristic to detect garbled text from PDF parsing errors (e.g., spaced-out letters)."""
    words = text.split()
    if len(words) < 10:
        return False
    # Count single-letter words (that are not 'a' or 'I')
    single_letter_words = sum(1 for w in words if len(w) == 1 and w.lower() not in ['a', 'i'])
    # If more than 40% of words are non-standard single letters, it's likely garbled.
    if (single_letter_words / len(words)) > 0.4:
        return True
    return False

def clean_text(text: str) -> str:
    """
    A robust helper function to clean raw text.
    This function is intended for cleaning text chunks before sentence tokenization.
    """
    from unstructured.cleaners.core import clean_extra_whitespace, replace_unicode_quotes

    # 1. Use unstructured's pre-built cleaners
    text = replace_unicode_quotes(text)
    
    # 2. Custom regex for specific patterns
    # Remove "Downloaded from..." watermarks
    text = re.sub(r'Downloaded from https?://\S+\s*', '', text, flags=re.IGNORECASE)
    # Join hyphenated words split across lines
    text = text.replace("-\\n", "")
    # Remove soft hyphens
    text = text.replace("\u00ad", "")
    
    # Remove reference markers like [1], [12], [6, 7]
    text = re.sub(r'\[\d+(,\s*\d+)*\]', ' ', text)
    # Remove reference markers like (Yokoi et al., 2015)
    text = re.sub(r'\([A-Za-z\s.,&]+et al\.,\s*\d{4}\)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*[A-Za-z\s.,&]+,\s*\d{4}\s*\)', ' ', text, flags=re.IGNORECASE)
    
    # Attempt to remove long gibberish words with no spaces
    text = re.sub(r'\b[a-zA-Z0-9]{25,}\b', ' ', text)

    # 3. Use unstructured's whitespace cleaner as a final step
    text = clean_extra_whitespace(text)
    
    return text.strip()
