import re
from typing import Tuple, Dict, Any, List

# Regex for academic paper metadata
# Improved author regex: handles multiple authors, initials, "and", and variations
# e.g., "A. Paolillo, J. Doe, and C. Smith" or "Paolillo, Antonio and Doe, John"
AUTHOR_REGEX = re.compile(
    r'^(?!\s*(?:abstract|introduction|doi|keywords)\s*:)(?:[A-Z][A-Za-z\'.`\s-]+\s?,\s?)+',
    re.IGNORECASE
)
# DOI regex
DOI_REGEX = re.compile(r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b', re.IGNORECASE)


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

def is_low_alphanum_ratio(text: str, threshold: float = 0.6) -> bool:
    """
    Checks if the ratio of alphanumeric characters to total characters is below a threshold.
    Useful for filtering out noise or layout artifacts (e.g. "......" or "| | |").
    """
    if not text:
        return True
    alphanum_count = len(re.findall(r'[a-zA-Z0-9]', text))
    ratio = alphanum_count / len(text)
    return ratio < threshold

def is_reference_section(text: str) -> bool:
    """Detects if the text block is likely a reference list item."""
    # Matches patterns like "[1] J. Smith..." or "(Smith et al., 2021)" at the start
    return re.search(r'^\s*(\[\d+\]|\(\d{4}\b)', text) is not None

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
    
    # Remove citation markers like [1], [12], [6, 7] within the text
    text = re.sub(r'\[\d+(,\s*\d+)*\]', ' ', text)
    # Remove reference markers like (Yokoi et al., 2015)
    text = re.sub(r'\([A-Za-z\s.,&]+et al\.,\s*\d{4}\)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*[A-Za-z\s.,&]+,\s*\d{4}\s*\)', ' ', text, flags=re.IGNORECASE)
    
    # Attempt to remove long gibberish words with no spaces
    text = re.sub(r'\b[a-zA-Z0-9]{25,}\b', ' ', text)

    # 3. Use unstructured's whitespace cleaner as a final step
    text = clean_extra_whitespace(text)
    
    return text.strip()

def extract_metadata_and_clean(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extracts metadata (DOI, authors) from a text chunk, removes it,
    and then cleans the remaining text.
    
    Returns a tuple of (cleaned_text, metadata_dict).
    """
    metadata = {}
    
    # 1. Extract DOI
    doi_match = DOI_REGEX.search(text)
    if doi_match:
        metadata['doi'] = doi_match.group(1)
        text = DOI_REGEX.sub('', text) # Remove from text

    # 2. Extract Authors (run on the first few lines of a document typically)
    # This regex is simplified and works best on headers.
    author_match = AUTHOR_REGEX.search(text)
    if author_match:
        authors_str = author_match.group(0).strip()
        # Clean up the matched string
        authors_str = re.sub(r'\s+and\s+$', '', authors_str.replace('\n', ' ')).strip()
        authors = [a.strip() for a in authors_str.split(',') if a.strip()]
        if authors:
            metadata['authors'] = authors
            text = AUTHOR_REGEX.sub('', text) # Remove from text

    # 3. Clean the remaining text
    cleaned_text = clean_text(text)
    
    return cleaned_text, metadata
