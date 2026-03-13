"""
Text Preprocessing
==================
Cleaning and preparing text data for analysis and machine learning.

Topics:
- Lowercase conversion
- Removing punctuation and special characters
- Tokenization
- Stopwords removal
- Stemming and lemmatization
- Text normalization

Run: python 06_text_preprocessing.py
"""

import re
import string
import pandas as pd

def main():
    print("=" * 60)
    print("Text Preprocessing")
    print("=" * 60)

    # 1. Basic text cleaning
    print("\n1. Basic Text Cleaning")
    print("-" * 40)

    text = "Hello WORLD! This is a Sample Text with 123 Numbers."
    print(f"Original text:\n{text}")

    # Lowercase
    text_lower = text.lower()
    print(f"\nLowercase:\n{text_lower}")

    # Remove numbers
    text_no_numbers = re.sub(r'\d+', '', text_lower)
    print(f"\nRemove numbers:\n{text_no_numbers}")

    # Remove punctuation
    text_no_punct = text_no_numbers.translate(str.maketrans('', '', string.punctuation))
    print(f"\nRemove punctuation:\n{text_no_punct}")

    # Remove extra whitespace
    text_clean = ' '.join(text_no_punct.split())
    print(f"\nRemove extra whitespace:\n{text_clean}")

    # 2. Removing punctuation
    print("\n2. Removing Punctuation")
    print("-" * 40)

    samples = [
        "Hello, World!",
        "What's your name?",
        "I love Python!!! It's amazing.",
        "Email: user@example.com"
    ]

    print("Original texts:")
    for i, sample in enumerate(samples, 1):
        print(f"{i}. {sample}")

    # Method 1: Using string.punctuation
    print("\nMethod 1 - Using string.punctuation:")
    for i, sample in enumerate(samples, 1):
        cleaned = sample.translate(str.maketrans('', '', string.punctuation))
        print(f"{i}. {cleaned}")

    # Method 2: Using regex
    print("\nMethod 2 - Using regex:")
    for i, sample in enumerate(samples, 1):
        cleaned = re.sub(r'[^\w\s]', '', sample)
        print(f"{i}. {cleaned}")

    # 3. Tokenization
    print("\n3. Tokenization")
    print("-" * 40)

    text = "Natural Language Processing is fascinating! It involves computers and human language."
    print(f"Text: {text}")

    # Simple word tokenization
    words = text.split()
    print(f"\nSimple split: {len(words)} tokens")
    print(words)

    # Better tokenization with punctuation handling
    text_clean = text.lower().translate(str.maketrans('', '', string.punctuation))
    words_clean = text_clean.split()
    print(f"\nCleaned split: {len(words_clean)} tokens")
    print(words_clean)

    # Regex-based tokenization
    words_regex = re.findall(r'\b\w+\b', text.lower())
    print(f"\nRegex tokenization: {len(words_regex)} tokens")
    print(words_regex)

    # Sentence tokenization
    text_multi = "First sentence. Second sentence! Third sentence?"
    sentences = re.split(r'[.!?]+', text_multi)
    sentences = [s.strip() for s in sentences if s.strip()]
    print(f"\nSentence tokenization:")
    for i, sent in enumerate(sentences, 1):
        print(f"{i}. {sent}")

    # 4. Stopwords removal
    print("\n4. Stopwords Removal")
    print("-" * 40)

    # Common English stopwords
    stopwords = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 'had',
        'what', 'when', 'where', 'who', 'which', 'why', 'how'
    }

    text = "This is a sample sentence with common stopwords that should be removed"
    words = text.lower().split()

    print(f"Original text: {text}")
    print(f"Original words: {words}")
    print(f"Word count: {len(words)}")

    # Remove stopwords
    words_filtered = [word for word in words if word not in stopwords]

    print(f"\nFiltered words: {words_filtered}")
    print(f"Word count after filtering: {len(words_filtered)}")
    print(f"Removed {len(words) - len(words_filtered)} stopwords")

    # 5. Stemming
    print("\n5. Stemming")
    print("-" * 40)

    def simple_stem(word):
        """
        Simple stemmer - removes common suffixes
        Note: This is a simplified version for educational purposes
        """
        # Remove common suffixes
        suffixes = ['ing', 'ed', 'es', 's', 'ly', 'er', 'est']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word

    words = ['running', 'runs', 'ran', 'easily', 'faster', 'fastest', 'programming', 'programmer']
    print("Original words and their stems:")
    for word in words:
        stem = simple_stem(word)
        print(f"  {word:15s} -> {stem}")

    # Stemming example in context
    text = "The runners were running quickly in the fastest race"
    words = text.lower().split()
    stemmed = [simple_stem(word) for word in words]

    print(f"\nOriginal: {' '.join(words)}")
    print(f"Stemmed:  {' '.join(stemmed)}")

    # 6. Lemmatization (rule-based)
    print("\n6. Lemmatization (Rule-Based)")
    print("-" * 40)

    # Simple lemmatization dictionary
    lemma_dict = {
        'are': 'be', 'is': 'be', 'was': 'be', 'were': 'be', 'been': 'be',
        'running': 'run', 'ran': 'run', 'runs': 'run',
        'better': 'good', 'best': 'good',
        'worse': 'bad', 'worst': 'bad',
        'children': 'child', 'feet': 'foot', 'geese': 'goose'
    }

    def lemmatize(word):
        """Simple lemmatizer using dictionary lookup"""
        return lemma_dict.get(word.lower(), word)

    words = ['running', 'ran', 'runs', 'are', 'were', 'better', 'children', 'feet']
    print("Original words and their lemmas:")
    for word in words:
        lemma = lemmatize(word)
        print(f"  {word:15s} -> {lemma}")

    text = "The children were running faster than their feet could carry them"
    words = text.lower().split()
    lemmatized = [lemmatize(word) for word in words]

    print(f"\nOriginal:   {' '.join(words)}")
    print(f"Lemmatized: {' '.join(lemmatized)}")

    # 7. Complete preprocessing pipeline
    print("\n7. Complete Text Preprocessing Pipeline")
    print("-" * 40)

    def preprocess_text(text, remove_stopwords=True, stem=True):
        """Complete text preprocessing pipeline"""
        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Tokenize
        words = text.split()

        # Remove stopwords
        if remove_stopwords:
            stopwords_set = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
                            'from', 'has', 'he', 'in', 'is', 'it', 'of', 'on', 'that',
                            'the', 'to', 'was', 'will', 'with'}
            words = [w for w in words if w not in stopwords_set]

        # Stem
        if stem:
            words = [simple_stem(w) for w in words]

        # Remove extra whitespace and empty strings
        words = [w for w in words if w.strip()]

        return ' '.join(words)

    texts = [
        "Check out this link: https://example.com #Python @user123",
        "I'm learning Natural Language Processing!!!",
        "The quick brown fox jumps over the lazy dog.",
        "Data Science is amazing! I love Python 3.9"
    ]

    print("Original texts:")
    for i, text in enumerate(texts, 1):
        print(f"{i}. {text}")

    print("\nPreprocessed texts:")
    for i, text in enumerate(texts, 1):
        processed = preprocess_text(text)
        print(f"{i}. {processed}")

    # 8. Text preprocessing for DataFrame
    print("\n8. Text Preprocessing in DataFrames")
    print("-" * 40)

    df = pd.DataFrame({
        'Review': [
            "This product is AMAZING!!! I love it.",
            "Terrible quality. Very disappointed :(",
            "Pretty good, but could be better...",
            "Best purchase ever! Highly recommend!!!",
            "Waste of money. Do NOT buy this."
        ],
        'Rating': [5, 1, 3, 5, 1]
    })

    print("Original reviews:")
    print(df)

    # Apply preprocessing
    df['Review_Cleaned'] = df['Review'].apply(lambda x: preprocess_text(x, remove_stopwords=True, stem=True))

    print("\nWith cleaned reviews:")
    print(df[['Review', 'Review_Cleaned']])

    # Token statistics
    df['Token_Count'] = df['Review_Cleaned'].apply(lambda x: len(x.split()))
    df['Char_Count'] = df['Review_Cleaned'].apply(len)

    print("\nWith statistics:")
    print(df[['Review_Cleaned', 'Token_Count', 'Char_Count']])

    # 9. Advanced text cleaning
    print("\n9. Advanced Text Cleaning")
    print("-" * 40)

    def advanced_clean(text):
        """Advanced cleaning with multiple techniques"""
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Normalize whitespace characters
        text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')

        return text.strip()

    messy_text = """
    <p>Hello World!</p>
    Contact: user@example.com
    Multiple    spaces     here
    Line breaks\nand\ttabs
    """

    print(f"Messy text:\n{messy_text}")
    print(f"\nCleaned text:\n{advanced_clean(messy_text)}")

    # 10. Case studies
    print("\n10. Preprocessing Comparison")
    print("-" * 40)

    sample = "The Runner's World: BEST Running shoes!!! #running @runnerstore https://example.com"

    print(f"Original:\n{sample}\n")

    # No preprocessing
    print(f"No preprocessing:\n{sample}\n")

    # Basic preprocessing (lowercase, remove punctuation)
    basic = sample.lower().translate(str.maketrans('', '', string.punctuation))
    print(f"Basic (lowercase + remove punctuation):\n{basic}\n")

    # Full preprocessing
    full = preprocess_text(sample, remove_stopwords=True, stem=True)
    print(f"Full preprocessing:\n{full}\n")

    print("\n" + "=" * 60)
    print("Summary - Text Preprocessing Steps:")
    print("  1. Lowercase: Convert to lowercase")
    print("  2. Remove punctuation: Clean special characters")
    print("  3. Tokenization: Split into words/sentences")
    print("  4. Stopwords: Remove common words (a, the, is)")
    print("  5. Stemming: Reduce words to root form (running -> run)")
    print("  6. Lemmatization: Reduce to dictionary form (better -> good)")
    print("  7. Special: Remove URLs, mentions, hashtags, numbers")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Exercises:")
    print("1. Build custom stopwords list for specific domain")
    print("2. Implement n-gram tokenization (bigrams, trigrams)")
    print("3. Create text preprocessing for multiple languages")
    print("4. Implement TF-IDF word importance scoring")
    print("5. Build vocabulary from corpus with frequency counts")
    print("=" * 60)

if __name__ == "__main__":
    main()
