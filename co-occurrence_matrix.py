from os.path import basename
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def main(*docs) -> None:
    documents = dict()
    all_tokens = []
    ALLOWED_CHARACTERS = "abcdefghijklmnopqrstuvwxyz -"
    for doc in docs:

        # Preprocessing
        with open(doc, 'r') as f:
            cleaned_text = text_to_single_paragraph(text=f.read())
        cleaned_text = cleaned_text.lower()
        cleaned_text = remove_unallowed_characters(text=cleaned_text, alphabet=ALLOWED_CHARACTERS)
        word_tokens = word_tokenize(cleaned_text)
        filtered_word_tokens = remove_stop_words(word_tokens=word_tokens)
        stemmed_word_tokens = stem_word_tokens(word_tokens=filtered_word_tokens)

        documents[basename(doc)] = stemmed_word_tokens
        all_tokens.extend(stemmed_word_tokens)

    all_tokens = list(set(all_tokens))

    matrix = [['#'] + all_tokens]

    for doc in docs:
        matrix.append([basename(doc)])
        for token in all_tokens:
            matrix[-1].append(documents[basename(doc)].count(token))

    # You can print only a subset of the matrix by assigning a limit number
    limit_number = 10
    for lst in matrix:
        print(lst[:limit_number])


def text_to_single_paragraph(text: str) -> str:
    # Convert the whole text to one paragraph
    return text.replace('\n', ' ')


def remove_unallowed_characters(text: str, alphabet: str) -> str:
    temp = text
    for char in text:
        if char not in alphabet:
            temp = temp.replace(char, '')
    return temp


def remove_stop_words(word_tokens: list) -> list:
    return [token for token in word_tokens if token not in stopwords.words('english') and len(token) > 1]


def stem_word_tokens(word_tokens: list) -> list:
    """
    Using Porter stemmer to stem all words
    """
    porter = PorterStemmer()
    return [porter.stem(token) for token in word_tokens]
