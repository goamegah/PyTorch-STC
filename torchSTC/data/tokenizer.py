import nltk
from nltk.tokenize import word_tokenize
import string

# Télécharger les ressources nécessaires pour NLTK
nltk.download('punkt')

def read_text_file(file_path):
    """Lire le fichier texte et renvoyer son contenu ligne par ligne."""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

def clean_text(text):
    """Nettoyer chaque ligne de texte en supprimant la ponctuation et en le convertissant en minuscules."""
    cleaned_lines = []
    for line in text:
        line = line.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(line.lower())
        cleaned_lines.append(tokens)
    return cleaned_lines

def build_vocab(cleaned_lines):
    """Construire le vocabulaire à partir des données nettoyées."""
    vocab = set()
    for tokens in cleaned_lines:
        vocab.update(tokens)
    return vocab

def build_word_to_index(vocab):
    """Créer un mapping de mots vers leurs indices correspondants."""
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    return word_to_index

def build_index_to_word(vocab):
    """Créer un mapping d'indices vers les mots correspondants."""
    index_to_word = {idx: word for idx, word in enumerate(vocab)}
    return index_to_word
