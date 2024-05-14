from os import urandom
from random import sample, shuffle

import nltk

nltk.download('words')
nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import words
from nltk import pos_tag


WORDS = [word for word, tag in pos_tag(words.words(), tagset='universal') if tag == "ADJ" or tag == "NOUN"]


def generate_random_prompt():
    words = sample(WORDS, k=min(len(WORDS), min(urandom(1)[0] % 32, 8)))
    shuffle(words)

    return ", ".join(words)
