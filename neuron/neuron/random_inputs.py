import os
import sys
from os import urandom
from random import sample, shuffle
from zoneinfo import ZoneInfo

import nltk

nltk.download('words')
nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger_eng')

from nltk.corpus import words
from nltk import pos_tag

BENCHMARK_SAMPLE_COUNT = 35
INFERENCE_SOCKET_TIMEOUT = 240
TIMEZONE = ZoneInfo("US/Pacific")

AVAILABLE_WORDS = [word for word, tag in pos_tag(words.words(), tagset='universal') if tag == "ADJ" or tag == "NOUN"]


def generate_random_prompt():
    sampled_words = sample(AVAILABLE_WORDS, k=min(len(AVAILABLE_WORDS), min(urandom(1)[0] % 32, 8)))
    shuffle(sampled_words)

    return ", ".join(sampled_words)


def random_seed():
    return int.from_bytes(os.urandom(4), sys.byteorder)
