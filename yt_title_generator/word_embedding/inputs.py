import collections
import re


def to_token(word, stemmer):
    '''Stems a word'''
    return stemmer.stem(word.lower())


def ignore_word(word):
    '''Filters out non-words'''
    if word.startswith("@"):
        return True
    if "_" in word:
        return True
    if "." in word:
        return True
    if re.match(r".*[0-9]", word):
        return True
    if re.fullmatch(r"\s+", word):
        return True
    return False


def tokenize(text, stemmer):
    '''Stems each word in the text, preserving emojis'''
    tokens = []
    for word in text.split():
        if ignore_word(word):
            continue
        for tok in re.findall(
            r"[\w]+|[\u263a-\U0001f645]|[\U0001F601-\U0001F94F]", word, re.U
        ):
            tokens.append(to_token(tok, stemmer))
    return tokens


def update_words_frequency(sentence, word_freq):
    '''Updates frequences for every token in a sentence'''
    for word in sentence:
        word_freq[word] += 1


def to_input(text, vocabulary, stemmer):
    '''Translates tokens into numbers / vocabulary indices'''
    return (vocabulary.get(tok, 0) for tok in tokenize(text, stemmer))
