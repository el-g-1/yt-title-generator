import io


def merge_sum(dst, other):
    for key, val in other.items():
        dst[key] += val


def save_vocabulary_tsv(vocab, filepath, save_frequencies=True):
    '''Saves vocabulary into a file'''
    out = io.open(filepath, "w", encoding="utf-8")
    if save_frequencies:
        for tok, count in vocab.items():
            out.write(tok + "\t" + str(count) + "\n")
    else:
        for tok, _ in vocab.items():
            out.write(tok + "\n")
    out.close()


def load_vocabulary_tsv(filepath, freq_min=None, freq_max=None):
    '''Loads vocabulary from a file'''
    with open(filepath, "r") as fdata:
        vocab = {}
        vocab["PAD"] = 0
        for line in fdata.readlines():
            parts = line.split()
            if len(parts) == 1:
                vocab[parts[0]] = len(vocab)
                continue
            if len(parts) != 2:
                continue
            word, freq = parts
            freq = int(freq)
            if freq_min is not None and freq < freq_min:
                continue
            if freq_max is not None and freq > freq_max:
                continue
            vocab[word] = len(vocab)
        return vocab
