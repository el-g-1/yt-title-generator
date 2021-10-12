import tensorflow as tf


def make_int_feature(ints):
    """Creates tf.train.Feature with the given list of ints"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=ints))


def make_string_feature(strings):
    """Creates tf.train.Feature with the given list of strings"""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[s.encode("utf-8") for s in strings])
    )


def make_example(features):
    """Creates tf.train.Example for the given dict of features"""
    return tf.train.Example(features=tf.train.Features(feature=features))


def pad_sequence(seq, maxlen):
    """Pads the input sequence to maxlen"""
    return tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=maxlen)[0]
