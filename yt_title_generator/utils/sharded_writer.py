import tensorflow as tf
import threading
import os


class ShardedWriter:
    """Writer that shuffles/shards data into multiple files"""

    def __init__(
        self, num_shards=100, output_dir="data", output_filename="data.tfrecord"
    ):
        self.num_shards = num_shards
        shard_padding = len(str(num_shards - 1))
        self.writers = []
        self.locks = []
        for i in range(num_shards):
            w = tf.io.TFRecordWriter(
                os.path.join(
                    output_dir, (output_filename + "." + str(i).zfill(shard_padding))
                )
            )
            self.writers.append(w)
            self.locks.append(threading.Lock())

    def write(self, data):
        """Writes data into one of the shards"""
        serialized = data.SerializeToString()
        shard = hash(serialized) % self.num_shards
        with self.locks[shard]:
            self.writers[shard].write(serialized)

    def close(self):
        for w in self.writers:
            w.close()

    def __enter__(self):
        pass

    def __exit__(self):
        self.close()
