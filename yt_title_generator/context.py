import os
import json


def get_with_run(data, key, run_id):
    val = data[key]
    if isinstance(val, str):
        return val.format(run=run_id)
    return val


class DbContext:
    def __init__(self, data, run_id):
        self.video_dir = get_with_run(data, "video_dir", run_id)
        self.script_dir = get_with_run(data, "script_dir", run_id)
        self.channel_dir = get_with_run(data, "channel_dir", run_id)

    def __str__(self):
        return str(
            {
                "video_dir": self.video_dir,
                "script_dir": self.script_dir,
                "channel_dir": self.channel_dir,
            }
        )


class EmbeddingContext:
    def __init__(self, data, run_id):
        self.data_dir = get_with_run(data, "data_dir", run_id)
        self.model_dir = get_with_run(data, "model_dir", run_id)
        self.word_frequencies_path = get_with_run(data, "word_frequencies_path", run_id)
        self.vocabulary_path = get_with_run(data, "vocabulary_path", run_id)
        self.num_workers = get_with_run(data, "num_workers", run_id)
        self.window_size = get_with_run(data, "window_size", run_id)
        self.embedding_dim = get_with_run(data, "embedding_dim", run_id)
        self.frequency_min = get_with_run(data, "frequency_min", run_id)
        self.frequency_max = get_with_run(data, "frequency_max", run_id)

    def __str__(self):
        return str(
            {
                "data_dir": self.data_dir,
                "model_dir": self.model_dir,
                "word_frequencies_path": self.word_frequencies_path,
                "vocabulary_path": self.vocabulary_path,
                "num_workers": self.num_workers,
                "window_size": self.window_size,
                "embedding_dim": self.embedding_dim,
                "frequency_min": self.frequency_min,
                "frequency_max": self.frequency_max,
            }
        )


class GeneratorContext:
    def __init__(self, data, run_id):
        self.data_dir = get_with_run(data, "data_dir", run_id)
        self.model_dir = get_with_run(data, "model_dir", run_id)
        self.full_model_dir = get_with_run(data, "full_model_dir", run_id)
        self.num_workers = get_with_run(data, "num_workers", run_id)
        self.max_title_tokens = get_with_run(data, "max_title_tokens", run_id)
        self.max_script_tokens = get_with_run(data, "max_script_tokens", run_id)
        self.num_fake_examples = get_with_run(data, "num_fake_examples", run_id)
        self.latent_dim = get_with_run(data, "latent_dim", run_id)
        self.num_latent_examples = get_with_run(data, "num_latent_examples", run_id)

    def __str__(self):
        return str(
            {
                "data_dir": self.data_dir,
                "model_dir": self.model_dir,
                "num_workers": self.num_workers,
            }
        )


class DiscriminatorContext:
    def __init__(self, data, run_id):
        self.data_dir = get_with_run(data, "data_dir", run_id)
        self.model_dir = get_with_run(data, "model_dir", run_id)
        self.num_workers = get_with_run(data, "num_workers", run_id)
        self.max_title_tokens = get_with_run(data, "max_title_tokens", run_id)
        self.max_script_tokens = get_with_run(data, "max_script_tokens", run_id)
        self.num_fake_examples = get_with_run(data, "num_fake_examples", run_id)
        self.latent_dim = get_with_run(data, "latent_dim", run_id)
        self.num_latent_examples = get_with_run(data, "num_latent_examples", run_id)

    def __str__(self):
        return str(
            {
                "data_dir": self.data_dir,
                "model_dir": self.model_dir,
                "num_workers": self.num_workers,
            }
        )


class GanContext:
    def __init__(self, data, run_id):
        self.discriminator = DiscriminatorContext(data["discriminator"], run_id)
        self.generator = GeneratorContext(data["generator"], run_id)

    def __str__(self):
        return str(
            {
                "discriminator": str(self.discriminator),
                "generator": str(self.generator),
            }
        )


class Context:
    def __init__(self, data, run_id):
        self.db = DbContext(data["db"], run_id)
        self.embedding = EmbeddingContext(data["embedding"], run_id)
        self.gan = GanContext(data["gan"], run_id)

    def __str__(self):
        return str(
            {
                "db": str(self.db),
                "embedding": str(self.embedding),
                "gan": str(self.gan),
            }
        )


def load_context(filepath, run_id):
    with open(os.path.join(filepath), "r") as fdata:
        data = json.load(fdata)
        return Context(data, run_id)
