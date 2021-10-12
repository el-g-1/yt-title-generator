# yt-title-generator

This is a set of tools to train a machine learning model to generate YouTube video titles that maximize view count.


## Input Data Format

Model training requires information about videos, channels and comments.

1. Video info must be stored in a json file in the following format:
    ```json
    {
      "id": "<VIDEO ID>",
      "comments_threads": [
          {
            "text": "<COMMENT TEXT>"
          },
          ...
      ],
      "info": {
        "snippet": {
          "title": "<VIDEO TITLE>",
          "channelId": "<CHANNEL ID>"
        },
        "statistics": {
          "viewCount": "<VIDEO VIEW COUNT>"
        }
      }
    }
    ```

1. Channel info must be stored in a json file with the name `<CHANNEL ID>` in the following format:
    ```json
    {
      "info": {
        "statistics": {
          "subscriberCount": "<SUBSCRIBER COUNT>"
        }
      }
    }
    ```

1. Script files must be named `<VIDEO ID>` and should be plain text.


## Usage

### Parameters

Parameters for data generation and training are stored in a context file in the following json format:
```json
{       
    "db": { 
        "video_dir": "<PATH TO VIDEO DIRECTORY>",
        "script_dir": "<PATH TO SCRIPT DIRECTORY>",
        "channel_dir": "<PATH TO CHANNEL DIRECTORY>"
    },
    "embedding": {
        "data_dir": "<PATH TO DATA DIRECTORY",
        "model_dir": "<PATH TO MODEL DIRECTORY>",
        "word_frequencies_path": "<PATH TO THE WORD FREQUENCY FILE>",
        "vocabulary_path": "<PATH TO THE VOCABULARY FILE>",
        "window_size": 2, # CBOW window size
        "embedding_dim": 128, # output vector size
        "frequency_min": 10, # min word count for vocabulary
        "frequency_max": null, # max word count for vocabulary
        "num_workers": 1 # number of workers to generate the data
    },
    "gan": {
        "generator": {
            "data_dir": "<PATH TO DATA DIRECTORY>",
            "full_model_dir": "<PATH TO THE FULL GAN MODEL>",
            "model_dir": "<PATH TO THE GENERATOR MODEL>",
            "latent_dim": 32, # dimension of latent space (size of random vector)
            "num_latent_examples": 5, # number of different examples for each script to add to the training set
            "num_workers": 1 # number of workers to generate the data
        },
        "discriminator": {
            "data_dir": "<PATH TO DATA DIRECTORY>",
            "model_dir": "<PATH TO THE DISCRIMINATOR MODEL>",
            "max_title_tokens": 15, # max title length (words)
            "max_script_tokens": 500, # max script length (words)
            "num_fake_examples": 10, # number of fake examples to add to the training set
            "num_workers": 1 # number of workers to generate the data
        }
    }
}
```

:warning: Make sure to delete the comments.

### Model Training
To train the model, follow these steps:
1. Generate data for word embedding: 
    ```bash
   python yt_title_generator/word_embedding/generate_data.py --context <PATH TO THE CONTEXT FILE> --run <RUN ID>
    ```
1. Train word embedding model:
    ```bash
   python yt_title_generator/word_embedding/train_model.py --context <PATH TO THE CONTEXT FILE> --run <RUN ID>
    ```
1. Generate data for discriminator model: 
    ```bash
   python yt_title_generator/generator/generate_discriminator_data.py --context <PATH TO THE CONTEXT FILE> --run <RUN ID>
    ```
1. Train discriminator model:
    ```bash
   python yt_title_generator/generator/train_discriminator_model.py --context <PATH TO THE CONTEXT FILE> --run <RUN ID>
    ```
1. Generate data for generator model: 
    ```bash
   python yt_title_generator/generator/generate_generator_data.py --context <PATH TO THE CONTEXT FILE> --run <RUN ID>
    ```
1. Train generator model:
    ```bash
   python yt_title_generator/generator/train_generator_model.py --context <PATH TO THE CONTEXT FILE> --run <RUN ID>
    ```

`<RUN ID>` is an arbitrary string; all `{run}` substrings in the context file will be replaced with `<RUN ID>`.

### Prediction (Title Generation)

To make prediction using the trained generator model for the specified script:
```bash
python yt_title_generator/generator/predict_generator.py --context <PATH TO THE CONTEXT FILE> --run <RUN ID> --script <PATH TO PLAIN TEXT SCRIPT FILE>
```

