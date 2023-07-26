from pathlib import Path
from dataclasses import dataclass


@dataclass
class TrainOptions:
    """ Defines all training arguments. """
    debug: bool = False
    data: str = ''
    image_path: Path = Path('')
    batch_size: int = 64
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-3
    patience: int = 2
    factor: float = 0.5
    epochs: int = 2

    model_name: str = 'resnet50'
    image_embedding: int = 2048
    # text_encoder_model = "distilbert-base-uncased"
    text_embedding: int = 1000
    # text_tokenizer = "distilbert-base-uncased"
    # max_length = 200

    pretrained: bool = True # for both image encoder and text encoder
    trainable: bool = True # for both image encoder and text encoder
    temperature: float = 1.0

    # image size
    size: int = 128
    in_chans: int = 3

    # for projection head; used for both image and text encoders
    num_projection_layers: int = 1
    projection_dim: int = 256 
    dropout: int = 0.1

    def update(self, new_opts):
        for key, value in new_opts.items():
            setattr(self, key, value)
