import torch
from torch import nn
import torch.nn.functional as F

from SCLIP.modules import ImageEncoder, ProjectionHead


class CLIPModel(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(model_name=cfg.model_name, 
                                          pretrained=cfg.pretrained, 
                                          trainable=cfg.trainable,
                                          in_chans=cfg.in_chans)
        # self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=cfg.image_embedding, 
                                               projection_dim=cfg.projection_dim,
                                               dropout=cfg.dropout)
        self.text_projection = ProjectionHead(embedding_dim=cfg.text_embedding, 
                                               projection_dim=cfg.projection_dim,
                                               dropout=cfg.dropout)
        self.temperature = cfg.temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = batch["caption"]
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    CLIP = CLIPModel()
    loss = CLIP(batch)
    print("")