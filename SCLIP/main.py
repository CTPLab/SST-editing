import os
# compatible to requests==2.27.1
# bypass proxy
# os.environ['CURL_CA_BUNDLE'] = ''
import gc
import pyrallis
import dataclasses
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
from torch import nn

from Dataset.transform import transform

from SCLIP.CLIP import CLIPModel
from SCLIP.dataset import CLIPDataset
from SCLIP.utils import AvgMeter, get_lr
from SCLIP.train_options import TrainOptions
np.random.seed(1)

def make_train_valid_dfs(pth, debug):
    df = pd.read_csv(str(pth / "metadata_img.csv"),
                     index_col=0)
    if debug:
        df = df.iloc[:100]
   
    path_id = df.index.values.tolist()
    valid_ids = np.random.choice(
        path_id, size=int(0.1 * len(df)), replace=False
    )
    train_df = df
    valid_df = df[df.index.isin(valid_ids)]
    return train_df, valid_df


def build_loaders(df, mode, size, path,
                  batch_size, num_workers):
    dataset = CLIPDataset(
        path,
        df.index.values,
        df.astype(np.int16).to_numpy(),
        transform=transform if mode == "train" else None,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items()}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items()}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

def get_save_dict(model, CFG):
    save_dict = {
        'state_dict': model.state_dict(),
        'cfg': dataclasses.asdict(CFG)}
    return save_dict

@pyrallis.wrap()
def main(CFG: TrainOptions):
    CFG.image_path = Path(f'Data/{CFG.data}/GAN/crop')
    train_df, valid_df = make_train_valid_dfs(CFG.image_path, CFG.debug)
    print(train_df.head(), len(train_df))
    print(valid_df.head(), len(valid_df))

    # tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader = build_loaders(train_df, mode="train", 
                                 size=CFG.size,
                                 path=CFG.image_path,
                                 batch_size=CFG.batch_size, 
                                 num_workers=CFG.num_workers)
    valid_loader = build_loaders(valid_df, mode="valid",
                                 size=CFG.size,
                                 path=CFG.image_path,
                                 batch_size=CFG.batch_size, 
                                 num_workers=CFG.num_workers)
    print(len(train_loader), len(valid_loader))

    model = CLIPModel(CFG).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(get_save_dict(model, CFG), 
                       f"SCLIP/{CFG.data}_best_{CFG.size}_{CFG.pretrained}.pt")
            print("Saved Best Model!")
        torch.save(get_save_dict(model, CFG), 
                   f"SCLIP/{CFG.data}_{epoch}_{CFG.size}_{CFG.pretrained}.pt")


if __name__ == "__main__":
    main()
