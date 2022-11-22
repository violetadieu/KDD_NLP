import argparse
import os, sys, time
import logging
import pandas as pd
import numpy as np
import torch
import spacy
import re
from torch import nn, optim
from config import *
from models.build_model import build_model
from data import Multi30k
from utils import get_bleu_score, greedy_decode


DATASET = Multi30k()


def train(model, data_loader, optimizer, criterion, epoch, checkpoint_dir):
    model.train()
    epoch_loss = 0

    for idx, (src, tgt) in enumerate(data_loader):
        src = src.to(model.device)
        tgt = tgt.to(model.device)
        tgt_x = tgt[:, :-1]
        tgt_y = tgt[:, 1:]

        optimizer.zero_grad()

        output, _ = model(src, tgt_x)

        y_hat = output.contiguous().view(-1, output.shape[-1])
        y_gt = tgt_y.contiguous().view(-1)
        loss = criterion(y_hat, y_gt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
    num_samples = idx + 1

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f"{epoch:04d}.pt")
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                   }, checkpoint_file)

    return epoch_loss / num_samples


def evaluate(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0

    total_bleu = []
    with torch.no_grad():
        for idx, (src, tgt) in enumerate(data_loader):
            src = src.to(model.device)
            tgt = tgt.to(model.device)
            tgt_x = tgt[:, :-1]
            tgt_y = tgt[:, 1:]

            output, _ = model(src, tgt_x)

            y_hat = output.contiguous().view(-1, output.shape[-1])
            y_gt = tgt_y.contiguous().view(-1)
            loss = criterion(y_hat, y_gt)

            epoch_loss += loss.item()
            score = get_bleu_score(output, tgt_y, DATASET.vocab_tgt, DATASET.specials)
            total_bleu.append(score)
        num_samples = idx + 1

    loss_avr = epoch_loss / num_samples
    bleu_score = sum(total_bleu) / len(total_bleu)
    return loss_avr, bleu_score


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(len(DATASET.vocab_src), len(DATASET.vocab_tgt), device=DEVICE, dr_rate=DROPOUT_RATE)
    model.load_state_dict(torch.load(args.resume_from)["model_state_dict"])

    criterion = nn.CrossEntropyLoss(ignore_index=DATASET.pad_idx)

    train_iter, valid_iter, test_iter = DATASET.get_iter(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    target_file=pd.read_csv("en.csv")
    result=pd.DataFrame({"en":[],
                         "de":[]})
    for idx in range(0,1000):
        item=target_file.loc[idx]
        try:
            now_en=item[0]
            now_de=DATASET.translate(model, now_en, greedy_decode)
            now_de=now_de.replace("<sos>","")
            now_de=now_de.replace("<unk>","")
            now_de=now_de.replace("<eos>","")
            now_df=pd.DataFrame({"en":[now_en],
                                 "de":[now_de]})
            result=pd.concat([result,now_df])
        except:
            break

    #result.to_csv("번역 결과.csv",encoding="utf-8-sig")
        # expected output: "Ein kleines Mädchen klettert in ein Spielhaus aus Holz ."
    print("끝")
    test_loss, bleu_score = evaluate(model, test_iter, criterion)
    logging.info(f"test_loss: {test_loss:.5f}, bleu_score: {bleu_score:.5f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", default="./3000.pt")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args)