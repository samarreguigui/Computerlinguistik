import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW

from Data import Data
from model import Model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("trainfile")
    parser.add_argument("devfile")
    parser.add_argument("paramfile")

    parser.add_argument("--embeddings_size", type=int, default=100)
    parser.add_argument("--lstm_size", type=int, default=400)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=3000)
    parser.add_argument("--dropout_rate", type=float, default=0.5)

    args = parser.parse_args()

    # Data
    data = Data(args.trainfile, args.devfile)
    data.save(args.paramfile + ".io")

    num_src_chars = len(data.srcChar2ID)
    num_tgt_chars = len(data.ID2tgtChar)

    # Model
    model = Model(
        num_src_chars,
        num_tgt_chars,
        args.embeddings_size,
        args.lstm_size,
        args.dropout_rate,
        1
    ).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(ignore_index=1)
    optimizer = AdamW(model.parameters())

    best_acc = 0.0

    for epoch in range(1, args.num_epochs + 1):

        # -------- TRAIN --------
        model.train()
        for src_ids, src_lengths, tgt_ids in data.train_batches(args.batch_size):

            src_ids = src_ids.transpose(0, 1).to(DEVICE)
            tgt_ids = tgt_ids.transpose(0, 1).to(DEVICE)

            optimizer.zero_grad()

            src_lengths = torch.tensor(src_lengths, device=src_ids.device)

            logits = model(src_ids, src_lengths, tgt_ids[:, :-1])

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                tgt_ids[:, 1:].reshape(-1)
            )

            loss.backward()
            optimizer.step()

        # -------- EVAL --------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for src_ids, src_lengths, tgt_ids in data.dev_batches(args.batch_size):

                src_ids = src_ids.transpose(0, 1).to(DEVICE)
                tgt_ids = tgt_ids.transpose(0, 1).to(DEVICE)

                src_lengths = torch.tensor(src_lengths, device=src_ids.device)

                logits = model(
                    src_ids,
                    src_lengths,
                    tgt_ids[:, :-1]
)

                preds = logits.argmax(dim=-1)
                mask = tgt_ids[:, 1:] != 1

                correct += ((preds == tgt_ids[:, 1:]) & mask).sum().item()
                total += mask.sum().item()

        acc = correct / total
        print(acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.paramfile + ".pth")

if __name__ == "__main__":
    main()
