import argparse
import torch

from Data import Data
from model import Model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()
    
    # Obligatorische Argumente
    parser.add_argument("paramfile")
    parser.add_argument("inputfile")
    
    # Optionale Argumente
    parser.add_argument("--batch_size", type=int, default=3000)
    parser.add_argument("--embeddings_size", type=int, default=100)
    parser.add_argument("--lstm_size", type=int, default=400)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Data aus gespeicherten Parametern laden
    data = Data(args.paramfile + ".io")
    
    num_src_chars = len(data.srcChar2ID)
    num_tgt_chars = len(data.ID2tgtChar)
    
    # Modell initialisieren
    model = Model(
        num_src_chars,
        num_tgt_chars,
        args.embeddings_size,
        args.lstm_size,
        args.dropout_rate,
        1  # pad_id
    ).to(DEVICE)
    
    # Modell-Parameter laden
    model.load_state_dict(torch.load(args.paramfile + ".pth", map_location=DEVICE))
    model.eval()
    
    # Eingabedatei verarbeiten
    all_lemmas = []
    
    with torch.no_grad():
        for srcs, src_ids, src_lengths, max_tgt_length in data.test_batches(args.inputfile, args.batch_size):
            # Transponieren von (seq_len, batch) zu (batch, seq_len)
            src_ids = src_ids.transpose(0, 1).to(DEVICE)
            src_lengths = torch.tensor(src_lengths, device=DEVICE)
            
            # Lemmatisierung
            tgt_char_ids = model.lemmatize(src_ids, src_lengths, max_tgt_length)
            
            # IDs zu Zeichen konvertieren und speichern
            pad_id = 1  # PAD token ID
            for i, word in enumerate(srcs):
                char_ids = tgt_char_ids[i].cpu().tolist()
                chars = []
                for idx in char_ids:
                    if idx == pad_id:
                        break
                    chars.append(data.ID2tgtChar.get(idx, "<unk>"))
                lemma = "".join(chars)
                all_lemmas.append(lemma)
    
    # Ergebnisse ausgeben (ein Lemma pro Zeile)
    for lemma in all_lemmas:
        print(lemma)

if __name__ == "__main__":
    main()

