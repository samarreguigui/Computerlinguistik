import os
import math
import sys
from collections import defaultdict
from train import extract_features

# load file with parameters
def load_weights(paramfile):
    weights = defaultdict(float)
    with open(paramfile, encoding="utf8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                c, f_name, value = parts
                weights[(c, f_name)] = float(value)
    return weights

def classify_mail(mail, classes, weights):
    scores = {}
    for c in classes:
        feats = extract_features(mail, c)
        # score = sum of (weight * feature_value)
        s = sum(weights[f] * v for f, v in feats.items() if f in weights)
        scores[c] = s
    # take class with the highest score
    return max(scores, key=scores.get)

def classify(paramfile, mail_dir, output_file="classifications.txt"):
    classes = [d for d in os.listdir(mail_dir) if os.path.isdir(os.path.join(mail_dir, d))]
    weights = load_weights(paramfile)
    results = []

    # iterate over all mails
    for folder in classes:
        folder_path = os.path.join(mail_dir, folder)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, encoding="latin1") as f:
                text = f.read()
            pred = classify_mail(text, classes, weights)
            results.append((filename, pred))

    # save results
    with open(output_file, "w", encoding="utf8") as out:
        for file_name, pred in results:
            out.write(f"{file_name}\t{pred}\n")

# Kommandozeilenaufruf
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 classify.py paramfile mail-dir")
        sys.exit(1)
    
    classify(sys.argv[1], sys.argv[2])