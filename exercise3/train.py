import sys
import random
import os
from collections import defaultdict
import math

def load_mails(data_dir, classes):
    data = []
    for label in classes:
        folder = data_dir + "/" + label
        for filename in os.listdir(folder):
            with open(folder + "/" + filename, encoding="latin1") as f:
                text = f.read()
                data.append((text, label))
    return data

def text_lenght(mail, klasse):
    tokens = mail.split()
    return {(klasse, "LENGTH"): len(tokens)}

def word_frequency(mail, klasse):
    tokens = mail.split()
    features = {}
    for token in tokens:
        key = (klasse, token)
        features[key] = features.get(key, 0) + 1
    return features

def wordpair_frequency(mail, klasse):
    tokens = mail.split()
    features = {}
    for i in range(len(tokens) - 1):
        pair = tokens[i] + "_" + tokens[i + 1]
        key = (klasse, pair)
        features[key] = features.get(key, 0) + 1
    return features

def extract_features(mail, klasse):
    feats = {}
    feats.update(text_lenght(mail, klasse))
    feats.update(word_frequency(mail, klasse))
    feats.update(wordpair_frequency(mail, klasse))
    return feats

def model_train(epochs, train_data, classes, learning_rate=0.1):
    # Gewichtsvektor initialisieren
    weight = defaultdict(float)
    
    #für n Epochen
    for n in range(epochs):
        #Trainingsdaten zufällig umordnen mit random.shuffle
        random.shuffle(train_data)
        
        for text, label in train_data:
            
            #Merkmalsvektoren fur alle Klassen berechnen 
            class_features_vector = {}
            for c in classes:
                class_features_vector[c] = extract_features(text, c)
            
            #Scores fur alle Klassen berechnen
            scores = {}
            for c in classes:
                score = 0.0
                for class_and_feature, vector in class_features_vector[c].items():
                    score += weight[class_and_feature] * vector
                scores[c] = score
            
            #p(Klasse|Mail) fur alle Klassen berechnen
            max_s = max(scores.values())
            logZ = math.log(sum(math.exp(scores[c] - max_s) for c in classes)) + max_s
            p = {c: math.exp(scores[c] - logZ) for c in classes}

            #Gewichte anpassen
            for c in classes:
              for class_and_feature, value in class_features_vector[c].items():
                  if c == label:
                      weight[class_and_feature] += learning_rate * value
                  
                  weight[class_and_feature] -= learning_rate * p[c] * value

        print(f"Epoche {n+1} finished")

    return weight

def evaluate(data, weights, classes):
    correct = 0
    for text, true_label in data:
        feats = {f: v for f, v in extract_features(text, true_label).items() if (true_label, f) in weights}
        scores = {}
        for c in classes:
            score = sum(weights[(c, f)] * v for f, v in feats.items() if (c, f) in weights)
            scores[c] = score
        pred = max(scores, key=scores.get)
        if pred == true_label:
            correct += 1
    return correct / len(data)

def save_weights(weights, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for (cls, feat), w in weights.items():
            f.write(f"{cls}\t{feat}\t{w}\n")

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Usage: python3 train.py train-dir paramfile")
        sys.exit(1)

    train_dir, paramfile = sys.argv[1], sys.argv[2]
    dev_dir = 'dev'
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

    train_data = load_mails(train_dir, classes)
    dev_data = load_mails(dev_dir, classes)
    
    # Verschiede learning rates on dev set
    learning_rates = [0.01, 0.05, 0.1, 0.5]
    best_acc = -1.0
    best_lr = None
    best_weights = None

    for lr in learning_rates:
        print(f"Learning rate is {lr}")
        weights = model_train(7, train_data, classes, learning_rate=lr)
        acc = evaluate(dev_data, weights, classes)
        print(f"Validation accuracy: {acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            best_lr = lr
            best_weights = weights

    print(f"Best learning rate: {best_lr} (accuracy = {best_acc:.3f})")
    save_weights(best_weights, paramfile)
    print(f"Weights were saved to the file {paramfile}")
