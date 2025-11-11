import sys
import math
import pickle
from collections import defaultdict

class CRFTagger:
    def __init__(self, learning_rate = 0.01, l1_lambda: float = 0.0):
        # Gewichtsvektor
        self.weights = defaultdict(float)
        self.tagset = set()
        self.learning_rate = learning_rate
        # L1 regularization strength (λ). Applied via proximal operator
        # (soft-thresholding) after each gradient step.
        self.l1_lambda = float(l1_lambda)


    def read_data(self, path):
        """Liest Trainingsdaten und fügt START und END Tokens ein."""
        data, sentence = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if sentence:
                        sentence = [("< >", "<//s>")] + sentence + [("< >", "<//s>")]
                        data.append(sentence)
                        sentence = []
                    continue
                word, tag = line.split()
                sentence.append((word, tag))
                self.tagset.add(tag)
        if sentence:
            sentence = [("< >", "<//s>")] + sentence + [("< >", "<//s>")]
            data.append(sentence)
        return data

    def lex_features(self, tag, words, i):
        """Lexikalische Merkmale für das Wort an Position i."""
        word = words[i]
        feats = []

        # (1) Wort + Tag
        feats.append(f"WT-{word}+{tag}")

        # (2) Wort-Suffix + Tag (für Suffixlängen 2–5)
        for l in range(2, 6):
            if len(word) >= l:
                suffix = word[-l:]
                feats.append(f"ST-{suffix}+{tag}")

        # (3) Wortform (Shape) + Tag
        shape = ''.join(
            'A' if c.isupper() else 'a' if c.islower() else '0'
            for c in word
        )
        feats.append(f"SH-{shape}+{tag}")

        return feats

    def context_features(self, prevtag, tag, words, i):
        """Kontextmerkmale abhängig vom vorherigen Wort und Tag."""
        feats = []

        # (4) Vorheriges Wort + aktuelles Tag
        prev_word = words[i - 1] if i > 0 else "<//s>"
        feats.append(f"PW-{prev_word}+{tag}")

        # (5) Vorheriges Tag + aktuelles Tag
        feats.append(f"PT-{prevtag}+{tag}")

        return feats
    
    def lex_score(self, tag, words, i):
        """Berechnet lexikalischen Score: Summe aller Gewichte aktiver lexikalischer Merkmale."""
        score = 0.0
        for feat in self.lex_features(tag, words, i):
            score += self.weights[feat]      # Gewicht * 1, weil Merkmal aktiv
        return score

    def context_score(self, prevtag, tag, words, i):
        """Berechnet Kontextscore: Summe aller Gewichte aktiver Kontextmerkmale."""
        score = 0.0
        for feat in self.context_features(prevtag, tag, words, i):
            score += self.weights[feat]      # Gewicht * 1, weil Merkmal aktiv
        return score

    def compute_score(self, prevtag, tag, words, i):
        """Kombiniert lexikalischen und Kontextscore zu einem Gesamtscore."""
        return self.lex_score(tag, words, i) + self.context_score(prevtag, tag, words, i)

    def forward(self, words):
        """Berechnet Forward-Scores α[i][tag] mit effizienter Berechnung der lexikalischen Scores."""
        alpha = [defaultdict(float)]
        alpha[0]["START"] = 0.0  # log(1)

        for i in range(1, len(words)):
            alpha.append(defaultdict(float))

            # Tag-Liste (letzte Position = END)
            tags = self.tagset if i < len(words) - 1 else ["//s"]

            # (1) Lexikalische Scores einmal pro Position berechnen
            lex_scores = {t: self.lex_score(t, words, i) for t in tags}

            # (2) Kontext-Scores separat berechnen und addieren
            for tag in tags:
                total_scores = []
                lex_s = lex_scores[tag]  # wiederverwendet statt neu berechnet
                for prev_tag, prev_score in alpha[i - 1].items():
                    cs = self.context_score(prev_tag, tag, words, i)
                    total_scores.append(prev_score + cs + lex_s)

                # Log-Summe zur Stabilität (Unterlauf vermeiden)
                m = max(total_scores)
                alpha[i][tag] = m + math.log(sum(math.exp(s - m) for s in total_scores))

        return alpha

    def backward(self, words):
        """Berechnet Backward-Scores β[i][tag] im Log-Raum."""
        n = len(words)
        beta = [defaultdict(float) for _ in range(n)]
        beta[-1]["END"] = 0.0  # log(1)

        for i in range(n - 2, -1, -1):
            for tag in self.tagset if i > 0 else ["START"]:
                total_scores = []
                for next_tag, next_score in beta[i + 1].items():
                    count_score = (
                        next_score
                        + self.context_score(tag, next_tag, words, i + 1)
                        + self.lex_score(next_tag, words, i + 1)
                    )
                    total_scores.append(count_score)

                m = max(total_scores)
                beta[i][tag] = m + math.log(sum(math.exp(s - m) for s in total_scores))

        return beta

    def observed_freq(self, words, tags):
        """Zählt beobachtete Merkmalsfrequenzen im Trainingssatz."""
        freq = defaultdict(float)
        for i in range(1, len(words)):
            tag = tags[i]
            prevtag = tags[i - 1]

            # lexikalische Merkmale
            for f in self.lex_features(tag, words, i):
                freq[f] += 1.0

            # kontextuelle Merkmale
            for f in self.context_features(prevtag, tag, words, i):
                freq[f] += 1.0
        return freq
    
    def expected_freq(self, words):
        """Berechnet erwartete Merkmalsfrequenzen aus Forward/Backward."""
        freq = defaultdict(float)
        alpha = self.forward(words)
        beta = self.backward(words)

        n = len(words)
        # Normierungskonstante Z = log-sum-exp über letzte α-Werte
        Z = max(alpha[-1].values())
        Z += math.log(sum(math.exp(v - Z) for v in alpha[-1].values()))

        for i in range(1, n):
            for tag in self.tagset:
                for prevtag in self.tagset:
                    # log(γ) = α(prevtag,i-1) + score(prevtag,tag,i) + β(tag,i) - log(Z)
                    score = self.compute_score(prevtag, tag, words, i)
                    log_gamma = (
                        alpha[i - 1][prevtag]
                        + score
                        + beta[i][tag]
                        - Z
                    )
                    gamma = math.exp(log_gamma)

                    # erwartete lexikalische Features
                    for f in self.lex_features(tag, words, i):
                        freq[f] += gamma
                    # erwartete Kontextfeatures
                    for f in self.context_features(prevtag, tag, words, i):
                        freq[f] += gamma

        return freq
    
    def update_weights(self, words, tags):
        """Aktualisiert Gewichte mit Gradientenverfahren: w += η * (obs - exp)."""
        obs = self.observed_freq(words, tags)
        exp = self.expected_freq(words)

        # Gradient update
        for feat in set(obs.keys()) | set(exp.keys()):
            self.weights[feat] += self.learning_rate * (obs[feat] - exp[feat])

        # L1 Regularisierung
        for feat in list(self.weights.keys()):
            w = self.weights[feat]
            if w > 0:
                w_new = max(0.0, w - self.learning_rate * self.l1_lambda)
            else:
                w_new = min(0.0, w + self.learning_rate * self.l1_lambda)
            self.weights[feat] = w_new

    # No evaluate_on_file here — evaluation is done by calling
    # `tag_accuracy` from `crf-annotate.py` directly in the training loop.

    #test
    def check_feature_names(self, words, tags):
        """Überprüft, ob alle erzeugten Merkmalsnamen korrekt formatiert sind."""
        all_features = []

        for i in range(1, len(words)):
            tag = tags[i]
            prevtag = tags[i - 1]

            all_features.extend(self.lex_features(tag, words, i))
            all_features.extend(self.context_features(prevtag, tag, words, i))

        for feat in all_features:
            if "-" in feat and "+" in feat:
                print("OK:", feat)
            else:
                print("FEHLER:", feat)

if __name__ == "__main__":
    if not (4 <= len(sys.argv) <= 5):
        print("Usage: python crf-train.py train.txt dev.txt param-file [l1_lambda]")
        sys.exit(1)

    train_path = sys.argv[1]
    dev_path = sys.argv[2]
    param_path = sys.argv[3]
    l1_lambda = float(sys.argv[4]) if len(sys.argv) == 5 else 0.0

    tagger = CRFTagger(l1_lambda=l1_lambda)
    train_data = tagger.read_data(train_path)

    # Einmaliges Training (mehrere Epochen sind möglich)
    EPOCHS = 3
    best_acc = -1.0
    best_weights = None

    for epoch in range(EPOCHS):
        for sentence in train_data:
            # optional: comment out verbose print if too noisy
            # print(sentence)
            words = [w for w, _ in sentence]
            tags = [t for _, t in sentence]
            tagger.update_weights(words, tags)
        # Nach jeder Epoche: Evaluierung auf Development-Daten
        # Dynamically load tag_accuracy from crf-annotate.py and call it.
        import os
        import importlib.util
        this_dir = os.path.dirname(__file__)
        crf_annotate_path = os.path.join(this_dir, "crf-annotate.py")
        spec = importlib.util.spec_from_file_location("crf_annotate_module", crf_annotate_path)
        crf_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(crf_mod)
        tag_accuracy = getattr(crf_mod, "tag_accuracy")

        dev_data = tagger.read_data(dev_path)
        acc = tag_accuracy(tagger, dev_data)
        print(f"Epoch {epoch+1}/{EPOCHS} - dev accuracy: {acc:.4f}")

        # Falls verbessert: aktuelle Gewichte speichern (Parameterdatei)
        if acc > best_acc:
            best_acc = acc
            best_weights = dict(tagger.weights)
            with open(param_path, "wb") as f:
                pickle.dump({
                    "weights": best_weights,
                    "tagset": list(tagger.tagset)
                }, f)
            print(f"New best dev accuracy {best_acc:.4f} - parameters saved to {param_path}")

    # Zum Schluss sicherstellen, dass die besten Gewichte gespeichert sind
    if best_weights is not None:
        with open(param_path, "wb") as f:
            pickle.dump({
                "weights": best_weights,
                "tagset": list(tagger.tagset)
            }, f)
    else:
        # Falls nie verbessert, speichere finale Gewichte
        with open(param_path, "wb") as f:
            pickle.dump({
                "weights": dict(tagger.weights),
                "tagset": list(tagger.tagset)
            }, f)
    
  #  for i, (feat, weight) in enumerate(tagger.weights.items()):
  #      if i >= 10:
  #          break
  #      print(f"{feat}: {weight:.4f}")
