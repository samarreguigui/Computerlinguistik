import sys
import math
import pickle
from collections import defaultdict

class CRFTagger:
    def __init__(self, learning_rate = 0.01):
        # Gewichtsvektor
        self.weights = defaultdict(float)
        self.tagset = set()
        self.learning_rate = learning_rate


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
        """Berechnet Forward-Scores α[i][tag] entsprechend Folie."""
        alpha = [defaultdict(float)]
        alpha[0]["START"] = 0.0  # log(1)

        for i in range(1, len(words)):
            alpha.append(defaultdict(float))

            # Bestimme Tag-Liste (letzte Position = END)
            tags = self.tagset if i < len(words) - 1 else ["//s"]

            for tag in tags:
                lexical_score = self.lex_score(tag, words, i)
                total_scores = []

                for prev_tag, prev_score in alpha[i - 1].items():
                    count_score = prev_score + self.context_score(prev_tag, tag, words, i) + lexical_score
                    total_scores.append(count_score)

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

        for feat in set(obs.keys()) | set(exp.keys()):
            self.weights[feat] += self.learning_rate * (obs[feat] - exp[feat])


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
    if len(sys.argv) != 3:
        print("Usage: python crf-train.py train.txt param-file")
        sys.exit(1)

    train_path = sys.argv[1]
    param_path = sys.argv[2]

    tagger = CRFTagger()
    train_data = tagger.read_data(train_path)

    # Einmaliges Training (mehere Epochen sind möglich)
    EPOCHS = 1
    for epoch in range(EPOCHS):
        for sentence in train_data:
            print(sentence)
            words = [w for w, _ in sentence]
            tags = [t for _, t in sentence]
            tagger.update_weights(words, tags)

    # Nach dem Training Parameter und Tagset speichern
    with open(param_path, "wb") as f:
        pickle.dump({
            "weights": dict(tagger.weights),
            "tagset": list(tagger.tagset)
        }, f)
    
  #  for i, (feat, weight) in enumerate(tagger.weights.items()):
  #      if i >= 10:
  #          break
  #      print(f"{feat}: {weight:.4f}")
