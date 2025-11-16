import sys
import math
import pickle
import re
from collections import defaultdict

class CRFTagger:
    def __init__(self, learning_rate = 0.01, l1_lambda: float = 0.0):
        # Gewichtsvektor
        self.weights = defaultdict(float)
        self.tagset = set()
        self.learning_rate = learning_rate
        self.l1_lambda = float(l1_lambda)
        
        # Caches für Merkmalsvektoren und Scores (werden nach jedem Satz gelöscht)
        self._lex_features_cache = {}  # Key: (tag, i)
        self._lex_score_cache = {}    # Key: (tag, i)
        self._context_features_cache = {}  # Key: (prevtag, tag, i)
        self._context_score_cache = {}      # Key: (prevtag, tag, i)
        
        # Lazy L1 Regularisierung: Track wann jedes Gewicht zuletzt aktualisiert wurde
        self._iteration_counter = 0  # Zählt Iterationen (nach jedem Satz erhöht)
        self._last_updated = {}  # feat -> iteration_counter wann zuletzt aktualisiert
    
    def clear_cache(self):
        """Löscht alle Caches nach jedem Satz."""
        self._lex_features_cache.clear()
        self._lex_score_cache.clear()
        self._context_features_cache.clear()
        self._context_score_cache.clear()


    def read_data(self, path):
        """Liest Trainingsdaten und fügt START und END Tokens ein."""
        data = []

        def store_sentence():
            if sentence:
                sentence = [("", "</s>")] + sentence + [("", "</s>")]
                data.append(sentence)
                sentence = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    word, tag = line.split()
                    sentence.append((word, tag))
                else:
                    store_sentence()
        store_sentence()
        self.tagset = set(tag for sent in data for _, tag in sent)
    
        return data

    def lex_features(self, tag, words, i):
        """Lexikalische Merkmale für das Wort an Position i."""
        # Cache-Key: (tag, i)
        cache_key = (tag, i)
        if cache_key in self._lex_features_cache:
            return self._lex_features_cache[cache_key]
        
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
        shape = self.compute_shape(word)
        feats.append(f"SH-{shape}+{tag}")

        # (4) Vorheriges Wort + aktuelles Tag (lexikalisches Merkmal)
        prev_word = words[i - 1] if i > 0 else "<//s>"
        feats.append(f"PW-{prev_word}+{tag}")

        # Im Cache speichern
        self._lex_features_cache[cache_key] = feats
        return feats

    def compute_shape(self, word):
        """Berechnet die Shape-Repräsentation eines Wortes."""
        shape = ''.join('A' if c.isupper() else 'a' if c.islower() else '0' if c.isdigit() else c
                        for c in word)
        shape = re.sub(r'(.)\1+', r'\1', shape)  # wiederholte Buchstaben löschen
        return shape

    def context_features(self, prevtag, tag, words, i):
        """Kontextmerkmale abhängig vom vorherigen Tag."""
        # Cache-Key: (prevtag, tag, i)
        cache_key = (prevtag, tag, i)
        if cache_key in self._context_features_cache:
            return self._context_features_cache[cache_key]
        
        feats = []

        # Vorheriges Tag + aktuelles Tag
        feats.append(f"PT-{prevtag}+{tag}")

        # Im Cache speichern
        self._context_features_cache[cache_key] = feats
        return feats
    
    def _regularize_weight(self, feat):
        """Wendet lazy L1 Regularisierung auf ein Gewicht an.
        
        Holt alle ausstehenden Regularisierungsschritte nach, die seit der
        letzten Aktualisierung hätten angewendet werden sollen.
        """
        if feat not in self._last_updated:
            # Neues Gewicht: markiere als aktualisiert in aktueller Iteration
            self._last_updated[feat] = self._iteration_counter
            return
        
        # Berechne Differenz zwischen aktuellem Counter und last_updated
        iterations_since_update = self._iteration_counter - self._last_updated[feat]
        
        if iterations_since_update == 0:
            # Bereits auf dem neuesten Stand
            return
        
        # Hole Regularisierungsschritte auf einen Rutsch nach
        mu = self.learning_rate * self.l1_lambda
        total_regularization = mu * iterations_since_update
        
        w = self.weights[feat]
        
        # Wende alle ausstehenden Regularisierungsschritte an
        if abs(w) <= total_regularization:
            # |θ| ≤ μ*iterations: Gewicht auf 0 setzen
            self.weights[feat] = 0.0
        else:
            # |θ| > μ*iterations: w := w - sign(w) * μ * iterations
            if w > 0:
                self.weights[feat] = w - total_regularization
            else:
                self.weights[feat] = w + total_regularization
        
        # Markiere als aktualisiert
        self._last_updated[feat] = self._iteration_counter

    def compute_score(self, features):
        """Berechnet Score für eine Liste von Features.
        
        Wichtig: Regularisiert Gewichte vor Verwendung (lazy L1).
        Regularisiert nur einmal pro eindeutigem Feature.
        """
        # Regularisiere alle verwendeten Gewichte vor Verwendung (nur eindeutige)
        unique_features = set(features)
        for feat in unique_features:
            self._regularize_weight(feat)
        
        return sum(self.weights[feat] for feat in features)

    def lex_score(self, tag, words, i):
        """Berechnet lexikalischen Score: Summe aller Gewichte aktiver lexikalischer Merkmale."""
        # Cache-Key: (tag, i)
        cache_key = (tag, i)
        if cache_key in self._lex_score_cache:
            return self._lex_score_cache[cache_key]
        
        score = self.compute_score(self.lex_features(tag, words, i))
        
        # Im Cache speichern
        self._lex_score_cache[cache_key] = score
        return score

    def context_score(self, prevtag, tag, words, i):
        """Berechnet Kontextscore: Summe aller Gewichte aktiver Kontextmerkmale."""
        # Cache-Key: (prevtag, tag, i)
        cache_key = (prevtag, tag, i)
        if cache_key in self._context_score_cache:
            return self._context_score_cache[cache_key]
        
        score = self.compute_score(self.context_features(prevtag, tag, words, i))
        
        # Im Cache speichern
        self._context_score_cache[cache_key] = score
        return score

    def logsumexp(self, scores):
        """Berechnet log-sum-exp zur numerischen Stabilität."""
        m = max(scores)
        return m + math.log(sum(math.exp(s - m) for s in scores))

    def prune_lex_scores(self, lex_scores, threshold=0.001):
        """Pruning: Behält nur Tags mit lex_score > max_lex_score + log(threshold)."""
        if not lex_scores:
            return {}
        max_lex = max(lex_scores.values())
        cut = max_lex + math.log(threshold)
        return {tag: score for tag, score in lex_scores.items() if score >= cut}

    def prune_forward_scores(self, forward_scores, threshold=0.001):
        """Pruning: Behält nur Tags mit forward_score > max_forward_score + log(threshold)."""
        if not forward_scores:
            return {}
        max_forward = max(forward_scores.values())
        cut = max_forward + math.log(threshold)
        return {tag: score for tag, score in forward_scores.items() if score >= cut}

    def forward(self, words):
        """Berechnet Forward-Scores α[i][tag] mit effizienter Berechnung der lexikalischen Scores und Pruning."""
        n = len(words)
        alpha = [{} for _ in range(n)]
        alpha[0]["START"] = 0.0  # log(1)

        for i in range(1, n):
            # Tag-Liste (letzte Position = END)
            tags = self.tagset if i < len(words) - 1 else ["</s>"]

            # (1) Lexikalische Scores einmal pro Position berechnen
            lex_scores_all = {t: self.lex_score(t, words, i) for t in tags}
            
            # Pruning 1: Nur Tags mit lex_score > max_lex_score + log(0.001) behalten
            lex_scores = self.prune_lex_scores(lex_scores_all)

            # (2) Kontext-Scores separat berechnen und addieren
            for tag, lex_s in lex_scores.items():
                scores = [prev_score + self.context_score(prev_tag, tag, words, i) + lex_s
                          for prev_tag, prev_score in alpha[i - 1].items()]
                alpha[i][tag] = self.logsumexp(scores)

            # Pruning 2: Nach Berechnung von forward[i] alle Tags eliminieren, deren
            # Forward-Score unter max_forward_score + log(0.001) liegt
            alpha[i] = self.prune_forward_scores(alpha[i])

        return alpha

    def backward(self, words, alpha):
        """Berechnet Backward-Scores β[i][tag] im Log-Raum.
        
        Pruning 3: Iteriert nur über Tags, die in forward[i] eingetragen sind.
        """
        n = len(words)
        beta = [{} for _ in range(n)]
        # Initialisiere beta[n-1] nur mit Tags aus forward[n-1]
        if "</s>" in alpha[-1]:
            beta[-1]["</s>"] = 0.0  # log(1)

        for i in range(n - 1, 0, -1):
            # Pruning 3: Nur über Tags in forward[i] iterieren
            tags_at_i = list(alpha[i].keys())  # Tags an Position i (aus forward[i])
            
            # Lexikalische Scores nur für Tags in forward[i] berechnen
            lex_scores = {tag: self.lex_score(tag, words, i) for tag in tags_at_i}
            
            # Nur über Tags in forward[i-1] iterieren (prevtag)
            tags_at_prev = list(alpha[i-1].keys()) if i > 1 else ["START"]
            
            for prevtag in tags_at_prev:
                # Summiere über alle Tags in forward[i] (nexttag)
                scores = []
                for tag in tags_at_i:
                    next_score = beta[i][tag]
                    context_s = self.context_score(prevtag, tag, words, i)
                    lex_s = lex_scores[tag]
                    scores.append(next_score + context_s + lex_s)
                
                beta[i-1][prevtag] = self.logsumexp(scores)

        return beta

    def observed_freq(self, words, tags):
        """Zählt beobachtete Merkmalsfrequenzen im Trainingssatz."""
        freq = defaultdict(float)
        for i, (prevtag, tag) in enumerate(zip(tags, tags[1:]), 1):
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
        beta = self.backward(words, alpha)

        n = len(words)
        Z = alpha[-1]['</s>']

        for i in range(1, n):
            lex_scores = {tag: self.lex_score(tag, words, i) for tag in beta[i].keys()}
            for tag, next_score in beta[i].items():
                for prevtag, prev_score in alpha[i-1].items():
                    # log(γ) = α(prevtag,i-1) + score(prevtag,tag,i) + β(tag,i) - log(Z)
                    context_s = self.context_score(prevtag, tag, words, i)
                    lex_s = lex_scores[tag]
                    log_gamma = prev_score + context_s + lex_s + next_score - Z
                    gamma = math.exp(log_gamma)

                    # erwartete lexikalische Features
                    for f in self.lex_features(tag, words, i):
                        freq[f] += gamma
                    # erwartete Kontextfeatures
                    for f in self.context_features(prevtag, tag, words, i):
                        freq[f] += gamma

        return freq
    
    def update_weights(self, words, tags):
        """Aktualisiert Gewichte mit Gradientenverfahren: w += η * (obs - exp).
        
        Mit lazy L1 Regularisierung: Regularisierung wird erst bei Zugriff angewendet.
        """
        obs = self.observed_freq(words, tags)
        exp = self.expected_freq(words)

        # Gradient update
        for feat in set(obs.keys()) | set(exp.keys()):
            # Regularisiere Gewicht vor Update (hole ausstehende Schritte nach)
            self._regularize_weight(feat)
            # Gradienten-Update
            self.weights[feat] += self.learning_rate * (obs[feat] - exp[feat])
            # Markiere als aktualisiert in aktueller Iteration
            self._last_updated[feat] = self._iteration_counter
        
        # Counter nach jedem Satz erhöhen
        self._iteration_counter += 1
    
    def regularize_all_weights(self):
        """Regularisiert alle Gewichte (z.B. vor dem Speichern).
        
        Wichtig: Gewichte müssen regularisiert sein bevor sie gespeichert werden.
        """
        for feat in list(self.weights.keys()):
            self._regularize_weight(feat)

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
            words, tags = zip(*sentence)
            tagger.update_weights(words, tags)
            # Cache nach jedem Satz löschen
            tagger.clear_cache()
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
            # Regularisiere alle Gewichte vor dem Speichern
            tagger.regularize_all_weights()
            best_weights = dict(tagger.weights)
            with open(param_path, "wb") as f:
                pickle.dump({
                    "weights": best_weights,
                    "tagset": list(tagger.tagset)
                }, f)
            print(f"New best dev accuracy {best_acc:.4f} - parameters saved to {param_path}")

    # Zum Schluss sicherstellen, dass die besten Gewichte gespeichert sind
    if best_weights is not None:
        # Regularisiere alle Gewichte vor dem Speichern
        tagger.regularize_all_weights()
        best_weights = dict(tagger.weights)
        with open(param_path, "wb") as f:
            pickle.dump({
                "weights": best_weights,
                "tagset": list(tagger.tagset)
            }, f)
    else:
        # Falls nie verbessert, speichere finale Gewichte
        tagger.regularize_all_weights()
        with open(param_path, "wb") as f:
            pickle.dump({
                "weights": dict(tagger.weights),
                "tagset": list(tagger.tagset)
            }, f)
    
  #  for i, (feat, weight) in enumerate(tagger.weights.items()):
  #      if i >= 10:
  #          break
  #      print(f"{feat}: {weight:.4f}")
