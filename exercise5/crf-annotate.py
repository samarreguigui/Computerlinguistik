import sys
import math
import pickle
from collections import defaultdict
from typing import List


def read_input_sentences(path: str) -> List[List[str]]:
    """Reads sentences from file: one word per line, empty line between sentences."""
    sentences = []
    sent = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if sent:
                    sentences.append(sent)
                    sent = []
                continue
            sent.append(line)
    if sent:
        sentences.append(sent)
    return sentences


def write_output_sentences(path: str, sentences_with_tags: List[List[tuple]]):
    """Writes sentences in training-data format: word<space>tag, blank line between sentences."""
    if path is None:
        out = sys.stdout
        for sent in sentences_with_tags:
            for w, t in sent:
                out.write(f"{w} {t}\n")
            out.write("\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            for sent in sentences_with_tags:
                for w, t in sent:
                    f.write(f"{w} {t}\n")
                f.write("\n")


def viterbi_decode(tagger, words_raw: List[str], beam: int = 8) -> List[str]:
    """Viterbi decoder with pruning, following forward algorithm structure.
    
    words_raw: list of words (no start/end). We add boundary tokens matching training.
    Returns full tag sequence including boundary tags (length = len(words)+2).
    """
    # Prepare words with boundary tokens matching crf-train.read_data format
    # Training uses empty string "" as boundary word and "</s>" as boundary tag
    words = [""] + words_raw + [""]
    n = len(words)
    if n <= 2:
        # empty sentence -> just boundaries
        return ["</s>", "</s>"]

    # Candidate tags = tagger.tagset (set of real tags)
    candidates = list(tagger.tagset)
    if not candidates:
        # nothing to predict, fallback to placeholder tag
        return ["</s>"] * n

    START_TAG = "</s>"
    END_TAG = "</s>"
    
    # Initialize like forward table
    viterbi = [{} for _ in range(n)]
    bestprevtag = [{} for _ in range(n)]
    viterbi[0][START_TAG] = 0.0  # log(1) = 0.0

    # Pruning helper: keeps tags with score > max_score + log(threshold)
    def prune_lex_scores(lex_scores, threshold=0.001):
        if not lex_scores:
            return {}
        max_lex = max(lex_scores.values())
        cut = max_lex + math.log(threshold)
        return {tag: score for tag, score in lex_scores.items() if score >= cut}

    def prune_viterbi_scores(viterbi_scores, threshold=0.001):
        if not viterbi_scores:
            return {}
        max_viterbi = max(viterbi_scores.values())
        cut = max_viterbi + math.log(threshold)
        return {tag: score for tag, score in viterbi_scores.items() if score >= cut}

    # Forward pass: compute viterbi scores
    for i in range(1, n):
        # Tag-Liste (letzte Position = END)
        tags = candidates if i < len(words) - 1 else [END_TAG]

        # (1) Lexikalische Scores einmal pro Position berechnen
        lex_scores_all = {t: tagger.lex_score(t, words, i) for t in tags}
        
        # Pruning 1: Nur Tags mit lex_score > max_lex_score + log(0.001) behalten
        lex_scores = prune_lex_scores(lex_scores_all)

        # (2) Für jedes Tag: max über alle prevtags berechnen
        for tag, lex_s in lex_scores.items():
            # Berechne Scores für alle prevtags
            scores = {}
            for prev_tag, prev_score in viterbi[i - 1].items():
                context_s = tagger.context_score(prev_tag, tag, words, i)
                scores[prev_tag] = prev_score + context_s + lex_s
            
            if scores:
                # viterbi[i][tag] = max(scores.values)
                viterbi[i][tag] = max(scores.values())
                # bestprevtag[i][tag] = argmax(scores)
                bestprevtag[i][tag] = max(scores, key=scores.get)

        # Pruning 2: Nach Berechnung von viterbi[i] alle Tags eliminieren, deren
        # Viterbi-Score unter max_viterbi_score + log(0.001) liegt
        viterbi[i] = prune_viterbi_scores(viterbi[i])
        # Entferne auch aus bestprevtag die geprunten Tags
        bestprevtag[i] = {tag: bestprevtag[i][tag] for tag in viterbi[i].keys()}

    # Extraktion der besten Tagfolge: starte vom End-Tag
    tags = [None] * n
    
    # t_n+1 = endetag (tag vom endesymbol muss endetag sein)
    # Falls END_TAG in viterbi[n-1] vorhanden, verwende es; sonst bestes Tag
    if END_TAG in viterbi[n - 1]:
        tags[-1] = END_TAG
    elif viterbi[n - 1]:
        tags[-1] = max(viterbi[n - 1], key=viterbi[n - 1].get)
    else:
        tags[-1] = END_TAG  # Fallback
    
    # Rückwärts: ti = bestprevtag[i+1][t_i+1]
    # Für Position i: schaue auf Position i+1, nimm bestprevtag[i+1][tags[i+1]]
    for i in range(n - 2, -1, -1):
        if tags[i + 1] is not None and tags[i + 1] in bestprevtag[i + 1]:
            tags[i] = bestprevtag[i + 1][tags[i + 1]]
        else:
            # Fallback: wähle bestes Tag an Position i+1 falls nicht gefunden
            if viterbi[i + 1]:
                tags[i + 1] = max(viterbi[i + 1], key=viterbi[i + 1].get)
                if tags[i + 1] in bestprevtag[i + 1]:
                    tags[i] = bestprevtag[i + 1][tags[i + 1]]
                else:
                    # Wenn kein bestprevtag gefunden, nimm bestes Tag an Position i
                    tags[i] = max(viterbi[i], key=viterbi[i].get) if viterbi[i] else START_TAG
            else:
                tags[i] = START_TAG
    
    # Stelle sicher, dass START_TAG an Position 0 ist
    tags[0] = START_TAG

    return tags


def tag_accuracy(tagger, dev_data):
    """Berechnet Tagging-Genauigkeit auf Development-Daten."""
    correct, total = 0, 0
    for sent in dev_data:
        # sent enthält bereits START/END Tokens (von read_data)
        # Extrahiere nur die echten Wörter (ohne START/END)
        words_raw = [w for w, _ in sent[1:-1]]  # ohne START/END
        gold_tags = [t for _, t in sent[1:-1]]  # ohne START/END
        
        # viterbi_decode erwartet words_raw (ohne Boundaries) und fügt sie selbst hinzu
        pred_tags_full = viterbi_decode(tagger, words_raw)
        # pred_tags_full enthält START/END, extrahiere nur echte Tags
        pred_tags = pred_tags_full[1:-1]
        
        for g, p in zip(gold_tags, pred_tags):
            total += 1
            if g == p:
                correct += 1
    return correct / total if total > 0 else 0.0


def main():
    if not (3 <= len(sys.argv) <= 5):
        print("Usage: python crf-annotate.py param-file input-file output-file [beam]")
        sys.exit(1)

    param_path = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) >= 4 else None
    beam = int(sys.argv[4]) if len(sys.argv) == 5 else 8

    # load parameters
    with open(param_path, "rb") as f:
        params = pickle.load(f)

    weights = params.get("weights", {})
    tagset = params.get("tagset", [])

    # Dynamically load CRFTagger from crf-train.py so this module can be
    # imported without causing circular imports.
    import os
    import importlib.util
    this_dir = os.path.dirname(__file__)
    crf_train_path = os.path.join(this_dir, "crf-train.py")
    spec = importlib.util.spec_from_file_location("crf_train_module", crf_train_path)
    crf_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(crf_mod)
    CRFTagger = getattr(crf_mod, "CRFTagger")

    tagger = CRFTagger()
    # replace weights and tagset
    tagger.weights = defaultdict(float)
    tagger.weights.update(weights)
    tagger.tagset = set(tagset)

    sentences = read_input_sentences(input_path)
    out_sentences = []

    for sent in sentences:
        tags = viterbi_decode(tagger, sent, beam=beam)
        # tags is length len(sent)+2; output only real tokens in training format
        annotated = []
        for i in range(1, len(tags) - 1):
            annotated.append((sent[i - 1], tags[i]))
        out_sentences.append(annotated)

    write_output_sentences(output_path, out_sentences)


if __name__ == "__main__":
    main()
