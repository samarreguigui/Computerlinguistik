from collections import Counter
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
import random

class Data:

  def __init__(self, *args):
    if len(args) == 1:
        self.init_test(args[0])
    else:
        self.init_train(args[0], args[1])
  def read_data(self, filename):
      data = []
      with open(filename, "r", encoding="utf8") as f:
          for line in f:
              line = line.strip()
              if not line:
                  continue
              parts = line.split("\t")
              if len(parts) != 2:
                  continue
              word, lemma = parts
              data.append((word, lemma))
      return data
  def make_table(self,words):
      counter = Counter()

      for w in words:
          counter.update(list(w))

      chars = [c for c, n in counter.items() if n >= 2]

      table = {}
      table["<unk>"] = 0
      table["<pad>"] = 1

      next_id = 2
      for c in sorted(chars):
          table[c] = next_id
          next_id += 1

      return table

  def init_train(self, traindata, devdata):
    self.train_data = self.read_data(traindata)
    self.dev_data = self.read_data(devdata)

    # zip für Wörter und Lemmata
    train_words, train_lemmas = zip(*self.train_data)

    self.srcChar2ID = self.make_table(train_words)
    self.numSrcChars = len(self.srcChar2ID)

    self.tgtChar2ID = self.make_table(train_lemmas)
    self.numTgtChars = len(self.tgtChar2ID)

    self.ID2tgtChar = {v: k for k, v in self.tgtChar2ID.items()}

    max_ratio = 0.0
    for word, lemma in self.train_data:
        lw = len(word)
        ll = len(lemma)
        if lw > 0:
            ratio = ll / lw
            if ratio > max_ratio:
                max_ratio = ratio

    self.max_len_factor = max_ratio
  def save(self, paramfile):
      params = {
          "srcChar2ID": self.srcChar2ID,
          "ID2tgtChar": self.ID2tgtChar,
          "max_len_factor": self.max_len_factor
      }

      with open(paramfile, "wb") as f:
          pickle.dump(params, f)

  def init_test(self, filename):
      with open(filename, "rb") as f:
          params = pickle.load(f)

      self.srcChar2ID = params["srcChar2ID"]
      self.ID2tgtChar = params["ID2tgtChar"]
      self.max_len_factor = params["max_len_factor"]

  def batches(self, data, max_batch_size):
      batch_words = []
      batch_lemmas = []
      current_size = 0

      def process_batch(words, lemmas):
          # Eingabesequenzen
          src_seqs = []
          src_lengths = []

          for w in words:
              ids = [self.srcChar2ID.get(c, self.srcChar2ID["<unk>"]) for c in w]
              src_lengths.append(len(ids))
              src_seqs.append(torch.tensor(ids, dtype=torch.long))

          srcIDvecs = pad_sequence(
              src_seqs,
              batch_first=False,
              padding_value=self.srcChar2ID["<pad>"]
          )

          # Ausgabesequenzen
          tgt_seqs = []
          for l in lemmas:
              ids = [self.tgtChar2ID.get(c, self.tgtChar2ID["<unk>"]) for c in l]
              ids = [self.tgtChar2ID["<pad>"]] + ids + [self.tgtChar2ID["<pad>"]]
              tgt_seqs.append(torch.tensor(ids, dtype=torch.long))

          tgtIDvecs = pad_sequence(
              tgt_seqs,
              batch_first=False,
              padding_value=self.tgtChar2ID["<pad>"]
          )

          return srcIDvecs, src_lengths, tgtIDvecs

      for word, lemma in data:
          cost = len(word) + len(lemma)

          if batch_words and current_size + cost > max_batch_size:
              yield process_batch(batch_words, batch_lemmas)
              batch_words = []
              batch_lemmas = []
              current_size = 0

          batch_words.append(word)
          batch_lemmas.append(lemma)
          current_size += cost

      if batch_words:
          yield process_batch(batch_words, batch_lemmas)


  def train_batches(self, max_batch_size):
        random.shuffle(self.train_data)
        return self.batches(self.train_data, max_batch_size)



  def dev_batches(self, max_batch_size):
      return self.batches(self.dev_data, max_batch_size)



  def test_batches(self, file, max_batch_size):
      # Wörter aus Datei lesen
      words = []
      with open(file, "r", encoding="utf8") as f:
          for line in f:
              w = line.strip()
              if w:
                  words.append(w)

      batch_words = []
      current_size = 0

      def process_batch(srcs):
          srcID = []
          src_lengths = []

          for w in srcs:
              ids = [self.srcChar2ID.get(c, self.srcChar2ID["<unk>"]) for c in w]
              src_lengths.append(len(ids))
              srcID.append(torch.tensor(ids, dtype=torch.long))

          srcIDvecs = pad_sequence(
              srcID,
              batch_first=False,
              padding_value=self.srcChar2ID["<pad>"]
          )

          maxTgtLen = int(max(src_lengths) * self.max_len_factor + 4)

          return srcs, srcIDvecs, src_lengths, maxTgtLen

      for w in words:
          cost = len(w)

          if batch_words and current_size + cost > max_batch_size:
              yield process_batch(batch_words)
              batch_words = []
              current_size = 0

          batch_words.append(w)
          current_size += cost

      if batch_words:
          yield process_batch(batch_words)

  def tgtIDs2chars(self, tgtCharIDs):
      pad_id = self.tgtChar2ID["<pad>"]
      chars = []

      for idx in tgtCharIDs:
          if idx == pad_id:
              break
          chars.append(self.ID2tgtChar.get(idx, "<unk>"))

      return chars


if __name__ == "__main__":
    # Data Objekt erzeugen und Trainingsmodus verwenden
    data = Data("train.txt", "dev.txt")

    print("Anzahl Source Characters:", data.numSrcChars)
    print("Anzahl Target Characters:", data.numTgtChars)
    print("Max Length Factor:", data.max_len_factor)

    # Erstes Batch aus den Trainingsdaten prüfen
    for srcIDvecs, src_lengths, tgtIDvecs in data.train_batches(200):
        print("srcIDvecs Größe:", srcIDvecs.size())
        print("tgtIDvecs Größe:", tgtIDvecs.size())
        print("src_lengths:", src_lengths)
        break

    # Speichern
    data.save("params.pkl")

    # Laden im Testmodus
    test_data = Data("params.pkl")
    print("Geladen im Testmodus. Max Length Factor:", test_data.max_len_factor)
