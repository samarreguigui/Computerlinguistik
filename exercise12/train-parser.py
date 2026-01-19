import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import importlib.util
from pathlib import Path
from parser import Parser

# Filename has hyphen
spec = importlib.util.spec_from_file_location("tree_reader", Path(__file__).parent / "tree-reader.py")
tree_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tree_reader)
parse = tree_reader.parse

PADDING_ID = 0
UNK_CHAR_ID = 1
NO_CONST_LABEL_ID = 0
MAX_CHAR_SEQ_LEN = 20


class Vocab:
    """
    Vocabulary class for mapping characters and labels to IDs.
    """
    
    def __init__(self, train_data):
        """
        Args:
            training_data: List of (words, constituents) pairs
        """
        # Collect all unique characters and labels
        all_chars = set()
        all_labels = set()
        
        for words, constituents in train_data:
            # Collect characters from words
            for word in words:
                all_chars.update(word.lower())
            # Collect labels from constituents
            for label, start, end in constituents:
                all_labels.add(label)
        
        # 0 = PADDING, 1 = UNK_CHAR, 2+ = actual characters
        self.char_to_id = {'<PAD>': PADDING_ID, '<UNK>': UNK_CHAR_ID}
        char_id = 2
        for char in sorted(all_chars):
            if char not in self.char_to_id:
                self.char_to_id[char] = char_id
                char_id += 1
        
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        
        # 0 = No_Constituent, 1+ = actual labels
        self.label_to_id = {'<No_Constituent>': NO_CONST_LABEL_ID}
        label_id = 1
        for label in sorted(all_labels):
            if label not in self.label_to_id:
                self.label_to_id[label] = label_id
                label_id += 1
        
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
    
    def words_to_char_id_tensors(self, words):
        """
        Convert a list of words to suffix and prefix tensors.
        
        Args:
            words: List of words (strings)
            
        Returns:
            prefix_tensor: Tensor of shape (len(words), MAX_CHAR_SEQ_LEN)
            suffix_tensor: Tensor of shape (len(words), MAX_CHAR_SEQ_LEN)
        """
        suffix_ids = []
        prefix_ids = []
        
        for word in words:
            word_lower = word.lower()
            chars = list(word_lower)
            
            suffix_char_ids = [self.char_to_id.get(c, UNK_CHAR_ID) for c in chars]
            if len(suffix_char_ids) < MAX_CHAR_SEQ_LEN:
                padding = [PADDING_ID] * (MAX_CHAR_SEQ_LEN - len(suffix_char_ids))
                suffix_char_ids = padding + suffix_char_ids
            else:
                suffix_char_ids = suffix_char_ids[-MAX_CHAR_SEQ_LEN:]
            suffix_ids.append(suffix_char_ids)
            
            prefix_chars = chars[::-1]
            prefix_char_ids = [self.char_to_id.get(c, UNK_CHAR_ID) for c in prefix_chars]
            if len(prefix_char_ids) < MAX_CHAR_SEQ_LEN:
                padding = [PADDING_ID] * (MAX_CHAR_SEQ_LEN - len(prefix_char_ids))
                prefix_char_ids = padding + prefix_char_ids
            else:
                prefix_char_ids = prefix_char_ids[-MAX_CHAR_SEQ_LEN:]
            prefix_ids.append(prefix_char_ids)
        
        suffix_tensor = torch.tensor(suffix_ids, dtype=torch.long)
        prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long)
        
        return prefix_tensor, suffix_tensor
    
    def num_char_types(self):
        """Return the number of character types."""
        return len(self.char_to_id)
    
    def num_label_types(self):
        """Return the number of label types."""
        return len(self.label_to_id)
    
    def store_parameters(self, filename):
        """
        Store vocabulary mappings to a file.
        """
        import pickle
        vocab_data = {
            'char_to_id': self.char_to_id,
            'id_to_char': self.id_to_char,
            'label_to_id': self.label_to_id,
            'id_to_label': self.id_to_label
        }
        with open(filename, 'wb') as f:
            pickle.dump(vocab_data, f)
    
def compute_errors(model, vocab, data, device):
    """
    Compute the number of incorrectly labeled spans in the data.
    """
    model.eval()
    total_errors = 0
    
    with torch.no_grad():
        for words, gold_constituents in data:
            if len(words) == 0:
                continue
            
            # Words to character tensors
            prefix_tensor, suffix_tensor = vocab.words_to_char_id_tensors(words)
            prefix_tensor = prefix_tensor.to(device)
            suffix_tensor = suffix_tensor.to(device)
            
            # Model predictions
            category_scores, _, span_positions = model(suffix_tensor, prefix_tensor)
            predicted_labels = torch.argmax(category_scores, dim=1)
            
            # Create gold label vector
            N = len(words)
            num_spans = len(span_positions)
            label_ids = torch.full((num_spans,), NO_CONST_LABEL_ID, dtype=torch.long, device=device)
            
            # Create span_id mapping
            span_id = {}
            for idx, (s, e) in enumerate(span_positions):
                span_id[(s, e)] = idx
            
            # Set gold labels
            for label, start, end in gold_constituents:
                if (start, end) in span_id:
                    label_id = vocab.label_to_id.get(label, NO_CONST_LABEL_ID)
                    label_ids[span_id[(start, end)]] = label_id
            
            # Count errors
            errors = (predicted_labels != label_ids).sum().item()
            total_errors += errors
    
    model.train()
    return total_errors


def train(train_file, dev_file, output_base, num_epochs=50):
    """
    Train the parser model.
    
    Args:
        train_file: Path to training data file
        dev_file: Path to development data file
        output_base: Base name for output files
        num_epochs: Number of training epochs
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            result = parse(line)
            if result is None:
                continue
            _, words, constituents = result
            train_data.append((words, constituents))
    print(f"Loaded {len(train_data)} training sentences")
    
    print("Reading development data...")
    dev_data = []
    with open(dev_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            result = parse(line)
            if result is None:
                continue
            _, words, constituents = result
            dev_data.append((words, constituents))
    print(f"Loaded {len(dev_data)} development sentences")
    
    print("Creating vocabulary...")
    vocab = Vocab(train_data)
    print(f"Character vocabulary size: {vocab.num_char_types()}")
    print(f"Label vocabulary size: {vocab.num_label_types()}")
    
    model = Parser(
        vocab_size=vocab.num_char_types(),
        embedding_dim=100,
        char_lstm_hidden=50,
        span_lstm_hidden=200,
        num_categories=vocab.num_label_types(),
        hidden_layer_size=100,
        dropout=0.3
    )
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_errors = float('inf')
    error_history = []
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Shuffle training data
        random.shuffle(train_data)
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for words, gold_constituents in train_data:
            if len(words) == 0:
                continue
            
            optimizer.zero_grad()
            
            # Words to character tensors
            prefix_tensor, suffix_tensor = vocab.words_to_char_id_tensors(words)
            prefix_tensor = prefix_tensor.to(device)
            suffix_tensor = suffix_tensor.to(device)
            
            # Forward pass
            category_scores, _, span_positions = model(suffix_tensor, prefix_tensor)
            
            # Create gold label vector
            N = len(words)
            num_spans = len(span_positions)
            label_ids = torch.full((num_spans,), NO_CONST_LABEL_ID, dtype=torch.long, device=device)
            
            # Create span_id mapping
            span_id = {}
            for idx, (s, e) in enumerate(span_positions):
                span_id[(s, e)] = idx
            
            # Set gold labels
            for label, start, end in gold_constituents:
                if (start, end) in span_id:
                    label_id = vocab.label_to_id.get(label, NO_CONST_LABEL_ID)
                    label_ids[span_id[(start, end)]] = label_id
            
            # Compute loss
            loss = criterion(category_scores, label_ids)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Average loss: {avg_loss:.4f}")
        
        # Evaluate on development data
        print("Evaluating on development data...")
        errors = compute_errors(model, vocab, dev_data, device)
        error_history.append(errors)
        print(f"Number of errors: {errors}")
        
        # Save best model
        if errors < best_errors:
            best_errors = errors
            print(f"New best model! Saving to {output_base}.pt")
            torch.save(model.state_dict(), f"{output_base}.pt")
            vocab.store_parameters(f"{output_base}.io")
    
    print(f"\nTraining completed!")
    print(f"Best number of errors: {best_errors}")
    
    # Save error history
    with open('num-errors.txt', 'w') as f:
        for epoch, errors in enumerate(error_history, 1):
            f.write(f"Epoch {epoch}: {errors}\n")
    
    print("Error history saved to num-errors.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a constituency parser')
    parser.add_argument('train_file', help='Path to training data file')
    parser.add_argument('dev_file', help='Path to development data file')
    parser.add_argument('output_base', help='Base name for output files')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    
    args = parser.parse_args()
    
    train(args.train_file, args.dev_file, args.output_base, args.num_epochs)
