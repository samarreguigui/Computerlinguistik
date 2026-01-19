import torch
import torch.nn as nn

class CharacterEncoder(nn.Module):
    """
    Encodes a word from its character sequence.
    """
    
    def __init__(self, vocab_size, embedding_dim=100, lstm_hidden=50, dropout=0.3):
        """
        Args:
            vocab_size: Size of character vocabulary
            embedding_dim: Dimension of character embeddings
            lstm_hidden: Hidden size for character LSTMs
            dropout: Dropout probability for regularization (default: 0.3, automatic)
        """
        super().__init__()
        
        self.char_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.forward_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden,
            bidirectional=False,
            batch_first=True
        )
        
        self.backward_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden,
            bidirectional=False,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, suffix_ids, prefix_ids):
        """
        Encode word representations from suffix and prefix character sequences.
        """
        suffix_embedded = self.char_embedding(suffix_ids)
        prefix_embedded = self.char_embedding(prefix_ids)
        
        suffix_embedded = self.dropout(suffix_embedded)
        prefix_embedded = self.dropout(prefix_embedded)
        
        # Process suffix with forward LSTM
        forward_hidden_state, _ = self.forward_lstm(suffix_embedded)
        forward_repr = forward_hidden_state[:, -1]
        
        # Process prefix with backward LSTM
        backward_hidden_state, _ = self.backward_lstm(prefix_embedded)
        backward_repr = backward_hidden_state[:, -1]
        
        word_repr = torch.cat([forward_repr, backward_repr], dim=1)
        word_repr = self.dropout(word_repr)
        
        return word_repr


class SpanEncoder(nn.Module):
    """
    Encodes spans using contextual BiLSTM and difference vectors.
    """
    
    def __init__(self, word_repr_dim, lstm_hidden=200, dropout=0.3):
        """
        Args:
            word_repr_dim: Dimension of word representations (input)
            lstm_hidden: Hidden size for span BiLSTM
            dropout: Dropout probability (default: 0.3)
        """
        super().__init__()
        
        self.lstm_hidden = lstm_hidden
        
        self.bilstm = nn.LSTM(
            input_size=word_repr_dim,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, word_repr):
        """
        Compute representations for all spans in the sentence.
        """
        N = word_repr.size(0) # Number of words in the sentence
        
        # Add dummy zero vectors at positions 0 and N+1
        padded_word_repr = torch.nn.functional.pad(word_repr, (0, 0, 1, 1))
        # Add batch dimension: (N+2, word_repr_dim) -> (1, N+2, word_repr_dim)
        padded_word_repr = padded_word_repr.unsqueeze(0)
        
        bilstm_seq_output, _ = self.bilstm(padded_word_repr)
        # Remove batch dimension: (1, N+2, 2*lstm_hidden) -> (N+2, 2*lstm_hidden)
        bilstm_output = bilstm_seq_output.squeeze(0)
        
        bilstm_output = self.dropout(bilstm_output)
        
        # Split forward and backward representations
        forward, backward = bilstm_output.chunk(2, dim=-1)
        
        # Remove boundary elements
        forward = forward[:-1]
        backward = backward[1:]
        
        indices = torch.arange(N + 1, device=word_repr.device)
        span_indices = torch.combinations(indices, r=2)
        start_indices = span_indices[:, 0]
        end_indices = span_indices[:, 1]
        span_positions = [(int(s), int(e)) for s, e in span_indices.tolist()]
        
        # Difference vectors
        forward_diffs = forward[end_indices] - forward[start_indices]
        backward_diffs = backward[start_indices] - backward[end_indices]

        # Concatenate forward and backward differences
        span_reprs = torch.cat([forward_diffs, backward_diffs], dim=1)
        
        span_reprs = self.dropout(span_reprs)
        
        return span_reprs, span_positions


class SpanClassifier(nn.Module):
    """
    Classifies spans into syntactic categories.
    """
    
    def __init__(self, span_repr_dim, num_categories, hidden_layer_size=100, dropout=0.3):
        """
        Args:
            span_repr_dim: Dimension of span representations (input)
            num_categories: Number of output categories
            hidden_layer_size: Size of hidden layer
            dropout: Dropout probability between layers (default: 0.3, automatic)
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(span_repr_dim, hidden_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer_size, num_categories)
        )
    
    def forward(self, span_reprs):
        scores = self.mlp(span_reprs)
        # Constraint: Label 0 ("not a constituent") always scores 0
        scores[:, 0] = 0
        
        return scores


class Parser(nn.Module):
    """    
    Architecture:
    1. CharacterEncoder: Character IDs to word representations
    2. SpanEncoder: Word representations to span representations
    3. SpanClassifier: Span representations to category scores
    """
    
    def __init__(self, vocab_size, embedding_dim=100, char_lstm_hidden=50, 
                 span_lstm_hidden=200, num_categories=50, hidden_layer_size=100, dropout=0.3):
        """
        Args:
            vocab_size: Size of character vocabulary
            embedding_dim: Dimension of character embeddings
            char_lstm_hidden: Hidden size for character LSTMs
            span_lstm_hidden: Hidden size for span BiLSTM
            num_categories: Number of syntactic categories
            hidden_layer_size: Size of hidden layer in classifier
            dropout: Dropout probability
        """
        super().__init__()
        
        # 1: Convert characters to word representations
        self.char_encoder = CharacterEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            lstm_hidden=char_lstm_hidden,
            dropout=dropout
        )
        
        # 2: Convert word representations to span representations
        self.span_encoder = SpanEncoder(
            word_repr_dim=2 * char_lstm_hidden,
            lstm_hidden=span_lstm_hidden,
            dropout=dropout
        )
        # 3: Classify spans into categories
        self.span_classifier = SpanClassifier(
            span_repr_dim= 2 * span_lstm_hidden,
            num_categories=num_categories,
            hidden_layer_size=hidden_layer_size,
            dropout=dropout
        )
    
    def forward(self, suffix_ids, prefix_ids):
        """
        Forward pass through the parser.
        """
        word_repr = self.char_encoder(suffix_ids, prefix_ids)
        span_reprs, span_positions = self.span_encoder(word_repr)
        category_scores = self.span_classifier(span_reprs)
        
        return category_scores, span_reprs, span_positions