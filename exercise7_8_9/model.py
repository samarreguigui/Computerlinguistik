import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, lstm_size, dropout_rate):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # 3-layer bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=lstm_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, src_letter_ids, src_lengths):
        embs = self.dropout(self.embedding(src_letter_ids))

        # Pack the padded sequence
        packed = pack_padded_sequence(
            embs, 
            src_lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )

        # Run the BiLSTM
        packed_output, (hidden, cell) = self.lstm(packed)

        # Unpack the output
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Extract LAST LAYER backward hidden state: hidden[-1]
        backward_hidden_last = hidden[-1:].contiguous()
        backward_cell_last = cell[-1:].contiguous()

        return output, backward_hidden_last, backward_cell_last
    
class Attention(nn.Module):
    def __init__(self, lstm_size, dropout_rate):
        super().__init__()
        
        self.ff1 = nn.Linear(3 * lstm_size, lstm_size)
        self.ff2 = nn.Linear(lstm_size, 1)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, encoder_states, decoder_states):

        batch_size, enc_len, _ = encoder_states.size()
        _, dec_len, _ = decoder_states.size()

        # Expand decoder states to match encoder length
        dec_exp = decoder_states.unsqueeze(2).expand(batch_size, dec_len, enc_len, -1)

        # Expand encoder states to match decoder steps
        enc_exp = encoder_states.unsqueeze(1).expand(batch_size, dec_len, enc_len, -1)

        # Concatenate encoder+decoder states
        combined = torch.cat([enc_exp, dec_exp], dim=-1)

        # Compute attention scores
        scores = self.ff2(torch.tanh(self.ff1(self.dropout(combined)))).squeeze(-1)

        # Softmax over encoder positions
        attn = torch.softmax(scores, dim=2).unsqueeze(-1)

        context = torch.sum(attn * enc_exp, dim=2)

        return context

class Model(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_size, lstm_size, dropout_rate, pad_id):
        super().__init__()

        self.pad_id = pad_id

        # Encoder + Attention
        self.encoder = Encoder(src_vocab_size, embedding_size, lstm_size, dropout_rate)
        self.attention = Attention(lstm_size, dropout_rate)

        # Decoder Embedding 
        self.embedding = nn.Embedding(tgt_vocab_size, embedding_size)

        # Three unidirectional LSTMs for the decoder
        self.lstm1 = nn.LSTM(embedding_size, lstm_size, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_size + 2*lstm_size, lstm_size, batch_first=True)
        self.lstm3 = nn.LSTM(lstm_size + 2*lstm_size, lstm_size, batch_first=True)

        # Projection layer
        self.output_projection = nn.Linear(lstm_size, embedding_size)

        # Output layer: embedding → vocabulary logits
        self.output_layer = nn.Linear(embedding_size, tgt_vocab_size, bias=False)

        # Tied embeddings: input and output share the same weight matrix
        self.output_layer.weight = self.embedding.weight

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, src_letter_ids, src_lengths, tgt_letter_ids):

        enc_states, enc_hidden, enc_cell = self.encoder(src_letter_ids, src_lengths)

        batch_size = src_letter_ids.size(0)

        #Embed target characters
        embs = self.dropout(self.embedding(tgt_letter_ids))

        # LSTM1: initializes with encoder backward states
        dec_states1, states1 = self.lstm1(embs, (enc_hidden, enc_cell))
    
        #Attention for LSTM1 output
        context1 = self.attention(enc_states, dec_states1)

        lstm2_input = torch.cat([dec_states1, context1], dim=-1)

        # LSTM2
        dec_states2, states2 = self.lstm2(lstm2_input, (enc_hidden, enc_cell))

        # Attention for LSTM2 output
        context2 = self.attention(enc_states, dec_states2)
        lstm3_input = torch.cat([dec_states2, context2], dim=-1)

        # LSTM3
        dec_states3, states3 = self.lstm3(lstm3_input, (enc_hidden, enc_cell))

        # Projection to embedding size
        proj = self.output_projection(self.dropout(dec_states3))

        #Output layer to vocabulary logits
        logits = self.output_layer(proj)
    
        return logits
    
    def lemmatize(self, src_letter_ids, src_lengths, max_tgt_length):
        self.eval()

        #Encode
        enc_states, enc_hidden, enc_cell = self.encoder(src_letter_ids, src_lengths)

        batch_size = src_letter_ids.size(0)
        pad_id = self.pad_id
        # Start with PAD token for every batch element
        tgt_char_ids = torch.full((batch_size, 1), pad_id, dtype=torch.long, device=src_letter_ids.device)

        # Initialize previous states
        prev_states1 = (enc_hidden, enc_cell)
        prev_states2 = (enc_hidden, enc_cell)
        prev_states3 = (enc_hidden, enc_cell)

        all_tgt_char_ids = []

        for _ in range(max_tgt_length):

            #Embedding
            embs = self.dropout(self.embedding(tgt_char_ids))

            # LSTM1
            dec_states1, prev_states1 = self.lstm1(embs, prev_states1)

            # Attention
            context1 = self.attention(enc_states, dec_states1)

            # LSTM2
            lstm2_input = torch.cat([dec_states1, context1], dim=-1)
            dec_states2, prev_states2 = self.lstm2(lstm2_input, prev_states2)

            # Attention again
            context2 = self.attention(enc_states, dec_states2)

            # LSTM3
            lstm3_input = torch.cat([dec_states2, context2], dim=-1)
            dec_states3, prev_states3 = self.lstm3(lstm3_input, prev_states3)

            # Output
            proj = self.output_projection(dec_states3)
            logits = self.output_layer(proj)

            next_char = logits.argmax(dim=-1)[:, -1]  # last timestep
            all_tgt_char_ids.append(next_char)

            # Prepare next input
            tgt_char_ids = next_char.unsqueeze(1)

            # Stop if all sequences have at least one PAD symbol
            all_generated = torch.stack(all_tgt_char_ids, dim=1)
            if torch.all(torch.any(all_generated == pad_id, dim=1), dim=0):
                break

        return torch.stack(all_tgt_char_ids, dim=1)

    @torch.no_grad()
    def predict(self, src_letter_ids, src_lengths, max_tgt_length):
        # Encode input
        enc_states, enc_hidden, enc_cell = self.encoder(src_letter_ids, src_lengths)
        batch_size = src_letter_ids.size(0)

        #Prepare output storage
        all_tgt_char_ids = []

        # Start with PAD token for every batch element
        tgt_char_ids = torch.full((batch_size, 1), self.pad_id, dtype=torch.long, device=src_letter_ids.device)

        # Initial states 
        states1 = (enc_hidden, enc_cell)
        states2 = (enc_hidden, enc_cell)
        states3 = (enc_hidden, enc_cell)

        for t in range(max_tgt_length):

            #Embed last generated character
            embs = self.embedding(tgt_char_ids)  # (batch, 1, emb_size)

            # --- LSTM1 ---
            dec_states1, states1 = self.lstm1(embs, states1)

            # Attention → context
            context1 = self.attention(enc_states, dec_states1)

            # concat context with LSTM1 output
            in2 = torch.cat([dec_states1, context1], dim=-1)

            # --- LSTM2 ---
            dec_states2, states2 = self.lstm2(in2, states2)

            context2 = self.attention(enc_states, dec_states2)
            in3 = torch.cat([dec_states2, context2], dim=-1)

            # --- LSTM3 ---
            dec_states3, states3 = self.lstm3(in3, states3)

            # Projection + output
            proj = self.output_projection(dec_states3)
            logits = self.output_layer(proj)

            # Greeedy decoding: pick the argmax
            next_char = torch.argmax(logits, dim=-1)  
            tgt_char_ids = next_char  

            # Save generated character
            all_tgt_char_ids.append(next_char.squeeze(1))  

            # Stopp if ALL sequences output PAD in this step
            all_generated = torch.stack(all_tgt_char_ids, dim=1) 
            if torch.all(torch.any(all_generated == self.pad_id, dim=1), dim=0):
                break

        # Final output: shape (batch, decoded_length)
        return torch.stack(all_tgt_char_ids, dim=1)