import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    """
    Simple bi‑LSTM encoder:
      - Embedding → BiLSTM → final hidden state → Linear → tanh
    Returns a fixed‐length vector of size `hidden_dim`.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int  = 300,
        hidden_dim:    int  = 256,
        num_layers:    int  = 2,
        dropout:       float= 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )  # :contentReference[oaicite:1]{index=1}
        self.fc = nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (B, L)
        emb = self.embedding(input_ids)       # (B, L, E)
        out, (h_n, c_n) = self.lstm(emb)      # h_n: (2*num_layers, B, H)
        # take last layer's forward + backward hidden states
        h_fwd = h_n[-2]  # (B, H)
        h_bwd = h_n[-1]  # (B, H)
        h = torch.cat([h_fwd, h_bwd], dim=1)  # (B, 2H)
        return torch.tanh(self.fc(h))         # (B, H)
