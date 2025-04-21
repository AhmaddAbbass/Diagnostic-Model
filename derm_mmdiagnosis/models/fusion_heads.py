import torch
import torch.nn as nn

class EarlyConcat(nn.Module):
    def __init__(self, dim_i, dim_t, hidden=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_i + dim_t, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden)
        )
    def forward(self, vi, vt):
        return self.mlp(torch.cat([vi, vt], dim=1))

class GMU(nn.Module):
    """
    Gated Multimodal Unit:
      h_i = tanh(W_i v_i + b_i)
      h_t = tanh(W_t v_t + b_t)
      z   = sigmoid(W_z [v_i; v_t] + b_z)
      out = z * h_i + (1-z) * h_t
    """
    def __init__(self, dim_i, dim_t, hidden=None):
        super().__init__()
        h = hidden or max(dim_i, dim_t)
        self.Wi = nn.Linear(dim_i, h)
        self.Wt = nn.Linear(dim_t, h)
        self.Wz = nn.Linear(dim_i + dim_t, h)
    def forward(self, vi, vt):
        hi = torch.tanh(self.Wi(vi))
        ht = torch.tanh(self.Wt(vt))
        z  = torch.sigmoid(self.Wz(torch.cat([vi, vt], dim=1)))
        return z * hi + (1 - z) * ht  # :contentReference[oaicite:3]{index=3}

class CrossAttention(nn.Module):
    """
    Two‑stream cross‑attention:
      img2txt = Attn(query=vi, key=vt, value=vt)
      txt2img = Attn(query=vt, key=vi, value=vi)
      concat & MLP
    """
    def __init__(self, dim_i, dim_t, hidden=512, heads=8):
        super().__init__()
        self.attn_i2t = nn.MultiheadAttention(dim_i, heads, batch_first=True)
        self.attn_t2i = nn.MultiheadAttention(dim_t, heads, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(dim_i + dim_t, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden)
        )
    def forward(self, vi, vt):
        # add sequence dim
        vi_seq = vi.unsqueeze(1)  # (B,1,dim_i)
        vt_seq = vt.unsqueeze(1)  # (B,1,dim_t)
        # attend
        i2t, _ = self.attn_i2t(query=vi_seq, key=vt_seq, value=vt_seq)
        t2i, _ = self.attn_t2i(query=vt_seq, key=vi_seq, value=vi_seq)
        # remove seq dim & concat
        feat = torch.cat([i2t.squeeze(1), t2i.squeeze(1)], dim=1)
        return self.proj(feat)    # :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

class Bilinear(nn.Module):
    """
    Simple bilinear pooling (outer product → flatten → MLP).
    """
    def __init__(self, dim_i, dim_t, hidden=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_i * dim_t, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden)
        )
    def forward(self, vi, vt):
        outer = torch.bmm(vi.unsqueeze(2), vt.unsqueeze(1))  # (B, dim_i, dim_t)
        flat  = outer.view(vi.size(0), -1)                   # (B, dim_i*dim_t)
        return self.fc(flat)
