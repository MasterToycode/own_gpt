import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

#------------------------------- 单头自注意力 -------------------------------------------------#
class SelfAttention(nn.Module):
    def __init__(self, d_model, dropout_rate=0.2):
        super().__init__()
        self.d_model = d_model
        self.keys = nn.Linear(d_model, d_model, bias=False)
        self.queries = nn.Linear(d_model, d_model, bias=False)
        self.values = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X):
        B, T, C = X.shape
        K = self.keys(X)        # (B, T, C)
        Q = self.queries(X)     # (B, T, C)
        V = self.values(X)      # (B, T, C)

        # 缩放点积
        scaled_dot = (Q @ K.transpose(-2, -1)) / math.sqrt(C)  # (B, T, T)

        # 动态 mask：只保留下三角
        mask = torch.tril(torch.ones(T, T, device=X.device)).unsqueeze(0)  # (1, T, T)
        scaled_dot = scaled_dot.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scaled_dot, dim=-1)
        attn = self.dropout(attn)
        out = attn @ V
        return out

#------------------------------- 多头注意力 -------------------------------------------------#
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout_rate=0.2):
        super().__init__()
        self.h = h
        self.d_k = d_model // h
        self.heads = nn.ModuleList([SelfAttention(self.d_k, dropout_rate) for _ in range(h)])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X):
        B, T, C = X.shape
        # 分头
        X_split = X.view(B, T, self.h, self.d_k).transpose(1, 2)  # (B, h, T, d_k)
        out_heads = []
        for i, head in enumerate(self.heads):
            out_heads.append(head(X_split[:, i]))  # (B, T, d_k)
        # 拼接回原始维度
        out = torch.cat(out_heads, dim=-1)  # (B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out

#------------------------------- 前馈网络 -------------------------------------------------#
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout_rate)
        )
    def forward(self, X):
        return self.net(X)

#------------------------------- Transformer 块 -------------------------------------------------#
class Block(nn.Module):
    def __init__(self, d_model, h, dropout_rate=0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, h, dropout_rate)
        self.ff = FeedForward(d_model, dropout_rate)

    def forward(self, X):
        X = X + self.attn(self.ln1(X))
        X = X + self.ff(self.ln2(X))
        return X

#------------------------------- BigramLM -------------------------------------------------#
class BigramLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, max_seq_len=1024, h=8, Nx=6, dropout_rate=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # learnable positional embedding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout_rate)

        # Transformer blocks
        blocks = [Block(d_model, h, dropout_rate) for _ in range(Nx)]
        blocks.append(nn.LayerNorm(d_model))
        self.blocks = nn.Sequential(*blocks)

        # output layer
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)               # (B, T, C)
        pos_idx = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        pos_emb = self.pos_embedding(pos_idx)
        X = self.dropout(tok_emb + pos_emb)

        X = self.blocks(X)
        logits = self.lm_head(X)

        loss = None
        if targets is not None:
            # Ignore padding tokens (ID 1) in loss calculation
            loss_mask = (targets != 1).view(-1)
            if loss_mask.sum() > 0:  # Only compute loss if there are non-padding tokens
                loss = F.cross_entropy(logits.view(B*T, -1), targets.view(-1), ignore_index=1)
            else:
                loss = torch.tensor(0.0, device=idx.device)
        return logits, loss

    # ---------------- 生成函数 ----------------#
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None, eos_token_id=None, repetition_penalty=1.2):
        B = idx.size(0)
        generated = idx.clone()

        if eos_token_id is None:
            sp = spm.SentencePieceProcessor()
            sp.load("tokenizer.model")
            eos_token_id = sp.piece_to_id("<END>")

        finished = torch.zeros(B, dtype=torch.bool, device=generated.device)

        for _ in range(max_new_tokens):
            logits, _ = self(generated)
            logits = logits[:, -1, :]

            # 重复惩罚
            if repetition_penalty != 1.0:
                for b in range(B):
                    for t in torch.unique(generated[b]):
                        if logits[b, t] < 0:
                            logits[b, t] *= repetition_penalty
                        else:
                            logits[b, t] /= repetition_penalty

            # temperature
            logits = logits / temperature

            # top-k
            if top_k is not None:
                topv, topi = torch.topk(logits, top_k, dim=-1)
                mask = torch.full_like(logits, float('-inf'))
                mask.scatter_(1, topi, logits.gather(1, topi))
                logits = mask

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            finished |= (idx_next.squeeze(1) == eos_token_id)
            idx_next[finished, :] = eos_token_id
            generated = torch.cat((generated, idx_next), dim=1)

            if finished.all():
                break

        return generated
