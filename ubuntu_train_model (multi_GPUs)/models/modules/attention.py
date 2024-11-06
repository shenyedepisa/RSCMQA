import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.matmul(k.transpose(1, 2), q)
        scores_1 = scores.sum(dim=-1)
        scores_2 = scores_1 / torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32))
        att_mask = torch.mul(v, scores_2)
        return att_mask


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, y):
        q = self.query(x)
        k = self.key(y)
        v = self.value(y)

        scores = torch.matmul(k.transpose(1, 2), q)
        scores_1 = scores.sum(dim=-1)
        scores_2 = scores_1 / torch.sqrt(torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32)))
        att_mask = torch.mul(v, scores_2.unsqueeze(1))
        return att_mask


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# MT attention
class MTA(nn.Module):
    def __init__(self, setting):
        super().__init__()
        self.config = setting
        self.embed_dim = self.config['embed_size']
        self.num_heads = self.config['heads']
        self.input_dim = self.config['mlp_input']
        self.hidden_dim = self.config['mlp_input'] * self.config['mlp_ratio']
        self.output_dim = self.config['mlp_output']
        self.dropout = self.config['attn_dropout']
        self.selfAttention = SelfAttention(self.embed_dim, self.num_heads, self.dropout)
        self.crossAttention = CrossAttention(self.embed_dim, self.num_heads, self.dropout)
        self.MLP = MLP(self.input_dim, self.hidden_dim, self.output_dim, self.dropout)

    def forward(self, x, y):
        x = self.selfAttention(x)
        output = self.crossAttention(x, y)
        output = self.MLP(output)

        return output
