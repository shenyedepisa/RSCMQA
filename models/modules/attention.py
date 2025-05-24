import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, x):
        x = x.transpose(0, 1)  # nn.MultiheadAttention expects inputs in the shape (seq_len, batch, embed_dim)
        attn_output, _ = self.multihead_attn(x, x, x)
        return attn_output.transpose(0, 1)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, x, y):
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        attn_output, _ = self.multihead_attn(x, y, y)
        return attn_output.transpose(0, 1)


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
        self.hidden_dim = self.config['mlp_input']*self.config['mlp_ratio']
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

