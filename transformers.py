import numpy as np
import torch
from torch import nn
from einops import rearrange


def model_factory(config, data):
    if config['data_dir'][9:13] == 'Segm':
        train_data = data['train_data']
        label = data['train_label']
        feat_dim = train_data.shape[2], data['max_len']  # dimensionality of data features
        num_labels = label.max()+1
        model = ConvTran(feat_dim, config['d_model'], config['num_heads'], config['dim_ff'], num_classes=num_labels)
    else:
        train_data = data['train_data']
        label = data['train_label']
        feat_dim = train_data.iloc[0].size, data['max_len']  # dimensionality of data features
        num_labels = label.max()[0]+1
        model = ConvTran(feat_dim, config['d_model'], config['num_heads'], config['dim_ff'], num_classes=num_labels)
    return model


def conv_dx1_bn(inp, oup, kernel_size, image_size, downsample=False):
    stride = 1 if downsample == False else [1, 2]
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding='same'),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


def conv_Mxd(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        # nn.Conv2d(inp, oup, kernel_size=[image_size[0], 1], stride=stride, padding='valid', groups=inp),
        nn.Conv2d(inp, oup, kernel_size=[image_size[0], 1], stride=stride, padding='valid'),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )

# From https://github.com/chinhsuanwu/coatnet-pytorch


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        '''
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )'''

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = int(dim_head * heads)
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.1):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.norm1 = nn.BatchNorm1d(self.iw, eps=1e-5)
        self.ff = FeedForward(oup, hidden_dim, dropout)
        # self.norm2 = nn.BatchNorm1d(self.iw, eps=1e-5)

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = self.norm1(x)
        x = x + self.ff(x)
        return x


class ConvTran(nn.Module):
    def __init__(self, feat_dim, d_model, num_heads, dim_ff, num_classes):
        super().__init__()
        # Convolutions
        t = 8
        kernel_size = [1, t]
        ih, iw = feat_dim
        M = 64
        self.s0 = conv_dx1_bn(1, M, kernel_size, feat_dim, downsample=False)
        self.s1 = conv_Mxd(M, M, feat_dim, downsample=False)

        self.s2 = conv_dx1_bn(1, M, kernel_size, feat_dim, downsample=False)
        self.s3 = conv_Mxd(M, M, [M, M], downsample=False)

        # Transformers
        # len = int(np.ceil((iw-(t-1))/2)-(t-1))
        # len = int(np.ceil((iw - (t - 1)) / 2))

        self.s4 = Transformer(M, M, [1, iw], num_heads, dim_head=d_model/num_heads, dropout=0.1)
        # self.s5 = Transformer(iw, iw, [1, M], num_heads, dim_head=d_model / num_heads, dropout=0.1)
        self.pool = nn.MaxPool2d([5, 1])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        # flat = int(np.floor(iw/5)*np.floor(M))
        self.fc = nn.Linear(M, num_classes, bias=False)

    def forward(self, x):
        x = self.s0(x)
        # x = x.permute(0, 2, 1, 3)
        x = self.s1(x)

        x = x.permute(0, 2, 1, 3)
        x = self.s2(x)
        x = self.s3(x)

        x = np.squeeze(x, axis=2)
        x = x.permute(0, 2, 1)
        x = self.s4(x)
        # x = x.permute(0, 2, 1)
        # x = self.s5(x)
        # x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x