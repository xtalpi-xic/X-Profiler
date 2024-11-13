import torch
from torch import nn
from torchvision.models import resnet18
from transformer import TransformerBlock


class MID(nn.Module):
    def __init__(self, cls_num, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.sub_model = resnet18(num_classes=self.emb_dim)
        self.cls_token = nn.Parameter(torch.randn((1, 1, self.emb_dim)) * 0.02, requires_grad=True)
        self.trm1 = TransformerBlock(self.emb_dim, self.emb_dim, 8)
        self.trm2 = TransformerBlock(self.emb_dim, self.emb_dim, 8)
        self.cls = nn.Sequential(nn.Linear(self.emb_dim, cls_num))

    def forward(self, p_img):
        n, p, c, h, w = p_img.shape
        p_seq = p_img.reshape(-1, c, h, w)
        p_seq = self.sub_model(p_seq)
        p_seq = p_seq.reshape(n, -1, self.emb_dim)
        p_seq = torch.cat([self.cls_token.repeat(n, 1, 1), p_seq], dim=1)
        p_seq, _ = self.trm1(p_seq)
        p_seq, _ = self.trm2(p_seq)
        img = self.cls(p_seq[:, 0, :])
        return img


if __name__ == '__main__':
   pass 
