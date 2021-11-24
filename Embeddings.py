import torch
import torch.nn.functional as F
from torch.nn.init import xavier_normal_


class Embeddings(torch.nn.Module):

    def __init__(self, data, cuda):
        super(Embeddings, self).__init__()
        self.epsilon = 2.0
        self.entity_embed = torch.nn.Embedding(len(data.entities), 400, padding_idx=0)
        self.relation_embed = torch.nn.Embedding(len(data.relations), 400, padding_idx=0)
        self.loss = self.cLoss
        self.score_func = self.ComplEx
        self.cuda1=cuda
        


    def cLoss(self, pred, tr):
        if self.cuda1:
            pred = pred.cuda()
        pred = F.log_softmax(pred, dim=-1)
        return -torch.sum(pred * (tr / tr.size(-1)))

    def init(self):
        xavier_normal_(self.entity_embed.weight.data)
        xavier_normal_(self.relation_embed.weight.data)

    # standard implementation from ComplexE paper
    def ComplEx(self, head, relation):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]
        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.entity_embed.weight, 2, dim=1)
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = torch.stack([re_score, im_score], dim=1)
        score = score.permute(1, 0, 2)
        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1, 0)) + torch.mm(im_score, im_tail.transpose(1, 0))
        return score


    def forward(self, entity, relation):
        return torch.sigmoid(self.score_func(self.entity_embed(entity), self.relation_embed(relation)))