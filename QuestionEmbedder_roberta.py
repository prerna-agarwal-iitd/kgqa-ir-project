import torch
import torch.nn as nn
import torch.nn.utils
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torch.nn.init import xavier_normal_


class QuestionEmbedder(nn.Module):

    def __init__(self, vocab_size, pretrainedE, device):
        super(QuestionEmbedder, self).__init__()
        self.device = device
        self.getScores = self.ComplEx
        self.LSTM1 = nn.LSTM(384, 256, 1, bidirectional=True,batch_first=True)
        self.loss = torch.nn.BCELoss(reduction='sum')
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrainedE), freeze=False)
        self.linearLayer1 = nn.Linear(256* 2, 256, bias=False)
        self.linearLayer2 = nn.Linear(256, 256, bias=False)
        xavier_normal_(self.linearLayer1.weight.data)
        xavier_normal_(self.linearLayer2.weight.data)
        self.hiddenLinear = nn.Linear(256, 400)
        self.drop_question = torch.nn.Dropout(0.1)
        self.drop_q_head = torch.nn.Dropout(0.2)
        self.drop_score = torch.nn.Dropout(0.2)



    def ComplEx(self, head, relation):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        head = self.drop_q_head(head)
        relation = self.drop_question(relation)

        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.embedding.weight, 2, dim=1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        score = self.drop_score(score)
        score = score.permute(1, 0, 2)

        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1, 0)) + torch.mm(im_score, im_tail.transpose(1, 0))
        pred = torch.sigmoid(score)
        return pred



    def forward(self, q, q_h,  q_t, q_len, id_word, batch_size):

       
        x1 = torch.zeros(batch_size, q_len.tolist()[0], 384, dtype=torch.float)
        i=0
        for ques in  q:
            data=ques.tolist()
            j=0
            for word_id in data:
                if word_id==0:
                    x1[i][j]=torch.zeros(384, dtype=torch.float)
                    j=j+1
                else:
                    x1[i][j]=id_word[word_id]
                    j=j+1
            i=i+1


        packed_output = pack_padded_sequence(x1.to(self.device), q_len, batch_first=True)
        outputs, (hidden, cell_state) = self.LSTM1(packed_output)
        outputs = torch.cat([hidden[0, :, :], hidden[1, :, :]], dim=-1)
        q_embedding  = self.hiddenLinear(F.relu(self.linearLayer2(F.relu(self.linearLayer1(outputs)))))
        score = self.getScores(self.embedding(q_h), q_embedding)
        loss = self.loss(score, q_t)
        return loss



    def get_results(self, q_h, q, q_len, id_word):

       
        x1 = torch.zeros(1, q_len.tolist()[0], 384, dtype=torch.float)
        i = 0
        data = q.tolist()
        j = 0
        for word_id in data:
            if word_id == 0:
                x1[i][j] = torch.zeros(384, dtype=torch.float)
                j = j + 1
            else:
                x1[i][j] = id_word[word_id]
                j = j + 1

       
        packed_output = pack_padded_sequence(x1.to(self.device),  q_len, batch_first=True)
        outputs, (hidden, cell_state) = self.LSTM1(packed_output)
        outputs = torch.cat([hidden[0, :, :], hidden[1, :, :]], dim=-1)
        q_embedding  = self.hiddenLinear(F.relu(self.linearLayer2(F.relu(self.linearLayer1(outputs)))))
        score = self.getScores(self.embedding(q_h).unsqueeze(0), q_embedding)
        top10 = torch.topk(score, k=11, largest=True, sorted=True)
        return top10




