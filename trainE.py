from KBLoader import KBData
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import torch
from Embeddings import Embeddings
from collections import defaultdict

class trainE:


    def get_samples(self, s_r_matrix, sr_key, idx, entities, batch_size=128, cuda = False):
        batch_samples = sr_key[idx:idx + batch_size]
        targets = torch.zeros([len(batch_samples), len(entities)], dtype=torch.float32)
        if cuda:
            targets = targets.cuda()
        for id, val in enumerate(batch_samples):
            targets[id, s_r_matrix[val]] = 1.
        return np.array(batch_samples), targets


    def model_validate(self, model,test_data_idxs,  s_r_matrix , cuda):
        model.eval()
        device = torch.device("cuda" if cuda else "cpu")
        test_data = np.array(test_data_idxs[0:len(test_data_idxs)])

        entity_id = torch.tensor(test_data[:, 0]).to(device)
        relation_id= torch.tensor(test_data[:, 1]).to(device)
        target_id= torch.tensor(test_data[:, 2]).to(device)
        predictions = model.forward(entity_id, relation_id)

        for j in range(test_data.shape[0]):
            t=predictions[j, target_id[j]].item()
            predictions[j, s_r_matrix[(test_data[j][0], test_data[j][1])]] = 0.0
            predictions[j, target_id[j]] = t

        _, sort_pred_idxs = torch.sort(predictions, dim=1, descending=True)
        sort_pred_idxs = sort_pred_idxs.cpu().numpy()
        hits = []
        for j in range(test_data.shape[0]):
            rank = np.where(sort_pred_idxs[j] == target_id[j].item())[0][0]
            if rank <= 10:
                hits.append(1.0)
            else:
                hits.append(0.0)

        hits10 = np.mean(hits)
        return hits10

        
       
    def train(self, kbPath, kbSplit,  datasetName, modelPath, cuda=False, learning_rate=0.0005, decay_rate=1.0, num_iterations = 500, batch_size =512):
        data = KBData(modelPath, datasetName, data_path=kbPath, data_split=kbSplit)
        Emb = Embeddings(data, cuda)
        Emb.init()
        print("Embedding Learning Starts...")
        print(Emb)
        if cuda:
            Emb.cuda()
        opt = torch.optim.Adam(Emb.parameters(), lr=learning_rate)
        scheduler = ExponentialLR(opt, decay_rate)

        s_r_matrix = defaultdict(list)
        for content in data.data_ids:
            s_r_matrix[(content[0], content[1])].append(content[2])

        bestV = 0
        for numit in range(1, num_iterations+1):
            Emb.train()
            lossM=0
            flagM=0
            np.random.shuffle(list(s_r_matrix.keys()))
            for j in tqdm(range(0, len(list(s_r_matrix.keys())), batch_size)):
                data_batch, targets = self.get_samples(s_r_matrix, list(s_r_matrix.keys()), j, data.entities, batch_size=batch_size, cuda=cuda)
                opt.zero_grad()
                entity= torch.tensor(data_batch[:,0])
                relation= torch.tensor(data_batch[:,1])
                if cuda:
                    entity= entity.cuda()
                    relation= relation.cuda()
                predictions = Emb.forward(entity, relation)
                loss = Emb.loss(predictions, targets)
                loss.backward()
                opt.step()
                lossM=lossM+loss.item()
                flagM=flagM+1
            if decay_rate:
                scheduler.step()

            print('Epoch', numit, ' Loss:', lossM/flagM)
            Emb.eval()
            with torch.no_grad():
                print("Validation Phase Begins:")
                valid = self.model_validate(Emb, data.data_ids, s_r_matrix, cuda)
                if valid >= bestV:
                    bestV= valid
                    print("Validation performance increased..Saving Model")
                    E_numpy = Emb.entity_embed.weight.data.cpu().numpy()
                    np.save(modelPath + 'E.npy', E_numpy)
                print('Best Perfrormance:', bestV)







# kbPath = "./Data/MetaQA/kb.txt"
# kbSplit = "|"
# datasetName = "MetaQA"
# modelPath = "./Models/MetaQA/model/"
# cuda = True
# learning_rate = 0.0005
# decay_rate = 1.0
# num_iterations = 500
# batch_size =256
# trainE=trainE()

# trainE.train(kbPath, kbSplit,  datasetName, modelPath, cuda, learning_rate, decay_rate, num_iterations, batch_size)



# kbPath = "./Data/PQ/full_kb.txt"
# kbSplit = "\t"
# datasetName = "PQ"
# modelPath = "./Models/PQ/model/"
# cuda = True
# learning_rate = 0.0005
# decay_rate = 1.0
# num_iterations = 500
# batch_size =256
# trainE=trainE()

# trainE.train(kbPath, kbSplit,  datasetName, modelPath, cuda, learning_rate, decay_rate, num_iterations, batch_size)


# kbPath = "./Data/PQ-3/full_kb.txt"
# kbSplit = "\t"
# datasetName = "PQ-3"
# modelPath = "./Models/PQ-3/model/"
# cuda = True
# learning_rate = 0.0005
# decay_rate = 1.0
# num_iterations = 500
# batch_size =256
# trainE=trainE()

# trainE.train(kbPath, kbSplit,  datasetName, modelPath, cuda, learning_rate, decay_rate, num_iterations, batch_size)


