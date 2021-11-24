import torch
import numpy as np
from tqdm import tqdm
from QuestionEmbedder_roberta import QuestionEmbedder
from torch.optim.lr_scheduler import ExponentialLR
import pickle


def read_file(text_file):
    data_file = open(text_file, 'r')
    data_array = []
    word_id = {}
    for data_line in data_file.readlines():
        data_line = data_line.strip().split('\t')
        question = data_line[0].split('[')[0]+'NE'+data_line[0].split('[')[1].split(']')[1]
        data_array.append([data_line[0].split('[')[1].split(']')[0].strip(), question.strip(), data_line[1].split('|')])
        for word in question.strip().split():
            if word not in word_id :
                word_id[word] = len(word_id)+1
    return data_array, word_id



def testing(data_path, device, model,entity_id, id_word):
    model.eval()
    data, _= read_file(data_path)
    answers = []
    total_correct = 0
    total5_correct = 0
    total10_correct = 0
    Recall3 = 0
    Recall5 = 0
    Recall10 = 0
    Recall1 = 0
    Pr1 = 0
    Pr3 = 0
    Pr5 = 0
    Pr10 = 0
    inv_map = {v: k for k, v in entity_id.items()}

    for i in tqdm(range(len(data))):
        data_sample = data[i]
        word_q = [word_id[word.strip()] for word in data_sample[1].strip().split(' ')]
        if type(data_sample[2]) is str:
            target = entity_id[data_sample[2]]
        else:
            target = [entity_id[entity.strip()] for entity in list(data_sample[2])]

        q_h = torch.tensor(entity_id[data_sample[0].strip()], dtype=torch.long).to(device)
        question = torch.tensor(word_q, dtype=torch.long).to(device)
        ques_len = torch.tensor(len(word_q), dtype=torch.long).unsqueeze(0)
        top_10 = model.get_results(q_h, question, ques_len, id_word)
        top_10_idx = top_10[1].tolist()[0]
        if top_10_idx[0] == q_h.tolist():
            pred_ans = top_10_idx[1]
            new_list = top_10_idx[1:11]
        else:
            pred_ans = top_10_idx[0]
            new_list = top_10_idx[0:10]

        if type(target) is int:
            target = [target]
        is_correct = 0
        if pred_ans in target:
            total_correct += 1
            is_correct = 1
        for lp in range(0, 5):
            if new_list[lp] in target:
                total5_correct += 1
                break
        for lp in range(0, 10):
            if new_list[lp] in target:
                total10_correct += 1
                break

        q_text = data_sample[1]

        answers.append(q_text + '\t' + inv_map[pred_ans] + '\t' + str(is_correct))

        total_num_relevant_items = len(target)

        correctans3 = 0
        for lp in range(0, 3):
            if new_list[lp] in target:
                correctans3 += 1
        Recall3 = Recall3 + correctans3 / total_num_relevant_items
        Pr3 = Pr3 + correctans3 / 3

        correctans5 = 0
        for lp in range(0, 5):
            if new_list[lp] in target:
                correctans5 += 1
        Recall5 = Recall5 + correctans5 / total_num_relevant_items
        Pr5 = Pr5 + correctans5 / 5

        correctans10 = 0
        for lp in range(0, 10):
            if new_list[lp] in target:
                correctans10 += 1
        Recall10 = Recall10 + correctans10 / total_num_relevant_items
        Pr10 = Pr10 + correctans10 / 10

        correctans1 = 0
        for lp in range(0, 1):
            if new_list[lp] in target:
                correctans1 += 1
        Recall1 = Recall1 + correctans1 / total_num_relevant_items
        Pr1 = Pr1 + correctans1 / 1

    accuracy = total_correct / len(data)
    hit5 = total5_correct / len(data)
    hit10 = total10_correct / len(data)
    Recall3 = Recall3 / len(data)
    Recall5 = Recall5 / len(data)
    Recall10 = Recall10 / len(data)
    Recall1 = Recall1 / len(data)
    Pr1 = Pr1 / len(data)
    Pr3 = Pr3 / len(data)
    Pr5 = Pr5 / len(data)
    Pr10 = Pr10 / len(data)

    return answers, accuracy, hit5, hit10, len(data), Recall1, Recall3, Recall5, Recall10, Pr1, Pr3, Pr5, Pr10



print("Testing QE Starts")

#Parameter/Paths Initialization


datasetName = "PQ-2"
embedding_folder ="./Models/"+datasetName+"/KGEmbed/"
test_data_path = "./Data/PQ-2/test_q.txt"
model_store_path = './Models/'+datasetName+'/checkpoints/'
r_emb_p=pickle.load((open("./RobertaEmb/"+datasetName+"/wordEmbed.pkl", "rb")))

entity_path = embedding_folder + 'E.npy'
entity_annot = embedding_folder + 'entities.dict'
use_cuda =False
device = torch.device("cuda" if use_cuda else "cpu")


#Reading Embedding file generating from Step 1 of Entity Embeeding generation
i = 0
entities = np.load(entity_path)
entity_id = {}
entity_embedding = []
file_annot = open(entity_annot, 'r')
for line in file_annot:
    line = line.strip().split('\t')
    entity_id[line[1].strip()] = i
    entity_embedding.append(entities[int(line[0])])
    i += 1
file_annot.close()



data =r_emb_p["da"]
word_id=r_emb_p["wd"]
id_word=r_emb_p["iw"]

model = QuestionEmbedder(len(word_id), entity_embedding, device)
model.to(device)

checkpoint_file_name = model_store_path +  "model.pt"
model.load_state_dict(torch.load(checkpoint_file_name))

answers, accuracy, hit5, hit10, lend, Recall1, Recall3, Recall5, Recall10, Pr1, Pr3, Pr5, Pr10= testing(test_data_path, device, model, entity_id, id_word)

print("--------Answers:----------")
print(answers)
print("Total Test Samples: " + str(lend))

print(".............Hits.............")
print("Hits@1: "+str(accuracy))
print("Hits@5: "+str(hit5))
print("Hits@10: "+str(hit10))

print(".............Recall.............")
print("Recall@1: "+str(Recall1))
print("Recall@5: "+str(Recall5))
print("Recall@10: "+str(Recall10))

print(".............Precision............")
print("Precision@1: "+str(Pr1))
print("Precision@5: "+str(Pr5))
print("Precision@10: "+str(Pr10))
           









