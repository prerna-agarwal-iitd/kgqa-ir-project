import torch
import numpy as np
from tqdm import tqdm
from QuestionEmbedder_vocab import QuestionEmbedder
from torch.optim.lr_scheduler import ExponentialLR
from trainE import*
import os

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
                word_id[word] = len(word_id)
    return data_array, word_id




def model_validation(data_path, device, model,entity_id):
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
        top_10 = model.get_results(q_h, question, ques_len)
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
        answers.append(q_text + '\t' + str(pred_ans) + '\t' + str(is_correct))

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




datasetName = "MetaQA"
embedding_folder ="./Models/"+datasetName+"_Vocab"+"/KGEmbed/"
data_path = "./Data/MetaQA/1-hop/vanilla/qa_train.txt"
valid_data_path = "./Data/MetaQA/1-hop/vanilla/qa_dev.txt"
kbPath ="./Data/MetaQA/kb.txt"
model_store_path = './Models/'+datasetName+"_Vocab"+'/checkpoints1/'



if not os.path.exists("./Models/"+datasetName+"_Vocab"):
    os.makedirs("./Models/"+datasetName+"_Vocab")
if not os.path.exists(embedding_folder):
    os.makedirs(embedding_folder)
if not os.path.exists(model_store_path):
    os.makedirs(model_store_path)

use_cuda =False
trainKBEmbeddingFlag = True


if trainKBEmbeddingFlag:
    kbSplit = "|"
    cuda = use_cuda
    learning_rate = 0.0005
    decay_rate = 1.0
    num_iterations = 500
    batch_size =256
    print("Training KB Starts")
    trainE=trainE()
    trainE.train(kbPath, kbSplit,  datasetName, embedding_folder, cuda, learning_rate, decay_rate, num_iterations, batch_size)
    print("Training KB Ends")




print("Training QE Starts")

#Parameter/Paths Initialization


entity_path = embedding_folder + 'E.npy'
entity_annot = embedding_folder + 'entities.dict'
batch_size=1024
nb_epochs =1000
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

#Read Train Question File
data, word_id = read_file(data_path)



#Create Question Embedding Learing Model
model = QuestionEmbedder(len(word_id), entity_embedding, device)
model.to(device)

#Set Optimmizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = ExponentialLR(optimizer, 1.0)
optimizer.zero_grad()
bscore = -float("inf")

for epoch in range(0,nb_epochs):
    model.train()
    lossV = 0
    #create batches
    total_samples_idx=[i for i  in  range(0,len(data))]
    np.random.shuffle(total_samples_idx)
    flagM=0
    for j in tqdm(range(0, len(data), batch_size)):
        data_batch=[]
        for idx in total_samples_idx[j:j+batch_size]:
            #create one hot
            last_ids=[entity_id[name.strip()] for name in data[idx][2]]
            one_hot_vector = torch.FloatTensor(len(entity_id))
            one_hot_vector.zero_()
            one_hot_vector.scatter_(0, torch.LongTensor(last_ids), 1)
            data_batch.append(tuple([[word_id[word] for word in data[idx][1].split()], entity_id[data[idx][0].strip()], one_hot_vector]))

        #find longest question to create array of same  size
        sorted_data = sorted(data_batch, key=lambda sample: len(sample[0]), reverse=True)
        question_lengths = []
        question_h = []
        question_t = []
        new_sample_data = torch.zeros(len(data_batch), len(sorted_data[0][0]), dtype=torch.long)
        for val in range(len(data_batch)):
            sample = sorted_data[val][0]
            question_h.append(sorted_data[val][1])
            question_t.append(sorted_data[val][2])
            question_lengths.append(len(sample))
            lenS=len(sample)
            sample = torch.tensor(sample, dtype=torch.long)
            sample = sample.view(sample.shape[0])
            new_sample_data[val].narrow(0, 0, lenS).copy_(sample)

        #create tensor objects
        model.zero_grad()
        question = new_sample_data.to(device)
        ques_len = torch.tensor(question_lengths, dtype=torch.long).to(device)
        q_head = torch.tensor(question_h,dtype=torch.long).to(device)
        q_tail = torch.stack(question_t).to(device)

        #calculate loss by calling forward
        loss = model(question, q_head, q_tail, ques_len)
        loss.backward()
        optimizer.step()
        lossV+= loss.item()
        flagM = flagM + len(data_batch)
    print('Epoch', epoch, ' Loss:', lossV / flagM)
    scheduler.step()

    #Validate after 10 epochs and save model if performance on validation data increased
    if epoch%10 == 0:
        model.eval()
        answers, score, _, _, _, _, _, _, _, _, _, _, _ = model_validation(valid_data_path, device, model, entity_id)
        if score > bscore:
            bscore = score
            print("Validation accuracy increased to: ", score)
            print("Saving Model")
            torch.save(model.state_dict(), model_store_path +  "model.pt")








