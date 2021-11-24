class KBData:

    def __init__(self, dicPath, dataset, data_path="", data_split="|"):
        self.data_path = data_path
        if dataset =="WebQA":
             self.data, self.entities, self.relations, self.entity_ids, self.relation_ids, self.data_ids = self.data_loading_Web(data_path,data_split, dicPath, dataset)

        else:
            self.data, self.entities, self.relations, self.entity_ids, self.relation_ids, self.data_ids = self.data_loading(data_path,data_split, dicPath, dataset)

    def data_loading(self, data_path, data_split, dicPath, dataset):
        data=[]
        entities =[]
        relations =[]
        file = open(data_path, 'r')
        while True:
            line = file.readline()
            if not line:
                break
            split_data = line.strip().split(data_split)
            data.append(split_data)
            entities.append(split_data[0])
            entities.append(split_data[2])
            relations.append(split_data[1])
        file.close()
        entities = list(set(entities))
        relations = list(set(relations))
        entity_ids = {entities[i]: i for i in range(len(entities))}
        relation_ids = {relations[i]: i for i in range(len(relations))}

        data_ids = [(entity_ids[data[i][0]], relation_ids[data[i][1]], entity_ids[data[i][2]]) for i in range(len(data))]

       
        f = open(dicPath + 'entities.dict', 'w')
        for key, value in entity_ids.items():
            line = key + '\t' + str(value) + '\n'
            line = line.rstrip().split('\t')
            name = line[0]
            id = int(line[1])
            f.write(str(id) + '\t' + name + '\n')
        f.close()

        return data, entities, relations, entity_ids, relation_ids, data_ids


    def data_loading_Web(self, data_path, data_split, dicPath, dataset):
        data=[]
        entities =[]
        relations =[]
        qEntity=[]
        file = open(data_path + "qa_train_webqsp.txt", 'r')
        for line in file.readlines():
            # if not line:
            #     break
            line = line.strip().split('\t')
            if len(line)>1:
                qEntity.append((line[0].split('['))[1].split(']')[0].strip())
                val = line[1].split('|')
                for x in val:
                    qEntity.append(x)

        file = open(data_path + "qa_test_webqsp.txt", 'r')
        for line in file.readlines():
            # if not line:
            #     break
            line = line.strip().split('\t')
            if len(line) > 1:
                qEntity.append((line[0].split('['))[1].split(']')[0].strip())
                val = line[1].split('|')
                for x in val:
                    qEntity.append(x)
        qEntity = list(set(qEntity))

        file = open(data_path+"valid.txt", 'r')
        for line in file.readlines():
            # if not line:
            #     break
            split_data = line.strip().split(data_split)
            if((split_data[0] in qEntity and split_data[2] in qEntity)):
                data.append(split_data)
                entities.append(split_data[0])
                entities.append(split_data[2])
                relations.append(split_data[1])
            elif(split_data[0] in qEntity and split_data[0] not in entities ):
                data.append(split_data)
                entities.append(split_data[0])
                entities.append(split_data[2])
                relations.append(split_data[1])
            elif(split_data[1] in qEntity and split_data[1] not in entities ):
                data.append(split_data)
                entities.append(split_data[0])
                entities.append(split_data[2])
                relations.append(split_data[1])

        file = open(data_path + "train.txt", 'r')
        for line in file.readlines():
            # if not line:
            #     break
            split_data = line.strip().split(data_split)
            if ((split_data[0] in qEntity and split_data[2] in qEntity)):
                data.append(split_data)
                entities.append(split_data[0])
                entities.append(split_data[2])
                relations.append(split_data[1])
            elif(split_data[0] in qEntity and split_data[0] not in entities ):
                data.append(split_data)
                entities.append(split_data[0])
                entities.append(split_data[2])
                relations.append(split_data[1])
            elif(split_data[1] in qEntity and split_data[1] not in entities ):
                data.append(split_data)
                entities.append(split_data[0])
                entities.append(split_data[2])
                relations.append(split_data[1])



        file = open(data_path + "test.txt", 'r')
        for line in file.readlines():
            # if not line:
            #     break
            split_data = line.strip().split(data_split)
            if ((split_data[0] in qEntity and split_data[2] in qEntity)):
                data.append(split_data)
                entities.append(split_data[0])
                entities.append(split_data[2])
                relations.append(split_data[1])
            elif(split_data[0] in qEntity and split_data[0] not in entities ):
                data.append(split_data)
                entities.append(split_data[0])
                entities.append(split_data[2])
                relations.append(split_data[1])
            elif(split_data[1] in qEntity and split_data[1] not in entities ):
                data.append(split_data)
                entities.append(split_data[0])
                entities.append(split_data[2])
                relations.append(split_data[1])


        file.close()
        entities = list(set(entities))
        relations = list(set(relations))
        entity_ids = {entities[i]: i for i in range(len(entities))}
        relation_ids = {relations[i]: i for i in range(len(relations))}

        data_ids = [(entity_ids[data[i][0]], relation_ids[data[i][1]], entity_ids[data[i][2]]) for i in range(len(data))]


        f = open(dicPath  + 'entities.dict', 'w')

        for key, value in entity_ids.items():
            line = key + '\t' + str(value) + '\n'
            line = line.rstrip().split('\t')
            name = line[0]
            id = int(line[1])
            f.write(str(id) + '\t' + name + '\n')
        f.close()
    
        return data, entities, relations, entity_ids, relation_ids, data_ids


