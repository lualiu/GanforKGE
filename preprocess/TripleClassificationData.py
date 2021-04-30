import os
import numpy as np
import torch
from utils.readdata import read_dicts_from_file,read_triples_from_file,turn_triples_to_label_dict

class TripleClassificationData(object):
    def __init__(self,data_path,train_data_name,valid_data_name,test_data_name,with_reverse=False):
        self.entity_dict,self.relation_dict = read_dicts_from_file(
            [os.path.join(data_path,train_data_name),
            os.path.join(data_path,valid_data_name),
            os.path.join(data_path,test_data_name)],
            with_reverse=with_reverse
        )
        self.entity_numbers = len(self.entity_dict.keys())
        self.relation_numbers = len(self.relation_dict.keys())

        self.train_triples_with_reverse = read_triples_from_file(os.path.join(data_path, train_data_name),
                                                                 self.entity_dict, self.relation_dict,
                                                                 with_reverse=with_reverse)
        self.valid_triples_with_reverse,self.valid_triples_for_classification = self.read_triple_from_file(os.path.join(data_path, valid_data_name),
                                                                 self.entity_dict, self.relation_dict,
                                                                 with_reverse=with_reverse)
        self.test_triples_with_reverse,self.test_triples_for_classification = self.read_triple_from_file(os.path.join(data_path, test_data_name),
                                                                 self.entity_dict, self.relation_dict,
                                                                 with_reverse=with_reverse)

        self.train_numbers = len(self.train_triples_with_reverse)

        self.train_triples_dict = turn_triples_to_label_dict(self.train_triples_with_reverse)
        self.valid_triples_dict = turn_triples_to_label_dict(self.valid_triples_with_reverse)
        self.test_triples_dict = turn_triples_to_label_dict(self.test_triples_with_reverse)

        self.gold_triples_dict = dict(list(self.train_triples_dict.items()) +
                                      list(self.valid_triples_dict.items()) +
                                      list(self.test_triples_dict.items()))

        #del self.train_triples_with_reverse
        del self.valid_triples_dict
        del self.test_triples_dict

        self.train_triples_numpy_array = np.array(self.train_triples_with_reverse).astype(np.int32)

        self.valid_triples_for_classification = np.array(self.valid_triples_for_classification).astype(np.int32)
        self.test_triples_for_classification = np.array(self.test_triples_for_classification).astype(np.int32)

    def read_triple_from_file(self,filename,entity_dict,relation_dict,with_reverse):
        triples_list = []
        classification_triples_label = []
        with open(filename) as file:
            for line in file:
                head, relation, tail, label = line.strip().split('\t')
                if int(label) == 1:
                    triples_list.append([
                        entity_dict[head],
                        relation_dict[relation],
                        entity_dict[tail]
                    ])
                    if with_reverse:
                        relation_reverse = relation + '_reverse'
                        triples_list.append([
                            entity_dict[tail],
                            relation_dict[relation_reverse],
                            entity_dict[head]
                        ])
                classification_triples_label.append([
                    entity_dict[head],
                    relation_dict[relation],
                    entity_dict[tail],
                    label
                ])

        return triples_list,classification_triples_label

    def get_batch(self,batch_size):
        random_index = np.random.permutation(self.train_numbers)
        random_train_triple = self.train_triples_numpy_array[random_index]

        pointer = 0
        while pointer < self.train_numbers:
            start_index = pointer
            end_index = start_index + batch_size
            if end_index >= self.train_numbers:
                end_index = self.train_numbers
            pointer = end_index

            current_batch_size = end_index - start_index
            new_batch_train_triple_true = random_train_triple[start_index:end_index,:].copy()
            new_batch_train_triple_fake = random_train_triple[start_index:end_index,:].copy()

            random_words = np.random.randint(0,self.entity_numbers,current_batch_size)

            for index in range(current_batch_size):
                while (new_batch_train_triple_fake[index,0],
                       new_batch_train_triple_fake[index,1],
                       random_words[index]) in self.train_triples_dict:
                    random_words[index] = np.random.randint(0,self.entity_numbers)
                new_batch_train_triple_fake[index,2] = random_words[index]

            yield torch.tensor(new_batch_train_triple_true).long().cuda(),torch.tensor(new_batch_train_triple_fake).long().cuda()

