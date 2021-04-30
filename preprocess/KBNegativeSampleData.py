import os
import torch
import numpy as np
from multiprocessing import Manager,Process
from utils.readdata import turn_triples_to_label_dict,cal_head_tail_selector
from preprocess.KBData import KBData


class KBNegativeSampleData(KBData):
    def __init__(self, data_path, train_data_name,valid_data_name,test_data_name,with_reverse=False):
        KBData.__init__(self,data_path, train_data_name, valid_data_name, test_data_name, with_reverse=with_reverse)

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

        self.train_triples_numpy_array = np.array(list(self.train_triples_dict.keys())).astype(np.int32)

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
