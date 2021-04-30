import os
from utils.readdata import read_dicts_from_file,read_triples_from_file
from abc import ABCMeta,abstractmethod

class KBData(object,metaclass=ABCMeta):
    def __init__(self,data_path,train_data_name,valid_data_name,test_data_name,with_reverse=False):
        self.entity_dict,self.relation_dict = read_dicts_from_file(
            [os.path.join(data_path,train_data_name),
            os.path.join(data_path,valid_data_name),
            os.path.join(data_path,test_data_name)],
            with_reverse=with_reverse
        )
        self.entity_numbers = len(self.entity_dict.keys())
        self.relation_numbers = len(self.relation_dict.keys())

        self.train_triples_with_reverse = read_triples_from_file(os.path.join(data_path,train_data_name),self.entity_dict,self.relation_dict,with_reverse=with_reverse)
        self.valid_triples_with_reverse = read_triples_from_file(os.path.join(data_path,valid_data_name),self.entity_dict,self.relation_dict,with_reverse=with_reverse)
        self.test_triples_with_reverse = read_triples_from_file(os.path.join(data_path,test_data_name),self.entity_dict,self.relation_dict,with_reverse=with_reverse)

    @abstractmethod
    def get_batch(self,batch_size):
        pass
