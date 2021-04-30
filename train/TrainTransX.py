import torch
import time
import os
import shutil
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from tensorboard_logger import configure, log_histogram, log_value

from train.TrainBaseClass import TrainBaseClass
from utils.evaluation import ranking_and_hits, turn_triples_to_separate_tensors,ranking_and_hits_filter


class TrainTransX(TrainBaseClass):
    def set_model(self, model):
        self.model = model

    def train(self,
              use_pretrained,
              pretrained_model_file,
              learning_rate,
              weight_decay,
              epochs,
              batch_size,
              margin,
              evaluation_times,
              save_times,
              save_path,
              log_path,
              print_file=None):
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
        os.mkdir(log_path)
        configure(log_path, flush_secs=5)

        self.print_file = None
        if print_file is not None:
            self.print_file = open(print_file,'w')

        if use_pretrained:
            self.model.init(pretrained_model_file)
            with torch.no_grad():
                self.model.eval()
                self.evaluation(self.data.test_triples_with_reverse, "test", epochs)

        train_op = torch.optim.Adam(self.model.parameters(),lr=learning_rate,weight_decay=weight_decay)

        for epoch in tqdm(range(epochs)):
            self.model.train()
            epoch_loss = 0.0

            start_epoch = time.time()

            for batch_true,batch_fake in self.data.get_batch(batch_size):
                self.model.zero_grad()
                train_op.zero_grad()

                batch_true_output = self.model.forward(batch_true[:,0],batch_true[:,1],batch_true[:,2])
                batch_fake_output = self.model.forward(batch_fake[:,0],batch_fake[:,1],batch_fake[:,2])
                loss = torch.mean(F.relu(batch_true_output-batch_fake_output + margin))
                loss.backward()
                train_op.step()
                epoch_loss += loss.item() * batch_true.size()[0]

            epoch_loss /= self.data.train_numbers
            end = time.time()
            print('epoch {}     loss: {}      time:{} s'.format(epoch,epoch_loss,end - start_epoch))

            self.log_histogram_and_value(epoch_loss,epoch)
            if epoch != 0 and epoch % evaluation_times == 0:
                with torch.no_grad():
                    self.model.eval()
                    self.print_file.write('\nepoch: {0}\n'.format(epoch))
                    self.evaluation(self.data.test_triples_with_reverse, "test",epoch)

            if epoch != 0 and epoch % save_times == 0:
                torch.save(self.model.state_dict(), os.path.join(save_path, "embedding-model-{0}.pkl".format(epoch)))

        with torch.no_grad():
            self.model.eval()
            self.print_file.write('\nepoch: {0}\n'.format(epochs))
            self.evaluation(self.data.valid_triples_with_reverse, "valid",epochs)
            self.evaluation(self.data.test_triples_with_reverse, "test",epochs)

        torch.save(self.model.state_dict(), os.path.join(save_path, "embedding-model-{0}.pkl".format(epochs)))

        if print_file is not None:
            self.print_file.close()

    def evaluation(self,eval_dataset,dataset_name,epoch):
        eval_dataset_len = len(eval_dataset) // 2
        head_relation_tail_data = [eval_dataset[i * 2] for i in range(eval_dataset_len)]
        tail_reverse_relation_head_data = [eval_dataset[i * 2 + 1] for i in range(eval_dataset_len)]

        head_list, relation_list = turn_triples_to_separate_tensors(head_relation_tail_data, 0, 1)
        tail_list, reverse_relation_list = turn_triples_to_separate_tensors(tail_reverse_relation_head_data, 0, 1)

        argsort_tail_list = []
        for head,relation in zip(head_list,relation_list):
            pred_tail = self.model.forward(torch.unsqueeze(head,dim=0),torch.unsqueeze(relation,dim=0))
            max_values, argsort_tail = torch.sort(pred_tail, 0, descending=False)
            argsort_tail_list.append(np.expand_dims(argsort_tail.cpu().numpy(), axis=0))
        argsort_tail_list = np.concatenate(argsort_tail_list, axis=0)

        argsort_head_list = []
        for tail,reverse_relation in zip(tail_list,reverse_relation_list):
            pred_head = self.model.forward(torch.unsqueeze(tail,dim=0),torch.unsqueeze(reverse_relation,dim=0))
            max_values, argsort_head = torch.sort(pred_head, 0, descending=False)
            argsort_head_list.append(np.expand_dims(argsort_head.cpu().numpy(), axis=0))
        argsort_head_list = np.concatenate(argsort_head_list, axis=0)

        MR,hit10,MRR = ranking_and_hits(dataset_name, head_relation_tail_data, tail_reverse_relation_head_data, argsort_tail_list,
                         argsort_head_list, print_file=self.print_file)

        if epoch != 0 and epoch % 100 == 0:
            ranking_and_hits_filter(dataset_name, head_relation_tail_data, tail_reverse_relation_head_data, argsort_tail_list,
                         argsort_head_list, self.data.gold_triples_dict, print_file=self.print_file)

        log_value(dataset_name+'/MR', MR, epoch)
        log_value(dataset_name + '/hit10', hit10, epoch)
        log_value(dataset_name + '/MRR', MRR, epoch)

    def log_histogram_and_value(self, epoch_loss, epoch):
        log_value('loss', epoch_loss, epoch)

        log_histogram('entity', self.model.entity_embedding.weight.data.cpu(), epoch)
        log_histogram('relation', self.model.relation_embedding.weight.data.cpu(), epoch)
