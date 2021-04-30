import torch
import time
import os
import numpy as np
import shutil
from tqdm import tqdm
from torch.nn import functional as F
from tensorboard_logger import configure, log_histogram, log_value

from train.TrainBaseClass import TrainBaseClass
from utils.evaluation import ranking_and_hits, ranking_and_hits_filter,turn_triples_to_separate_tensors

class TrainTripleClassifcation(TrainBaseClass):
    def set_model(self, embedding, geneartor, discriminator):
        self.embedding = embedding
        self.geneartor = geneartor
        self.discriminator = discriminator

    def model_eval(self):
        self.embedding.eval()
        self.geneartor.eval()
        self.discriminator.eval()

    def model_train(self):
        self.embedding.train()
        self.geneartor.train()
        self.discriminator.train()

    def model_zero_grad(self):
        self.embedding.zero_grad()
        self.geneartor.zero_grad()
        self.discriminator.zero_grad()

    '''
    :param use_pretrained: is use pretrained model
    :param pretrained_model_file: the file path of pretrained model, if use_pretrained is True, use this parameter
    :param learning_rate: learning rate
    :param weight_decay: weight decay to avoid overfitting
    :param margin: the margin in margin loss
    :param epochs: max epochs
    :param batch_size: batch size
    :param evaluation_times: the number of separations for each evaluation
    :param save_times: the number of separations for each save 
    :param save_path: the path to save model
    :param log_path: the path to log(use for tensorboard) 
    :param d_tune_embedding: whether the discriminator network tune the embedding parameter. Default is True
    :param g_tune_embedding: whether the discriminator network tune the embedding parameter. Default is Tru
    :param d_margin_type: the margin type loss used in the loss function of discriminator network    
                          if d_margin_type is True, d_loss = relu(2 * true - fake - G_fake + 1.0)
                          else  d_loss = relu(true - fake + 1.0) + true - G_fake
                          defaut is True
    :param g_use_negative: whether the generator network use negative sample
    :param mean_or_sum: the type of loss function
    :param print_file: the file path to print result. Default is None
    '''
    def train(self,
              use_pretrained,
              pretrained_model_file,
              learning_rate,
              weight_decay,
              margin,
              epochs,
              batch_size,
              evaluation_times,
              save_times,
              save_path,
              log_path,
              d_tune_embedding = True,
              g_tune_embedding = True,
              d_margin_type = True,
              g_use_negative = False,
              mean_or_sum = 'mean',
              print_file = None):
        self.internals = 100
        loss_function = torch.mean
        if mean_or_sum == 'sum':
            loss_function = torch.sum

        if os.path.exists(log_path):
            shutil.rmtree(log_path)
        os.mkdir(log_path)
        configure(log_path, flush_secs=5)
        self.model_eval()

        self.print_file = None
        if print_file is not None:
            self.print_file = open(print_file,'w')

        with torch.no_grad():
            self.evaluation_d(self.data.valid_triples_for_classification, self.data.test_triples_for_classification,
                          'test(d)', 0)
        if use_pretrained:
            self.embedding.init(pretrained_model_file)
            self.model_eval()
            torch.cuda.empty_cache()
            self.evaluation_d(self.data.valid_triples_for_classification, self.data.test_triples_for_classification, 'test(d)', 0)
            self.evaluation_g(self.data.test_triples_with_reverse, 'test(g)', 0)

        if d_tune_embedding:
            d_parameters = list(self.embedding.parameters()) + list(self.discriminator.parameters())
        else:
            d_parameters = list(self.discriminator.parameters())

        if g_tune_embedding:
            g_parameters = list(self.embedding.parameters()) + list(self.geneartor.parameters())
        else:
            g_parameters = list(self.geneartor.parameters())

        train_d_op = torch.optim.RMSprop(d_parameters, lr=learning_rate,weight_decay=weight_decay)
        train_g_op = torch.optim.RMSprop(g_parameters, lr=learning_rate,weight_decay=weight_decay)
        #train_d_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(train_d_op,milestones=[100],gamma=0.1)
        #train_g_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(train_g_op,milestones=[100],gamma=0.1)

        base_critic = 1

        for epoch in tqdm(range(epochs)):
            #train_g_lr_scheduler.step()
            #train_d_lr_scheduler.step()
            torch.cuda.empty_cache()

            self.model_train()
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_d_hard_loss = 0.0
            epoch_d_soft_loss = 0.0

            start_epoch = time.time()

            # train d
            for i in range(base_critic):
                for batch_true, batch_fake in self.data.get_batch(batch_size):
                    self.embedding.zero_grad()
                    self.discriminator.zero_grad()
                    train_d_op.zero_grad()

                    head_embedding, relation_embedding, tail_embedding = self.embedding.forward(batch_true[:,0],
                                                                                                batch_true[:,1],
                                                                                                e2=batch_true[:,2])
                    _, _, tail_embedding_fake_hard = self.embedding.forward(batch_fake[:, 0],
                                                                            batch_fake[:, 1],
                                                                            e2=batch_fake[:, 2])
                    fake_g_output = self.geneartor.forward(head_embedding,relation_embedding)
                    d_loss_true = self.discriminator.forward(head_embedding, relation_embedding, tail_embedding)
                    d_loss_fake_soft = self.discriminator.forward(head_embedding, relation_embedding, fake_g_output)
                    d_loss_fake_hard = self.discriminator.forward(head_embedding, relation_embedding, tail_embedding_fake_hard)
                    if d_margin_type is True:
                        d_loss = loss_function(F.relu(2 * d_loss_true - d_loss_fake_hard - d_loss_fake_soft + margin))
                    else:
                        d_loss = loss_function(F.relu(d_loss_true - d_loss_fake_hard + margin)) + loss_function(d_loss_true - d_loss_fake_soft)
                    d_loss.backward()
                    train_d_op.step()
                    epoch_d_loss += d_loss.item() * batch_true.size()[0]

                    self.embedding.l2_normalize()

            epoch_d_loss /= base_critic
            epoch_d_soft_loss /= base_critic
            epoch_d_hard_loss /= base_critic


            for batch_true, batch_fake in self.data.get_batch(batch_size):
                self.embedding.zero_grad()
                self.geneartor.zero_grad()
                train_g_op.zero_grad()

                # train g
                head_embedding,relation_embedding,tail_embedding = self.embedding.forward(batch_true[:, 0],
                                                                                          batch_true[:, 1],
                                                                                          e2=batch_true[:, 2])
                fake_g_output = self.geneartor.forward(head_embedding,relation_embedding)
                g_loss = loss_function(self.discriminator.forward(head_embedding,relation_embedding,fake_g_output))

                if g_use_negative:
                    _, _, tail_embedding_fake_hard = self.embedding.forward(batch_fake[:, 0],
                                                                            batch_fake[:, 1],
                                                                            e2=batch_fake[:, 2])
                    batch_true_loss = torch.mean((head_embedding + relation_embedding - tail_embedding)**2,dim=1)
                    batch_fake_loss = torch.mean((head_embedding + relation_embedding - tail_embedding_fake_hard)**2,dim=1)
                    g_loss += loss_function(F.relu(batch_true_loss - batch_fake_loss + margin))

                g_loss.backward()
                train_g_op.step()
                self.embedding.l2_normalize()
                epoch_g_loss += g_loss.item() * batch_true.size()[0]

            end = time.time()
            epoch_d_hard_loss /= self.data.train_numbers
            epoch_d_soft_loss /= self.data.train_numbers
            epoch_g_loss /= self.data.train_numbers
            epoch_d_loss /= self.data.train_numbers

            self.log_histogram_and_value(epoch_d_loss,epoch_g_loss,epoch_d_hard_loss,epoch_d_soft_loss,epoch)

            print('epoch {}     d_loss: {}       g_loss: {}   time:{} s'.
                  format(epoch,
                         epoch_d_loss,
                         epoch_g_loss,
                         end - start_epoch))

            if epoch != 0 and epoch % evaluation_times == 0:
                with torch.no_grad():
                    self.model_eval()
                    #self.print_file.write('\nepoch: {0}\n'.format(epoch))
                    #self.evaluation_g(self.data.test_triples_with_reverse, "test(g)", epoch)

                    self.print_file.write('\nepoch: {0}. TransE score: \n'.format(epoch))
                    self.evaluation_d(self.data.valid_triples_for_classification, self.data.test_triples_for_classification, "test(d)",epoch)

            if epoch != 0 and epoch % save_times == 0:
                torch.save(self.embedding.state_dict(),
                           os.path.join(save_path, "embedding-model-{0}.pkl".format(epoch)))
                torch.save(self.geneartor.state_dict(), os.path.join(save_path, "g-model-{0}.pkl".format(epoch)))
                torch.save(self.discriminator.state_dict(), os.path.join(save_path, "d-model-{0}.pkl").format(epoch))

        with torch.no_grad():
            self.model_eval()
            #self.print_file.write('\nepoch: {0}\n'.format(epochs))
            #self.evaluation_g(self.data.test_triples_with_reverse, "test(g)", epochs)

            self.print_file.write('\nepoch: {0}. TransE score: \n'.format(epochs))
            self.evaluation_d(self.data.valid_triples_for_classification, self.data.test_triples_for_classification, "test(d)", epochs)

        torch.save(self.embedding.state_dict(), os.path.join(save_path, "embedding-model-{0}.pkl".format(epochs)))
        torch.save(self.geneartor.state_dict(), os.path.join(save_path, "g-model-{0}.pkl".format(epochs)))
        torch.save(self.discriminator.state_dict(), os.path.join(save_path, "d-model-{0}.pkl").format(epochs))

        if print_file is not None:
            self.print_file.close()

    def evaluation_g(self, eval_dataset, dataset_name,epoch):
        eval_dataset_len = len(eval_dataset) // 2
        head_relation_tail_data = [eval_dataset[i * 2] for i in range(eval_dataset_len)]
        tail_reverse_relation_head_data = [eval_dataset[i * 2 + 1] for i in range(eval_dataset_len)]

        head_list, relation_list = turn_triples_to_separate_tensors(head_relation_tail_data, 0, 1)
        tail_list, reverse_relation_list = turn_triples_to_separate_tensors(tail_reverse_relation_head_data, 0, 1)

        argsort_tail_list = []
        for head,relation in zip(head_list,relation_list):
            head_embedding,relation_embedding = self.embedding.forward(torch.unsqueeze(head,0),torch.unsqueeze(relation,0))
            pred_tail = self.geneartor.forward(head_embedding,relation_embedding)
            pred_tail = self.embedding.get_distance(pred_tail)
            max_values, argsort_tail = torch.sort(pred_tail, 0, descending=False)
            argsort_tail_list.append(np.expand_dims(argsort_tail.cpu().numpy(),axis=0))
        argsort_tail_list = np.concatenate(argsort_tail_list,axis=0)

        argsort_head_list = []
        for tail,reverse_relation in zip(tail_list,reverse_relation_list):
            tail_embedding, relation_reverse_embedding = self.embedding.forward(torch.unsqueeze(tail,0),torch.unsqueeze(reverse_relation,0))
            pred_head = self.geneartor.forward(tail_embedding,relation_reverse_embedding)
            pred_head = self.embedding.get_distance(pred_head)
            max_values, argsort_head = torch.sort(pred_head, 0, descending=False)
            argsort_head_list.append(np.expand_dims(argsort_head.cpu().numpy(),axis=0))
        argsort_head_list = np.concatenate(argsort_head_list,axis=0)

        MR, hit10, MRR = ranking_and_hits(dataset_name, head_relation_tail_data, tail_reverse_relation_head_data, argsort_tail_list,
                         argsort_head_list,print_file=self.print_file)
        if epoch!=0 and epoch%100==0:
            ranking_and_hits_filter(dataset_name, head_relation_tail_data, tail_reverse_relation_head_data, argsort_tail_list, argsort_head_list, self.data.gold_triples_dict, print_file=self.print_file)

        log_value(dataset_name+'/MR', MR, epoch)
        log_value(dataset_name + '/hit10', hit10, epoch)
        log_value(dataset_name + '/MRR', MRR, epoch)

    def evaluation_d(self, valid_dataset, test_dataset, dataset_name, epoch):
        valid_result = torch.tensor(valid_dataset).long().cuda()
        head_embedding, relation_embedding, tail_embedding = self.embedding.forward(valid_result[:,0],valid_result[:,1],e2=valid_result[:,2])
        valid_result = self.discriminator.forward(head_embedding,relation_embedding,tail_embedding)
        valid_result = valid_result.cpu().numpy()

        test_result = torch.tensor(test_dataset).long().cuda()
        head_embedding, relation_embedding, tail_embedding = self.embedding.forward(test_result[:,0],test_result[:,1],e2=test_result[:,2])
        test_result = self.discriminator.forward(head_embedding,relation_embedding,tail_embedding)
        test_result = test_result.cpu().numpy()

        test_accuracy = []
        for rel in self.data.relation_dict.values():
            valid_rel_score = []
            for i in range(len(valid_dataset)):
                if valid_dataset[i][1] == rel:
                    valid_rel_score.append([valid_result[i],True if valid_dataset[i][3]==1 else False])
            valid_rel_score = np.array(valid_rel_score)

            test_rel_score = []
            for i in range(len(test_dataset)):
                if test_dataset[i][1] == rel:
                    test_rel_score.append([test_result[i],True if test_dataset[i][3]==1 else False])
            test_rel_score = np.array(test_rel_score)

            if valid_rel_score.shape[0] != 0 and test_rel_score.shape[0] != 0:
                max_score = np.max(valid_rel_score[:,0])
                min_score = np.min(valid_rel_score[:,0])

                internals_value = np.arange(min_score,max_score,(max_score-min_score)/self.internals)

                max_accuracy = -1.0
                best_internal = 0
                for internal in internals_value:
                    accuracy = self.cal_accuracy(valid_rel_score,internal)
                    if accuracy > max_accuracy:
                        max_accuracy = accuracy
                        best_internal = internal
                test_accuracy.append(self.cal_accuracy(test_rel_score,best_internal))
        accuracy = np.mean(test_accuracy)

        print('In {0} dataset: '.format(dataset_name))
        print('Accuracy is {0}'.format(accuracy))

        if self.print_file is not None:
            self.print_file.write('In {0} dataset: \n'.format(dataset_name))
            self.print_file.write('Accuracy is {0}'.format(accuracy))
        log_value(dataset_name + '/accuracy', accuracy, epoch)

    def cal_accuracy(self,rel_score,threshold):
        accuracy = rel_score[:, 0] < threshold
        return np.mean(accuracy == rel_score[:, 1])

    '''
    def evaluation_d(self, eval_dataset, dataset_name, epoch):
        eval_dataset_len = len(eval_dataset) // 2
        head_relation_tail_data = [eval_dataset[i * 2] for i in range(eval_dataset_len)]
        tail_reverse_relation_head_data = [eval_dataset[i * 2 + 1] for i in range(eval_dataset_len)]

        head_list, relation_list = turn_triples_to_separate_tensors(head_relation_tail_data, 0, 1)
        tail_list, reverse_relation_list = turn_triples_to_separate_tensors(tail_reverse_relation_head_data, 0, 1)

        argsort_tail_list = []
        for head,relation in zip(head_list,relation_list):
            head_embedding, relation_embedding = self.embedding.forward(head, relation)
            tail_embeddings = self.embedding.get_evaluation_weight(relation)
            head_embedding = head_embedding.expand(tail_embeddings.size()[0],tail_embeddings.size()[1])
            relation_embedding = relation_embedding.expand(tail_embeddings.size()[0],tail_embeddings.size()[1])
            pred_tail = self.discriminator.forward(head_embedding,relation_embedding,tail_embeddings)
            max_values, argsort_tail = torch.sort(pred_tail, 0, descending=False)
            argsort_tail_list.append(np.expand_dims(argsort_tail.cpu().numpy(), axis=0))
        argsort_tail_list = np.concatenate(argsort_tail_list, axis=0)

        argsort_head_list = []
        for tail,reverse_relation in zip(tail_list,reverse_relation_list):
            tail_embedding, relation_reverse_embedding = self.embedding.forward(tail,reverse_relation)
            head_embeddings = self.embedding.get_evaluation_weight(reverse_relation)
            tail_embedding = tail_embedding.expand(head_embeddings.size()[0],head_embeddings.size()[1])
            relation_reverse_embedding = relation_reverse_embedding.expand(head_embeddings.size()[0],head_embedding.size()[1])
            pred_head = self.discriminator.forward(tail_embedding, relation_reverse_embedding, head_embeddings)
            max_values, argsort_head = torch.sort(pred_head, 0, descending=False)
            argsort_head_list.append(np.expand_dims(argsort_head.cpu().numpy(), axis=0))
        argsort_head_list = np.concatenate(argsort_head_list, axis=0)

        MR,hit10,MRR = ranking_and_hits(dataset_name, head_relation_tail_data, tail_reverse_relation_head_data, argsort_tail_list,
                         argsort_head_list,print_file=self.print_file)

        if epoch!=0 and epoch%100==0:
            ranking_and_hits_filter(dataset_name, head_relation_tail_data, tail_reverse_relation_head_data, argsort_tail_list, argsort_head_list, self.data.gold_triples_dict, print_file=self.print_file)

        log_value(dataset_name+'/MR', MR, epoch)
        log_value(dataset_name + '/hit10', hit10, epoch)
        log_value(dataset_name + '/MRR', MRR, epoch)
    '''

    def log_histogram_and_value(self,epoch_d_loss,epoch_g_loss,epoch_d_hard_loss,epoch_d_soft_loss,epoch):
        log_value('d_loss',epoch_d_loss,epoch)
        log_value('g_loss',epoch_g_loss,epoch)
        log_value('d_hard_loss',epoch_d_hard_loss,epoch)
        log_value('d_soft_loss',epoch_d_soft_loss,epoch)

        log_histogram('entity',self.embedding.entity_embedding.weight.data.cpu(),epoch)
        log_histogram('relation',self.embedding.relation_embedding.weight.data.cpu(),epoch)

        head_list = []
        relation_list = []
        tail_list = []
        for triple in self.data.valid_triples_with_reverse:
            head_list.append(triple[0])
            relation_list.append(triple[1])
            tail_list.append(triple[2])

        with torch.no_grad():
            head_list = torch.LongTensor(head_list).cuda()
            relation_list = torch.LongTensor(relation_list).cuda()
            tail_list = torch.LongTensor(tail_list).cuda()

            head_embedding, relation_embedding, tail_embedding = self.embedding.forward(head_list, relation_list, e2=tail_list)
            g_output = self.geneartor.forward(head_embedding,relation_embedding).data

            g_true_loss = torch.mean(self.discriminator.forward(head_embedding,relation_embedding,tail_embedding))
            g_fake_loss = torch.mean(self.discriminator.forward(head_embedding,relation_embedding,g_output))

            log_value('valid_g_true_loss',g_true_loss.cpu(),epoch)
            log_value('valid_g_fake_loss',g_fake_loss.cpu(),epoch)

            #for name,f in self.geneartor.named_parameters():
            #    log_histogram(name,f.data.cpu(),epoch)




