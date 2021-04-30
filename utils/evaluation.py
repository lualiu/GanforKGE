import torch
import numpy as np

def ranking_and_hits(dataset_name,
                     head_relation_tail_data,
                     tail_reverse_relation_head_data,
                     argsort_tail,
                     argsort_head,
                     print_file = None):
    hits_head = []
    hits_tail = []
    hits = []
    ranks = []
    ranks_head = []
    ranks_tail = []

    for i in range(10):
        hits_head.append([])
        hits_tail.append([])
        hits.append([])

    for i in range(len(head_relation_tail_data)):
        rank_tail = np.where(argsort_tail[i] == head_relation_tail_data[i][2])[0][0]
        rank_head = np.where(argsort_head[i] == tail_reverse_relation_head_data[i][2])[0][0]
        ranks.append(rank_head + 1)
        ranks.append(rank_tail + 1)
        ranks_tail.append(rank_tail + 1)
        ranks_head.append(rank_head + 1)

        for j in range(10):
            if rank_head <= j:
                hits[j].append(1.0)
                hits_head[j].append(1.0)
            else:
                hits[j].append(0.0)
                hits_head[j].append(0.0)

            if rank_tail <= j:
                hits[j].append(1.0)
                hits_tail[j].append(1.0)
            else:
                hits[j].append(0.0)
                hits_tail[j].append(0.0)

    print_evalution_result(dataset_name, hits_head, hits_tail, hits,ranks_head, ranks_tail, ranks,
                           print_file, 'raw')
    return np.mean(ranks),np.mean(hits[9]),np.mean(1.0 / np.array(ranks))


def ranking_and_hits_filter(dataset_name,
                            head_relation_tail_data,
                            tail_reverse_relation_head_data,
                            argsort_tail,
                            argsort_head,
                            gold_triples,
                            print_file=None):
    hits_head = []
    hits_tail = []
    hits = []
    ranks = []
    ranks_head = []
    ranks_tail = []

    for i in range(10):
        hits_head.append([])
        hits_tail.append([])
        hits.append([])

    entity_number = argsort_tail.shape[1]
    for i in range(len(head_relation_tail_data)):
        rank_tail = 0
        for j in range(entity_number):
            if argsort_tail[i][j] == head_relation_tail_data[i][2]:
                break
            else:
                if (head_relation_tail_data[i][0],head_relation_tail_data[i][1],argsort_tail[i][j]) in gold_triples:
                    continue
                else:
                    rank_tail += 1
        ranks.append(rank_tail + 1)
        ranks_tail.append(rank_tail + 1)
        for j in range(10):
            if rank_tail <= j:
                hits[j].append(1.0)
                hits_tail[j].append(1.0)
            else:
                hits[j].append(0.0)
                hits_tail[j].append(0.0)

    for i in range(len(tail_reverse_relation_head_data)):
        rank_head = 0
        for j in range(entity_number):
            if argsort_head[i][j] == tail_reverse_relation_head_data[i][2]:
                break
            else:
                if (tail_reverse_relation_head_data[i][0],tail_reverse_relation_head_data[i][1],argsort_head[i][j]) in gold_triples:
                    continue
                else:
                    rank_head += 1
        ranks.append(rank_head + 1)
        ranks_head.append(rank_head + 1)
        for j in range(10):
            if rank_head <= j:
                hits[j].append(1.0)
                hits_head[j].append(1.0)
            else:
                hits[j].append(0.0)
                hits_head[j].append(0.0)

    print_evalution_result(dataset_name, hits_head, hits_tail, hits,ranks_head, ranks_tail, ranks,
                           print_file, 'filter')

def turn_triples_to_separate_tensors(triples,entity_dim,relation_dim):
    entity_list = []
    relation_list = []
    for triple in triples:
        entity_list.append(triple[entity_dim])
        relation_list.append(triple[relation_dim])

    return torch.LongTensor(entity_list).cuda(),torch.LongTensor(relation_list).cuda()


def print_evalution_result(dataset_name,
                           hits_head,hits_tail,hits,
                           ranks_head,ranks_tail,ranks,
                           print_file,print_type):

    print(print_type)
    print('In {0} dataset: '.format(dataset_name))
    for i in range(10):
        print('Hits left @{0}: {1}'.format(i + 1, np.mean(hits_head[i])))
        print('Hits right @{0}: {1}'.format(i + 1, np.mean(hits_tail[i])))
        print('Hits @{0}: {1}'.format(i + 1, np.mean(hits[i])))
    print('Mean rank left: {0}'.format(np.mean(ranks_head)))
    print('Mean rank right: {0}'.format(np.mean(ranks_tail)))
    print('Mean rank: {0}'.format(np.mean(ranks)))
    print('Mean reciprocal rank left: {0}'.format(np.mean(1.0 / np.array(ranks_head))))
    print('Mean reciprocal rank right: {0}'.format(np.mean(1.0 / np.array(ranks_tail))))
    print('Mean reciprocal rank: {0}'.format(np.mean(1.0 / np.array(ranks))))

    if print_file is not None:
        print_file.write(print_type + '\n')
        print_file.write('In {0} dataset: \n'.format(dataset_name))
        for i in range(10):
            print_file.write('Hits left @{0}: {1}\n'.format(i + 1, np.mean(hits_head[i])))
            print_file.write('Hits right @{0}: {1}\n'.format(i + 1, np.mean(hits_tail[i])))
            print_file.write('Hits @{0}: {1}\n'.format(i + 1, np.mean(hits[i])))
        print_file.write('Mean rank left: {0}\n'.format(np.mean(ranks_head)))
        print_file.write('Mean rank right: {0}\n'.format(np.mean(ranks_tail)))
        print_file.write('Mean rank: {0}\n'.format(np.mean(ranks)))
        print_file.write('Mean reciprocal rank left: {0}\n'.format(np.mean(1.0 / np.array(ranks_head))))
        print_file.write('Mean reciprocal rank right: {0}\n'.format(np.mean(1.0 / np.array(ranks_tail))))
        print_file.write('Mean reciprocal rank: {0}\n'.format(np.mean(1.0 / np.array(ranks))))