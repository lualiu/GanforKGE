from itertools import count

def read_dicts_from_file(filenames,with_reverse=False):
    entity_set = set()
    relation_set = set()

    for filename in filenames:
        with open(filename) as file:
            for line in file:
                head,relation,tail = line.strip().split('\t')[:3]
                entity_set.add(head)
                entity_set.add(tail)
                relation_set.add(relation)
                if with_reverse:
                    relation_reverse = relation +'_reverse'
                    relation_set.add(relation_reverse)

    return turn_set_to_dict(entity_set),turn_set_to_dict(relation_set)

def read_triples_from_file(filename,entity_dict,relation_dict,with_reverse=False):
    triples_list = []
    with open(filename) as file:
        for line in file:
            head, relation, tail = line.strip().split('\t')[:3]
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

    return triples_list


def turn_set_to_dict(data_set):
    data_list = sorted(list(data_set))
    return dict(zip(data_list,count()))

def turn_triples_to_label_dict(triples):
    triple_dict = dict()
    for triple in triples:
        triple_dict[(triple[0],triple[1],triple[2])] = 1.0
    return triple_dict


def cal_head_tail_selector(triples,relation_dict):
    left_entity = {}
    right_entity = {}

    for head,relation,tail in triples:
        if relation not in left_entity:
            left_entity[relation] = {}
        if head not in left_entity[relation]:
            left_entity[relation][head] = 0
        left_entity[relation][head] += 1

        if relation not in right_entity:
            right_entity[relation] = {}
        if tail not in right_entity[relation]:
            right_entity[relation][tail] = 0
        right_entity[relation][tail] += 1

    left_avg = {}
    for relation in relation_dict.values():
        left_avg[relation] = sum(left_entity[relation].values()) * 1.0 / len(left_entity[relation])

    right_avg = {}
    for relation in relation_dict.values():
        right_avg[relation] = sum(right_entity[relation].values()) * 1.0 / len(right_entity[relation])

    headTailSelector = {}
    for relation in relation_dict.values():
        headTailSelector[relation] = 1000 * right_avg[relation] / (right_avg[relation] + left_avg[relation])

    return headTailSelector


