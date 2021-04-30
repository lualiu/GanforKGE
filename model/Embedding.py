import torch
import torch.nn.functional as F
from model.Model import Model
from utils.projection import *

class Embedding(Model):
    def __init__(self,num_entities,num_relations,embedding_dim):
        super(Embedding,self).__init__()

        self.entity_embedding = torch.nn.Embedding(num_entities,embedding_dim)
        self.relation_embedding = torch.nn.Embedding(num_relations,embedding_dim)

    def get_distance(self, embedding):
        return torch.sum((embedding - self.entity_embedding.weight) ** 2, dim=1)

    def get_evaluation_weight(self, relation):
        pass


class TransE_E(Embedding):
    def __init__(self,num_entities,num_relations,embedding_dim):
        super(TransE_E, self).__init__(num_entities,num_relations,embedding_dim)

        self.l2_normalize()

    def l2_normalize(self):
        self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight, dim=1, p=2)
        self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight, dim=1, p=2)

    def forward(self,e1,rel,e2=None):
        if e2 is not None:
            e1_embedded = self.entity_embedding(e1)
            rel_embedded = self.relation_embedding(rel)
            e2_embedded = self.entity_embedding(e2)
            return e1_embedded,rel_embedded,e2_embedded
        else:
            e1_embedded = self.entity_embedding(e1)
            rel_embedded = self.relation_embedding(rel)
            return e1_embedded,rel_embedded

    def get_evaluation_weight(self, relation):
        return self.entity_embedding.weight

class TransH_E(Embedding):
    def __init__(self,num_entities,num_relations,embedding_dim):
        super(TransH_E, self).__init__(num_entities,num_relations,embedding_dim)

        self.norm_weight = torch.nn.Embedding(num_relations, embedding_dim)
        self.l2_normalize()

    def l2_normalize(self):
        self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight, dim=1, p=2)
        self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight, dim=1, p=2)
        self.norm_weight.weight.data = F.normalize(self.norm_weight.weight,dim=1,p=2)

    def forward(self,e1,rel,e2=None):
        e1_embedding = self.entity_embedding(e1)
        rel_embedding = self.relation_embedding(rel)
        rel_norm_embedding = F.normalize(self.norm_weight(rel),dim=1,p=2)
        e1_relation_projection = projection_transH(e1_embedding, rel_norm_embedding)

        if e2 is not None:
            e2_embedding = self.entity_embedding(e2)
            e2_relation_projection = projection_transH(e2_embedding,rel_norm_embedding)

            return e1_relation_projection,rel_embedding,e2_relation_projection
        else:
            return e1_relation_projection,rel_embedding

    def get_evaluation_weight(self, relation):
        rel_norm_embedding = F.normalize(self.norm_weight(relation), dim=1, p=2)
        return projection_transH(self.entity_embedding.weight,rel_norm_embedding)

class TransD_E(Model):
    def __init__(self,num_entities,num_relations,embedding_dim):
        super(TransD_E, self).__init__()

        self.entity_proj_embedding = torch.nn.Embedding(num_entities,embedding_dim)
        self.relation_proj_embedding = torch.nn.Embedding(num_relations,embedding_dim)

        self.l2_normalize()

    def l2_normalize(self):
        self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight,dim=1,p=2)
        self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight,dim=1,p=2)
        self.entity_proj_embedding.weight.data = F.normalize(self.entity_proj_embedding.weight,dim=1,p=2)
        self.relation_proj_embedding.weight.data = F.normalize(self.relation_proj_embedding.weight,dim=1,p=2)

    def forward(self,e1,rel,e2=None):
        e1_embedded = self.entity_embedding(e1)
        rel_embedded = self.relation_embedding(rel)
        e1_proj_embedded = self.entity_proj_embedding(e1)
        rel_proj_embedded = self.relation_proj_embedding(rel)
        e1_relation_projection = projection_transD(e1_embedded,e1_proj_embedded,rel_proj_embedded)

        if e2 is not None:
            e2_embedded = self.entity_embedding(e2)
            e2_proj_embedded = self.entity_proj_embedding(e2)
            e2_relation_projection = projection_transD(e2_embedded,e2_proj_embedded,rel_proj_embedded)
            return e1_relation_projection,rel_embedded,e2_relation_projection
        else:
            return e1_relation_projection,rel_embedded

    def get_evaluation_weight(self, relation):
        rel_proj_embedded = self.relation_proj_embedding(rel)
        return projection_transD(self.entity_embedding.weight,rel_proj_embedded,self.entity_proj_embedding.weight)