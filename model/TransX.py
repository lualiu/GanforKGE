import torch
import torch.nn.functional as F
from model.Model import Model
from utils.projection import *


class TransE(Model):
    def __init__(self,num_entities,num_relations,embedding_dim):
        super(TransE,self).__init__()

        self.entity_embedding = torch.nn.Embedding(num_entities,embedding_dim)
        self.relation_embedding = torch.nn.Embedding(num_relations,embedding_dim)

        self.l2_normalize()

    def forward(self,e1,rel,e2 = None):
        e1_embedding = self.entity_embedding(e1)
        rel_embedding = self.relation_embedding(rel)
        if e2 is not None:
            e2_embedding = self.entity_embedding(e2)
            return torch.sqrt(torch.sum((e1_embedding + rel_embedding - e2_embedding)**2,dim=1))
        else:
            return torch.sqrt(torch.sum((e1_embedding + rel_embedding - self.entity_embedding.weight)**2,dim=1))

    def l2_normalize(self):
        self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight, dim=1, p=2)
        self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight, dim=1, p=2)


class TransH(Model):
    def __init__(self,num_entities,num_relations,embedding_dim):
        super(TransH,self).__init__()

        self.entity_embedding = torch.nn.Embedding(num_entities,embedding_dim)
        self.relation_embedding = torch.nn.Embedding(num_relations,embedding_dim)
        self.norm_weight = torch.nn.Embedding(num_relations,embedding_dim)

        self.l2_normalize(only_norm=False)

    def forward(self, e1, rel, e2=None):
        e1_embedding = self.entity_embedding(e1)
        rel_embedding = self.relation_embedding(rel)
        rel_norm_embedding = self.norm_weight(rel)
        e1_relation_projection = projection_transH(e1_embedding, rel_norm_embedding)

        if e2 is not None:
            e2_embedding = self.entity_embedding(e2)
            e2_relation_projection = projection_transH(e2_embedding,rel_norm_embedding)

            return torch.sum((e1_relation_projection + rel_embedding - e2_relation_projection) ** 2, dim=1)
        else:
            e2_relation_projections = projection_transH(self.entity_embedding.weight,rel_norm_embedding)

            return torch.sum((e1_relation_projection + rel_embedding - e2_relation_projections) ** 2, dim=1)

    def l2_normalize(self, only_norm=True):
        if only_norm is False:
            self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight, dim=1, p=2)
            self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight, dim=1, p=2)
        self.norm_weight.weight.data = F.normalize(self.norm_weight.weight, dim=1, p=2)


class TransR(Model):
    def __init__(self,num_entities,num_relations,embedding_dim):
        super(TransR,self).__init__()

        self.entity_embedding = torch.nn.Embedding(num_entities,embedding_dim)
        self.relation_embedding = torch.nn.Embedding(num_relations,embedding_dim)
        self.proj_embedding = torch.nn.Embedding(num_relations,embedding_dim * embedding_dim)
        self.embedding_dim = embedding_dim
        self.l2_normalize()

    def forward(self, e1, rel, e2=None):
        if e2 is not None:
            e1_embedding = self.entity_embedding(e1)
            rel_embedding = self.relation_embedding(rel)
            e2_embedding = self.entity_embedding(e2)
            rel_norm_embedding = self.proj_embedding(rel)

            e1_relation_projection = projection_transR(e1_embedding,rel_norm_embedding,self.embedding_dim)
            e2_relation_projection = projection_transR(e2_embedding,rel_norm_embedding,self.embedding_dim)

            return torch.sum((e1_relation_projection + rel_embedding - e2_relation_projection) ** 2, dim=1)
        else:
            e1_embedding = self.entity_embedding(e1)
            rel_embedding = self.relation_embedding(rel)
            rel_norm_embedding = self.proj_embedding(rel)
            e1_relation_projection = projection_transR(e1_embedding,rel_norm_embedding)
            e2_relation_projections = projection_transR(self.entity_embedding.weight,rel_norm_embedding,self.embedding_dim)

            return torch.sum((e1_relation_projection + rel_embedding - e2_relation_projections) ** 2, dim=1)

    def l2_normalize(self):
        self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight, dim=1, p=2)
        self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight, dim=1, p=2)


class TransD(Model):
    def __init__(self,num_entities,num_relations,embedding_dim):
        super(TransD,self).__init__()

        self.entity_embedding = torch.nn.Embedding(num_entities,embedding_dim)
        self.relation_embedding = torch.nn.Embedding(num_relations,embedding_dim)
        self.entity_proj_embedding = torch.nn.Embedding(num_entities,embedding_dim)
        self.relation_proj_embedding = torch.nn.Embedding(num_relations,embedding_dim)
        self.embedding_dim = embedding_dim
        self.l2_normalize()

    def forward(self, e1, rel, e2=None):
        if e2 is not None:
            e1_embedding = self.entity_embedding(e1)
            rel_embedding = self.relation_embedding(rel)
            e2_embedding = self.entity_embedding(e2)
            e1_proj_embedding = self.entity_proj_embedding(e1)
            rel_proj_embedding = self.relation_proj_embedding(rel)
            e2_proj_embedding = self.entity_proj_embedding(e2)

            e1_relation_projection = projection_transD(e1_embedding,e1_proj_embedding,rel_proj_embedding)
            e2_relation_projection = projection_transD(e2_embedding,e2_proj_embedding,rel_proj_embedding)

            return torch.sum((e1_relation_projection + rel_embedding - e2_relation_projection) ** 2, dim=1)
        else:
            e1_embedding = self.entity_embedding(e1)
            rel_embedding = self.relation_embedding(rel)
            e1_proj_embedding = self.entity_proj_embedding(e1)
            rel_proj_embedding = self.relation_proj_embedding(rel)

            e1_relation_projection = projection_transD(e1_embedding,e1_proj_embedding,rel_proj_embedding)
            e2_relation_projections = projection_transD(self.entity_embedding.weight,self.entity_proj_embedding.weight,rel_proj_embedding)

            return torch.sum((e1_relation_projection + rel_embedding - e2_relation_projections) ** 2, dim=1)

    def l2_normalize(self):
        self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight, dim=1, p=2)
        self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight, dim=1, p=2)
        self.entity_proj_embedding.weight.data = F.normalize(self.entity_proj_embedding.weight, dim=1, p=2)
        self.relation_proj_embedding.weight.data = F.normalize(self.relation_proj_embedding.weight, dim=1, p=2)


class DistMult(Model):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(DistMult, self).__init__()
        self.entity_embedding = torch.nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = torch.nn.Embedding(num_relations, embedding_dim)

        self.l2_normalize()

    def l2_normalize(self):
        self.entity_embedding.weight.data = F.normalize(self.entity_embedding.weight,dim=1,p=2)
        self.relation_embedding.weight.data = F.normalize(self.relation_embedding.weight,dim=1,p=2)

    def forward(self, e1, rel, e2 = None):
        if e2 is not None:
            e1_embedding = self.entity_embedding(e1)
            rel_embedding = self.relation_embedding(rel)
            e2_embedding = self.entity_embedding(e2)

            return torch.sum(e1_embedding * rel_embedding * e2_embedding,dim=1)
        else:
            e1_embedding = self.entity_embedding(e1)
            rel_embedding = self.relation_embedding(rel)
            return torch.sum(e1_embedding * rel_embedding * self.entity_embedding.weight,dim=1)


class ComplEx(Model):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(ComplEx, self).__init__()
        self.entity_embedding_real = torch.nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding_real = torch.nn.Embedding(num_relations, embedding_dim)
        self.entity_embedding_imaginary = torch.nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding_imgatinary = torch.nn.Embedding(num_relations,embedding_dim)

        self.l2_normalize()

    def l2_normalize(self):
        self.entity_embedding_real.weight.data = F.normalize(self.entity_embedding_real.weight,dim=1,p=2)
        self.relation_embedding_real.weight.data = F.normalize(self.relation_embedding_real.weight,dim=1,p=2)
        self.entity_embedding_imaginary.weight.data = F.normalize(self.entity_embedding_imaginary.weight,dim=1,p=2)
        self.relation_embedding_imgatinary.weight.data = F.normalize(self.relation_embedding_imgatinary.weight,dim=1,p=2)

    def forward(self, e1, rel, e2 = None):
        if e2 is not None:
            head_real = self.entity_embedding_real(e1)
            relation_rel = self.relation_embedding_real(rel)
            tail_real = self.entity_embedding_real(e2)
            head_im = self.entity_embedding_imaginary(e1)
            relation_im = self.relation_embedding_imgatinary(rel)
            tail_im = self.entity_embedding_imaginary(e2)

            return self.calculate_score(head_real,head_im,tail_real,tail_im,relation_rel,relation_im)
        else:
            head_real = self.entity_embedding_real(e1)
            relation_rel = self.relation_embedding_real(rel)
            head_im = self.entity_embedding_imaginary(e1)
            relation_im = self.relation_embedding_imgatinary(rel)

            return self.calculate_score(head_real, head_im, self.entity_embedding_real.weight, self.entity_embedding_imaginary.weight, relation_rel, relation_im)


    def calculate_score(self,h_re, h_im, t_re, t_im, r_re, r_im):
        return torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            -1,
        )
