import torch
import torch.nn.functional as F

def projection_transH(entity, relation_norm):
    return entity - torch.sum(entity * relation_norm, dim=1, keepdim=True) * relation_norm

def projection_transR(entity,proj_matrix,embedding_dim):
    entity = entity.view(-1,embedding_dim,1)
    proj_matrix = proj_matrix.view(-1,embedding_dim,embedding_dim)
    return torch.matmul(proj_matrix,entity).view(-1,embedding_dim)


def projection_transD(entity,entity_proj,relation_proj):
    return F.normalize(entity + torch.sum(entity * entity_proj,dim=1,keepdim=True) * relation_proj,dim=1,p=2)
