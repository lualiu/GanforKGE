from model.TransX import *
from model.Embedding import *
from model.Generator import *
from model.Discriminator import *

def read_transX_model(model_name,entity_numbers,relation_numbers,embeddingdim):
    if model_name == 'TransE':
        return TransE(entity_numbers,relation_numbers,embeddingdim)
    elif model_name == 'TransH':
        return TransH(entity_numbers, relation_numbers, embeddingdim)
    elif model_name == 'TransR':
        return TransR(entity_numbers,relation_numbers,embeddingdim)
    elif model_name == 'TransD':
        return TransD(entity_numbers, relation_numbers, embeddingdim)
    elif model_name == 'DistMult':
        return DistMult(entity_numbers,relation_numbers,embeddingdim)
    elif model_name == 'Complex':
        return ComplEx(entity_numbers, relation_numbers, embeddingdim)
    else:
        return None

def read_gan_model(Flags,entity_number,relation_number):
    def get_embedding_model(model_name,entity_number,relation_number,embedding_dim):
        if model_name == 'TransE':
            return TransE_E(entity_number,relation_number,embedding_dim)
        elif model_name == 'TransH':
            return TransH_E(entity_number,relation_number,embedding_dim)
        elif model_name == 'TransD':
            return TransD_E(entity_number,relation_number,embedding_dim)
        else:
            return None

    def get_generator_model(model_name,Flags):
        if model_name == 'Translation':
            return Translation_G()
        elif model_name == 'FC':
            return FC_G(Flags.embeddingdim,Flags.hiddenlayers)
        elif model_name == 'ConvE':
            return ConvE_G(Flags.embeddingdim,Flags.numfilter)
        elif model_name == 'ConvTransE':
            return ConvTransE_G(Flags.inputdropout,Flags.featuredropout,Flags.numfilter,Flags.kernelsize,Flags.embeddingdim)
        else:
            return None

    def get_discriminator_model(model_name,Flags):
        if model_name == 'Translation':
            return Translation_D()
        elif model_name == 'FC':
            return FC_D(Flags.embeddingdim,Flags.hiddenlayers)
        elif model_name == 'Conv':
            return Conv_D(Flags.embeddingdim,Flags.numfilter)
        else:
            return None

    return get_embedding_model(Flags.embeddingname,entity_number,relation_number,Flags.embeddingdim),\
           get_generator_model(Flags.gname,Flags), \
           get_discriminator_model(Flags.dname,Flags)
