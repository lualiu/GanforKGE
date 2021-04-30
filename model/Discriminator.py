import torch
import torch.nn.functional as F
from model.Model import Model
from abc import abstractmethod

class Discriminator(Model):
    def __init__(self):
        super(Discriminator,self).__init__()

    @abstractmethod
    def forward(self,e1_embedding, rel_embedding, e2_embedding):
        pass


class Translation_D(Discriminator):
    def __init__(self):
        super(Translation_D,self).__init__()

    def forward(self,e1_embedding, rel_embedding, e2_embedding):
        return torch.sqrt(torch.sum((e1_embedding + rel_embedding - e2_embedding) ** 2, dim=1))


class FC_D(Discriminator):
    def __init__(self,embedding_dim,hidden_layer_list):
        super(FC_D,self).__init__()

        self.fc_list = torch.nn.ModuleList()
        hidden_layer_list_and_input_output = []
        hidden_layer_list_and_input_output.append(embedding_dim*3)
        for hidden_layer in hidden_layer_list:
            hidden_layer_list_and_input_output.append(hidden_layer)
            hidden_layer_list_and_input_output.append(1)

        for i in range(len(hidden_layer_list_and_input_output)-1):
            self.fc_list.append(torch.nn.Linear(hidden_layer_list_and_input_output[i],hidden_layer_list_and_input_output[i+1]))

    def forward(self,e1_embedding, rel_embedding, e2_embedding):
        x = torch.cat([e1_embedding,rel_embedding,e2_embedding],dim=1)
        for fc in self.fc_list:
            x = fc(x)
        output = x
        return output

class Conv_D(Discriminator):
    def __init__(self,embedding_dim,num_filter):
        super(Conv_D,self).__init__()

        self.conv = torch.nn.Conv2d(1,num_filter,(1,3))
        self.fc = torch.nn.Linear(num_filter*embedding_dim,1)
        self.embedding_dim = embedding_dim
        self.num_filter = num_filter


    def forward(self,e1_embedding, rel_embedding, e2_embedding):
        e1_embedding = e1_embedding.view(-1,1,self.embedding_dim,1)
        rel_embedding = rel_embedding.view(-1,1,self.embedding_dim,1)
        e2_embedding = e2_embedding.view(-1,1,self.embedding_dim,1)

        x = torch.cat([e1_embedding,rel_embedding,e2_embedding],3)
        x = self.conv(x)
        x = F.relu(x)
        x = x.view(-1,self.num_filter * self.embedding_dim)
        x = self.fc(x)
        return x
