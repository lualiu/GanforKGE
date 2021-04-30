import torch
import math
import torch.nn.functional as F
from model.Model import Model
from abc import abstractmethod

class Generator(Model):
    def __init__(self):
        super(Generator,self).__init__()

    @abstractmethod
    def forward(self,e1_embedding, rel_embedding):
        pass


class Translation_G(Generator):
    def __init__(self):
        super(Translation_G,self).__init__()

    def forward(self,e1_embedding, rel_embedding):
        return e1_embedding + rel_embedding


class FC_G(Generator):
    def __init__(self,embedding_dim,hidden_layer_list):
        super(FC_G,self).__init__()

        self.fc_list = torch.nn.ModuleList()
        hidden_layer_list_and_input_output=[]
        hidden_layer_list_and_input_output.append(embedding_dim*2)
        for hidden_layer in hidden_layer_list:
            hidden_layer_list_and_input_output.append(hidden_layer)
        hidden_layer_list_and_input_output.append(embedding_dim)

        for i in range(len(hidden_layer_list_and_input_output)-1):
            self.fc_list.append(torch.nn.Linear(hidden_layer_list_and_input_output[i],hidden_layer_list_and_input_output[i+1]))

    def forward(self,e1_embedding, rel_embedding):
        x = torch.cat([e1_embedding,rel_embedding],dim=1)
        for fc in self.fc_list:
            x = fc(x)
        output = x
        return output


class ConvE_G(Generator):
    def __init__(self,embedding_dim,num_filter):
        super(ConvE_G,self).__init__()

        self.conv = torch.nn.Conv2d(1,num_filter,(1,2))
        self.fc = torch.nn.Linear(num_filter*embedding_dim,embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_filter = num_filter

    def forward(self,e1_embedding, rel_embedding):
        e1_embedding = e1_embedding.view(-1,1,self.embedding_dim,1)
        rel_embedding = rel_embedding.view(-1,1,self.embedding_dim,1)

        x = torch.cat([e1_embedding,rel_embedding],3)
        x = self.conv(x)
        x = F.relu(x)
        x = x.view(-1,self.num_filter * self.embedding_dim)
        x = self.fc(x)
        return x


class ConvTransE_G(Generator):
    def __init__(self,input_drop,feature_map_dropout,output_channel,kernel_size,embedding_dim):
        super(ConvTransE_G, self).__init__()

        self.input_dropout = torch.nn.Dropout(input_drop)
        self.feature_map_dropout = torch.nn.Dropout(feature_map_dropout)

        self.conv1 = torch.nn.Conv1d(2,output_channel,kernel_size,stride=1,padding=int(math.floor(kernel_size/2)),bias=True)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(output_channel)
        self.fc = torch.nn.Linear(output_channel*embedding_dim,embedding_dim,bias=True)

        self.embedding_dim = embedding_dim

    def forward(self,head_embedding,rel_embedding):
        batch_size = head_embedding.size()[0]

        head_embedding = torch.unsqueeze(head_embedding,1)
        rel_embedding = torch.unsqueeze(rel_embedding,1)

        x = torch.cat([head_embedding,rel_embedding],1)
        x = self.bn0(x)
        x = self.input_dropout(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        x = self.feature_map_dropout(x)
        x = x.view(batch_size,-1)
        x = self.fc(x)
        output = x
        return output

