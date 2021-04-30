import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def init(self,saved_model_path=None):
        if saved_model_path is not None:
            print('loading pretrained from ' + saved_model_path)
            pretrained_dict = torch.load(saved_model_path)
            model_dict = self.state_dict()

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            print(pretrained_dict.keys())
            model_dict.update(pretrained_dict)
            self.load_state_dict(pretrained_dict)
        else:
            torch.nn.init.uniform_(self.entity_embedding.weight.data, a=-self.bound, b=self.bound)
            torch.nn.init.uniform_(self.relation_embedding.weight.data, a=-self.bound, b=self.bound)
