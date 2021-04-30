from abc import abstractmethod,ABCMeta

class TrainBaseClass(object,metaclass=ABCMeta):
    def set_data(self,data):
        self.data = data

    @abstractmethod
    def set_model(self, *input):
        pass

    @abstractmethod
    def train(self, *input):
        pass