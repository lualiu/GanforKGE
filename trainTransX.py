import argparse
import os
from preprocess.KBNegativeSampleData import KBNegativeSampleData
from train.TrainTransX import TrainTransX
from utils.readmodel import read_transX_model

os.environ["CUDA_VISIBLE_DEVICES"] = '5'
FLAGS = None

def main(FLAGS):
    data = KBNegativeSampleData(
        os.path.join(FLAGS.datapath,FLAGS.dataset),
        FLAGS.trainfilename,
        FLAGS.validfilename,
        FLAGS.testfilename,
        FLAGS.withreverse
    )
    print(data.entity_numbers)
    print(data.relation_numbers)
    model = read_transX_model(
        FLAGS.modelname,
        data.entity_numbers,
        data.relation_numbers,
        FLAGS.embeddingdim
    )
    if FLAGS.cuda:
        model.cuda()

    trainTransX = TrainTransX()
    trainTransX.set_data(data)
    trainTransX.set_model(model)

    trainTransX.train(
        FLAGS.usepretrained,
        FLAGS.pretrainedpath,
        FLAGS.learningrate,
        FLAGS.weightdecay,
        FLAGS.epochs,
        FLAGS.batchsize,
        FLAGS.margin,
        FLAGS.evaluationtimes,
        FLAGS.savetimes,
        os.path.join(os.path.join('saved_model',FLAGS.modelname),FLAGS.dataset),
        os.path.join(os.path.join('log', FLAGS.modelname),FLAGS.dataset),
        os.path.join(os.path.join('log', FLAGS.modelname), FLAGS.dataset+'.txt')
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", default=True, type=bool)

    parser.add_argument("--modelname", default='TransH',type=str)

    parser.add_argument("--datapath",default='data',type=str)
    parser.add_argument("--dataset",default="FB15k-237",type=str)
    parser.add_argument("--trainfilename", default="train.txt", type=str)
    parser.add_argument("--validfilename", default="valid.txt", type=str)
    parser.add_argument("--testfilename", default="test.txt", type=str)

    parser.add_argument("--embeddingdim", default=100, type=int)
    parser.add_argument("--usepretrained", default=False, type=bool)
    parser.add_argument("--pretrainedpath", default='saved_model/TransE/baseline/WN18RR/embedding-model-2000.pkl', type=str)
    parser.add_argument("--learningrate", default=0.001, type=float)
    parser.add_argument("--epochs", default=2000, type=int)
    parser.add_argument("--batchsize", default=5000, type=int)
    parser.add_argument("--margin", default=1.0, type=float)
    parser.add_argument("--weightdecay", default=1e-5, type=float)

    parser.add_argument("--evaluationtimes", default=100, type=int)
    parser.add_argument("--savetimes", default=500, type=int)
    parser.add_argument("--logtimes", default=1, type=int)

    parser.add_argument("--withreverse", default=True, type=bool)

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)