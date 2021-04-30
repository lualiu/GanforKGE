import argparse
import os
from preprocess.TripleClassificationData import TripleClassificationData
from train.TrainTripleClassification import TrainTripleClassifcation
from utils.readmodel import *

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
FLAGS = None

def main(FLAGS):
    data = TripleClassificationData(
        os.path.join(FLAGS.datapath,FLAGS.dataset),
        FLAGS.trainfilename,
        FLAGS.validfilename,
        FLAGS.testfilename,
        FLAGS.withreverse
    )

    embedding, generator, discriminator = read_gan_model(FLAGS, data.entity_numbers,data.relation_numbers)

    if FLAGS.cuda:
        embedding.cuda()
        generator.cuda()
        discriminator.cuda()

    trainGan = TrainTripleClassifcation()
    trainGan.set_data(data)
    trainGan.set_model(embedding,generator,discriminator)

    trainGan.train(
        FLAGS.usepretrained,
        FLAGS.pretrainedpath,
        FLAGS.learningrate,
        FLAGS.weightdecay,
        FLAGS.margin,
        FLAGS.epochs,
        FLAGS.batchsize,
        FLAGS.evaluationtimes,
        FLAGS.savetimes,
        FLAGS.savepath,
        FLAGS.logpath,
        FLAGS.dtuneembedding,
        FLAGS.gtuneembedding,
        FLAGS.dmargintype,
        FLAGS.gusenegative,
        FLAGS.meanorsum,
        print_file = FLAGS.logpath+'.txt'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", default=True, type=bool)

    # parameters for model name
    parser.add_argument("--embeddingname", default='TransE',type=str)
    parser.add_argument("--gname", default='ConvTransE', type=str)
    parser.add_argument("--dname", default='Translation', type=str)

    # parameters for dataset
    parser.add_argument("--datapath",default='data',type=str)
    parser.add_argument("--dataset",default="Wordnet11",type=str)
    parser.add_argument("--trainfilename", default="train.txt", type=str)
    parser.add_argument("--validfilename", default="dev.txt", type=str)
    parser.add_argument("--testfilename", default="test.txt", type=str)

    parser.add_argument("--withreverse", default=False, type=bool)

    # parameters for super parameters
    parser.add_argument("--embeddingdim", default=100, type=int)
    parser.add_argument("--usepretrained", default=False, type=bool)
    parser.add_argument("--pretrainedpath", default='saved_model/TransE/baseline/WN18RR/embedding-model-2000.pkl', type=str)
    parser.add_argument("--learningrate", default=0.001, type=float)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--batchsize", default=1000, type=int)
    parser.add_argument("--margin", default=2.0, type=float)
    parser.add_argument("--weightdecay", default=1e-6, type=float)

    # parameters for save and log times and path
    parser.add_argument("--evaluationtimes", default=100, type=int)
    parser.add_argument("--savetimes", default=500, type=int)
    parser.add_argument("--logtimes", default=1, type=int)
    parser.add_argument("--savepath", default='saved_model/FC_TransE/WN11', type=str)
    parser.add_argument("--logpath", default='log/FC_TransE/WN11', type=str)

    # parameters for fully connected layer
    parser.add_argument("--hiddenlayers",default=[200,100],type=list)

    # parameters for convolutional layer
    parser.add_argument("--numfilter", default=32, type=int)
    parser.add_argument("--inputdropout", default=0.2, type=float)
    parser.add_argument("--featuredropout", default=0.3, type=float)
    parser.add_argument("--kernelsize", default=3, type=int)

    # parameters for different selection strategies for GN and DN
    parser.add_argument("--dtuneembedding", default=True, type=bool)
    parser.add_argument("--gtuneembedding", default=False, type=bool)
    parser.add_argument("--dmargintype", default=True, type=bool)
    parser.add_argument("--gusenegative", default=False, type=bool)
    parser.add_argument("--meanorsum", default='mean', type=str)

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
