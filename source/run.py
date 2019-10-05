import os
import pickle
import numpy as np
import time,random
import argparse, torch,data
from utils import Option
from models import RNNModel, PredictingModel
from experiment import Experiment
from sampling import simulatedAnnealing_batch, testing

def main():

    parser = argparse.ArgumentParser(description="Experiment setup")
    # misc
    parser.add_argument('--seed', default=33, type=int)
    parser.add_argument('--gpu', default="2", type=str)
    parser.add_argument('--no_train', default=False, action="store_true")
    parser.add_argument('--exps_dir', default=None, type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--load', default=None, type=str)

    # data property
    parser.add_argument('--data_path', default='data/quora/quora.txt', type=str)
    parser.add_argument('--dict_path', default='data/quora/dict.pkl', type=str)
    parser.add_argument('--emb_path', default='data/quora/emb.pkl', type=str)
    parser.add_argument('--dict_size', default=30000, type=int)
    parser.add_argument('--vocab_size', default=30004, type=int)
    parser.add_argument('--backward', default=False, action="store_true")
    parser.add_argument('--keyword_pos', default=True, action="store_false")
    parser.add_argument('--pos_path', default='POS/english-models', type=str)

    # model architecture
    parser.add_argument('--num_steps', default=15, type=int)
    parser.add_argument('--num_edits', default=15, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--emb_size', default=300, type=int)
    parser.add_argument('--hidden_size', default=300, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--model', default=0, type=int)
    parser.add_argument('--repeat_size', default=8, type=int)
    parser.add_argument('--topk', default=30, type=int)
    parser.add_argument('--step_reward', default=0.002, type=float)
    parser.add_argument('--prate', default=0.1, type=float)
    # optimization
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.00, type=float)
    parser.add_argument('--clip_norm', default=5, type=float)
    parser.add_argument('--no_cuda', default=False, action="store_true")
    parser.add_argument('--pretrain', default=True, action="store_true")
    parser.add_argument('--threshold', default=0.1, type=float)

    # evaluation
    parser.add_argument('--sim', default='word_max', type=str)
    parser.add_argument('--mode', default='rl', type=str)
    parser.add_argument('--accuracy', default=False, action="store_true")
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--accumulate_step', default=1, type=int)
    parser.add_argument('--backward_path', default=None, type=str)
    parser.add_argument('--forward_path', default=None, type=str)
    parser.add_argument('--uprl_path', default=None, type=str)

    # objective
    parser.add_argument('--mcmc', default='sa', type=str)
    parser.add_argument('--use_data_path', default='data/input/input.txt', type=str)
    parser.add_argument('--reference_path', default=None, type=str)
    parser.add_argument('--max_key', default=3, type=float)
    parser.add_argument('--max_key_rate', default=0.5, type=float)
    parser.add_argument('--rare_since', default=30000, type=int)
    parser.add_argument('--sample_time', default=100, type=int)
    parser.add_argument('--search_size', default=100, type=int)
    parser.add_argument('--action_prob', default=[0.3,0.3,0.3,0.3], type=list)
    parser.add_argument('--just_acc_rate', default=0.0, type=float)
    parser.add_argument('--sim_mode', default='keyword', type=str)
    parser.add_argument('--save_path', default='temp.txt', type=str)
    parser.add_argument('--max_grad_norm', default=5, type=float)
    parser.add_argument('--keep_prob', default=1, type=float)
    parser.add_argument('--C', default=0.05, type=float)
    parser.add_argument('--M_kw', default=2, type=float)
    parser.add_argument('--M_flu', default=2, type=float)
    parser.add_argument('--M_div', default=1, type=float)
    parser.add_argument('--M_bleu', default=1, type=float)

    d = vars(parser.parse_args())
    option = Option(d)
    random.seed(option.seed)
    np.random.seed(option.seed)
    torch.manual_seed(option.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu

    if option.exp_name is None:
      option.tag = time.strftime("%y-%m-%d-%H-%M")
    else:
      option.tag = option.exp_name  
    if option.accuracy:
      assert option.top_k == 1
    
    dataclass = data.Data(option)       
    print("Data prepared.")

    option.this_expsdir = os.path.join(option.exps_dir, option.tag)
    if not os.path.exists(option.this_expsdir):
        os.makedirs(option.this_expsdir)
    option.ckpt_dir = os.path.join(option.this_expsdir, "ckpt")
    if not os.path.exists(option.ckpt_dir):
        os.makedirs(option.ckpt_dir)
    option.model_path = os.path.join(option.ckpt_dir, "model")
    
    option.save()
    print("Option saved.")

    device = torch.device("cuda" if torch.cuda.is_available() and not option.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    learner = RNNModel(option)
    learner.to(device)

    if option.load is  not None: 
        with open(option.load, 'rb') as f:
            learner.load_state_dict(torch.load(f))

    if option.mode=='lm':
        experiment = Experiment(option, learner=learner, data=dataclass)
        print("Experiment created.")
        if option.pretrain:
            experiment.init_embedding(option.emb_path)
        print("Start training...")
        experiment.train()
    elif option.mode == 'rl':
        forwardmodel = RNNModel(option).cuda()
        backwardmodel = RNNModel(option).cuda()
        embmodel = RNNModel(option).cuda()
        if option.forward_path is  not None: 
            with open(option.forward_path, 'rb') as f:
                forwardmodel.load_state_dict(torch.load(f))

        if option.backward_path is  not None: 
            with open(option.backward_path, 'rb') as f:
                backwardmodel.load_state_dict(torch.load(f))

        with open(option.emb_path, 'rb') as f: 
            print('xxx')
            wordvec,embedding = pickle.load(f, encoding = 'latin1')
        for key, embs in wordvec.items():
            if key in dataclass.vocab:
                id = dataclass.vocab[key]
                embmodel.encoder.weight.data[id,:] = torch.tensor(wordvec[key],dtype=torch.float).cuda()
            else:
                print(key)
        
        print('xxx')
        simulatedAnnealing_batch(option, dataclass, forwardmodel, backwardmodel, embmodel)
        print("="*36 + "Finish" + "="*36)
    elif option.mode == 'rl-test':
        forwardmodel = RNNModel(option).cuda()
        backwardmodel = RNNModel(option).cuda()
        embmodel = RNNModel(option).cuda()
        if option.forward_path is  not None: 
            with open(option.forward_path, 'rb') as f:
                forwardmodel.load_state_dict(torch.load(f))

        if option.backward_path is  not None: 
            with open(option.backward_path, 'rb') as f:
                backwardmodel.load_state_dict(torch.load(f))



        testing(option, dataclass, forwardmodel, backwardmodel, embmodel)
        print("="*36 + "Finish" + "="*36)




if __name__ == "__main__":
    main()

