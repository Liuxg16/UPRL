import sys
import os
import time
import pickle
from collections import Counter
import numpy as np
import torch
import utils
import torch.nn as nn

class Experiment():
    """
    This class handles all experiments related activties, 
    including training, testing, early stop, and visualize
    results, such as get attentions and get rules. 

    Args:
        sess: a TensorFlow session 
        saver: a TensorFlow saver
        option: an Option object that contains hyper parameters
        learner: an inductive learner that can  
                 update its parameters and perform inference.
        data: a Data object that can be used to obtain 
              num_batch_train/valid/test,
              next_train/valid/test,
              and a parser for get rules.
    """
    
    def __init__(self, option, learner=None, data=None):
        self.option = option
        self.learner = learner
        self.data = data
        self.vocab = self.data.vocab
        # helpers
        self.msg_with_time = lambda msg: \
                "%s Time elapsed %0.2f hrs (%0.1f mins)" \
                % (msg, (time.time() - self.start) / 3600., 
                        (time.time() - self.start) / 60.)

        self.start = time.time()
        self.epoch = 0
        self.best_valid_loss = np.inf
        self.best_valid_in_top = 0.
        self.train_stats = []
        self.valid_stats = []
        self.test_stats = []
        self.early_stopped = False
        # self.log_file = open(os.path.join(self.option.this_expsdir, "log.txt"), "w")

        self.write_log_file("-----------has built a model---------\n")

        param_optimizer = list(self.learner.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        # t_total = -1 # self.data.num_batch_train
        # self.optimizer = BertAdam(optimizer_grouped_parameters,
        #                  lr= self.option.learning_rate,
        #                  warmup= 0.01,
        #                  t_total=t_total)
        self.optimizer = torch.optim.Adam(self.learner.parameters(), self.option.learning_rate,\
                weight_decay = self.option.weight_decay)

        self.device = torch.device("cuda" if torch.cuda.is_available() and not option.no_cuda else "cpu")

    def init_embedding(self, filename):
        with open(filename, 'rb') as f: 
            print('xxx')
            wordvec,embedding = pickle.load(f, encoding = 'latin1')
        for key, embs in wordvec.items():
            if key in self.vocab:
                id = self.vocab[key]
                self.learner.encoder.weight.data[id,:] = torch.tensor(wordvec[key],dtype=torch.float).cuda()
            else:
                print(key)



    def one_epoch(self, mode, num_batch, next_fn):
        num_batch = int(num_batch)
        epoch_loss = [0]
        epoch_in_top = [0]
        self.optimizer.zero_grad()
        for batch in range(num_batch):
            data, lengths, target= next_fn() # query(relation), head(target), tails
            data = data.to(self.device)
            target = target.to(self.device)

            if mode == "train":
                loss, acc, outputs = self.learner(data, target)

                loss.backward()
                if self.option.clip_norm>0: 
                    torch.nn.utils.clip_grad_norm(self.learner.parameters(),self.option.clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                loss, acc, outputs = self.learner(data, target)
            epoch_loss += [loss.item()]
            epoch_in_top += [acc.item()]
            # msg = 'intop:{}'.format(np.mean(epoch_loss))
            # print msg
            # self.write_log_file(msg)


        msg = self.msg_with_time(
                "Epoch %d mode %s Loss %0.4f In top %0.4f." 
                % (self.epoch+1, mode, np.mean(epoch_loss), np.mean(epoch_in_top)))
        print(msg)
        self.write_log_file(msg)
        # self.log_file.write(msg + "\n")
        return epoch_loss, epoch_in_top

    def one_epoch_train(self):
        self.learner.train()
        loss, in_top = self.one_epoch("train", 
                                      self.data.train_data.length/self.option.batch_size, 
                                      self.data.next_train)

        
        self.train_stats.append([loss, in_top])
        
    def one_epoch_valid(self):
        self.learner.eval()
        loss, in_top = self.one_epoch("valid", 
                                      self.data.valid_data.length/self.option.batch_size, 
                                      self.data.next_valid)
        self.valid_stats.append([loss, in_top])
        self.best_valid_loss = min(self.best_valid_loss, np.mean(loss))
        self.best_valid_in_top = max(self.best_valid_in_top, np.mean(in_top))

    def one_epoch_test(self):
        self.learner.eval()
        loss, in_top = self.one_epoch("test", 
                                      self.data.test_data.length/self.option.batch_size,
                                      self.data.next_test)
        self.test_stats.append([loss, in_top])
    
    def early_stop(self):
        loss_improve = self.best_valid_loss == np.mean(self.valid_stats[-1][0])
        in_top_improve = self.best_valid_in_top == np.mean(self.valid_stats[-1][1])


        # if in_top_improve:
        if loss_improve:
            self.write_log_file('-------------best-------------')
            with open(self.option.model_path+'-best.pkl', 'wb') as f:
                torch.save(self.learner.state_dict(), f)
            return False

        else:
            if self.epoch > self.option.epochs:
                return True
            else:
                return False

    def train(self):
        while (self.epoch < self.option.epochs and not self.early_stopped):
            self.one_epoch_train()
            self.one_epoch_valid()
            self.one_epoch_test()
            self.epoch += 1

            if self.early_stop():
                self.early_stopped = True
                print("Early stopped at epoch %d" % (self.epoch))
        
        all_test_in_top = [np.mean(x[1]) for x in self.test_stats]
        best_test_epoch = np.argmax(all_test_in_top)
        best_test = all_test_in_top[best_test_epoch]
        
        msg = "Best test in top: %0.4f at epoch %d." % (best_test, best_test_epoch + 1)       
        print(msg)
        self.write_log_file(msg + "\n")
        pickle.dump([self.train_stats, self.valid_stats, self.test_stats],
                    open(os.path.join(self.option.this_expsdir, "results.pckl"), "w"))


    def get_vocab_embedding(self):
        vocab_embedding = self.learner.get_vocab_embedding(self.sess)
        msg = self.msg_with_time("Vocabulary embedding retrieved.")
        print(msg)
        self.log_file.write(msg + "\n")
        
        vocab_embed_file = os.path.join(self.option.this_expsdir, "vocab_embed.pckl")
        pickle.dump({"embedding": vocab_embedding, "labels": self.data.query_vocab_to_number}, open(vocab_embed_file, "w"))
        msg = self.msg_with_time("Vocabulary embedding stored.")
        print(msg)
        self.log_file.write(msg + "\n")

    def close_log_file(self):
        self.log_file.close()

    def write_log_file(self, string):
        self.log_file = open(os.path.join(self.option.this_expsdir, "log.txt"), "a+")
        self.log_file.write(string+ "\n")
        self.log_file.close()

