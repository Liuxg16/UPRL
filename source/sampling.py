from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, math
import torch.optim as optim
from copy import copy
import time, random
import numpy as np
import pickle as pkl
from utils import *
from models import *
import data, RAKE
from data import array_data
from zpar import ZPar

def simulatedAnnealing_batch(option, dataclass, forwardmodel = None, backwardmodel=None,
        embmodel=None):
    option = option
    similarityfun = similarity_keyword_bleu_tensor

    device = torch.device("cuda" if torch.cuda.is_available() and not option.no_cuda else "cpu")
    agent = UPRL_LM(option)
    agent.to(device)
    if option.uprl_path is not None:
        with open(option.uprl_path, 'rb') as f:
            agent.load_state_dict(torch.load(f))

    fileobj = open(option.emb_path,'r')
    emb_word,emb_id=pkl.load(StrToBytes(fileobj), encoding='latin1')
    fileobj.close()
    sim=option.sim

    use_data, sta_vec_list = read_data_use(option, dataclass.sen2id)
    id2sen = dataclass.id2sen
    generateset = []
    optimizer = optim.Adadelta(agent.parameters(),lr = option.learning_rate)
   
    batch_size = option.batch_size
    maxvalue = 0
    avg_rewards = []
    print('number of samples ', use_data.length) 
    intervals = 10
    for ba in range(0,1000000):
        if ba % intervals == 0:
            agent.eval()
        else:
            agent.train()

        avg_rewards = []
        for i in range(0,100):
            sen_id = (ba*100+i)% (use_data.length//batch_size)
            sta_vec=sta_vec_list[sen_id*batch_size:sen_id*batch_size+batch_size]
            sta_vec = np.array(sta_vec)
            inp, sequence_length, _=use_data(batch_size, sen_id)
            assert len(inp)==len(sequence_length)
            batch_size = len(inp)

            input = torch.tensor(inp).long().view(1, batch_size, -1).repeat(option.repeat_size,1,1)
            poskeys = torch.tensor(sta_vec).float().view(1, batch_size, -1).repeat(option.repeat_size,1,1)
            sequence_length = torch.tensor(sequence_length).long().view(1, batch_size).repeat(option.repeat_size,1)

            input = input.view(option.repeat_size*batch_size,-1).to(device)
            poskeys = poskeys.view(option.repeat_size*batch_size,-1).to(device)
            sequence_length = sequence_length.view(option.repeat_size*batch_size,-1).to(device)

            loss, rewards, st , temp = agent(input, poskeys, sequence_length, forwardmodel,
                    backwardmodel, embmodel) # bs,15; bs,steps
            loss = torch.mean(loss)
            if i  == 99:
                st = st.view(option.repeat_size,batch_size, -1)
                rewards = rewards.view(option.repeat_size, batch_size)
                temp = temp.view(option.repeat_size, batch_size).detach()
                print(' '.join(id2sen(inp[1])))
                print(inp[1])
                print('key words ', sta_vec[1])
                print('generated:  '+' '.join(id2sen(st.cpu().numpy()[2,1])))
                print('generated:  ', st.cpu().numpy()[2,1])
                print(rewards.cpu().numpy()[2,1])
                print(temp.cpu().numpy()[2,1])
                
                
                sources = ' '.join(id2sen(inp[0]))
                targetss = ' '.join(id2sen(st.cpu().numpy()[2,0]))
                rewardss= 'reward '+'{} '.format(rewards.cpu().numpy()[2,0])
                appendtext(sources, option.model_path+'log.log')
                appendtext(targetss, option.model_path+'log.log')
                appendtext(rewardss, option.model_path+'log.log')

            a_rewards =  torch.mean(rewards).item()
            avg_rewards.append(a_rewards)
            if agent.training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1)
                optimizer.step()

        if agent.training:
            trainstr = 'Training: {}, avg reward {}, loss {}'.format(ba, np.mean(avg_rewards),loss.item())
        else:
            trainstr = 'Testing: {}, avg reward {}, loss {}'.format(ba, np.mean(avg_rewards),loss.item())

        appendtext(trainstr, option.model_path+'log.log')
        
        avg_R = np.mean(avg_rewards)
        if not agent.training and avg_R>maxvalue:
            maxvalue = avg_R.item()
            with open(option.model_path+'-best.pkl', 'wb') as f:
                torch.save(agent.state_dict(), f)



def testing(option, dataclass, forwardmodel = None, backwardmodel=None):
    option = option
    similarityfun = similarity_keyword_bleu_tensor

    device = torch.device("cuda" if torch.cuda.is_available() and not option.no_cuda else "cpu")
    agent = UPRL_LM(option)
    agent.to(device)
    if option.uprl_path is not None:
        with open(option.uprl_path, 'rb') as f:
            agent.load_state_dict(torch.load(f))

    fileobj = open(option.emb_path,'r')
    emb_word,emb_id=pkl.load(StrToBytes(fileobj), encoding='latin1')
    fileobj.close()
    sim=option.sim
    sta_vec=list(np.zeros([option.num_steps-1]))

    use_data, sta_vec_list = read_data_use(option, dataclass.sen2id)
    id2sen = dataclass.id2sen
    generateset = []
    optimizer = optim.Adadelta(agent.parameters(),lr = 1)
   
    batch_size = option.batch_size
    maxvalue = 0
    avg_rewards = []
    nums = use_data.length//batch_size
    print(nums)
    for i in range(nums):
        sen_id = i% (use_data.length//batch_size)
        sta_vec=sta_vec_list[sen_id*batch_size:sen_id*batch_size+batch_size]
        sta_vec = np.array(sta_vec)
        inp, sequence_length, _=use_data(batch_size, sen_id)
        assert len(inp)==len(sequence_length)
        batch_size = len(inp)

        agent.eval()
        input = torch.tensor(inp).long().view(1, batch_size, -1).repeat(option.repeat_size,1,1)
        poskeys = torch.tensor(sta_vec).float().view(1, batch_size, -1).repeat(option.repeat_size,1,1)
        sequence_length = torch.tensor(sequence_length).long().view(1, batch_size).repeat(option.repeat_size,1)

        input = input.view(option.repeat_size*batch_size,-1).to(device)
        poskeys = poskeys.view(option.repeat_size*batch_size,-1).to(device)
        sequence_length = sequence_length.view(option.repeat_size*batch_size,-1).to(device)

        loss, rewards, st , temp = agent(input, poskeys, sequence_length, forwardmodel,
                backwardmodel, id2sen) # bs,15; bs,steps

        st = st.view(option.repeat_size,batch_size, -1).cpu().numpy()
        if False:
            rewards = rewards.view(option.repeat_size, batch_size)
            temp = temp.view(option.repeat_size, batch_size).detach()
            print(' '.join(id2sen(inp[1])))
            print(inp[1])
            print('length ', sequence_length[1])
            print('key words ', sta_vec[1])
            print('generated:  '+' '.join(id2sen(st.cpu().numpy()[2,1])))
            print('generated:  ', st.cpu().numpy()[2,1])
            print('reward', rewards.cpu().numpy()[2,1])
            print('temp', temp.cpu().numpy()[2,1])
        for j in range(batch_size):
            appendtext(' '.join(id2sen(st[0,j], True)), option.save_path)
