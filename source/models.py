import torch, pickle
import torch.nn as nn
import numpy as np
import time, random
from utils import *

def reverse_input(input, sequence_length):
    batch_size=input.shape[0]
    num_steps=input.shape[1]
    input_new= torch.clone(input)
    for i in range(batch_size):
        length=sequence_length[i]-2
        for j in range(length):
            input_new[i][j+1]=input[i][length-j]
    return input_new

class RNNModel(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self, option):
		super(RNNModel, self).__init__()
		rnn_type = 'LSTM'
		self.length = 15
		self.option = option
		dropout = option.dropout
		ntoken = option.vocab_size
		self.vocab_size = option.vocab_size
		ninp = option.emb_size
		nhid = option.hidden_size
		self.nlayers = option.num_layers
		self.drop = nn.Dropout(dropout)
		self.encoder = nn.Embedding(ntoken, ninp)
		self.rnn = nn.LSTM(ninp, nhid, self.nlayers, dropout = dropout ,batch_first=True)
		self.decoder = nn.Linear(nhid, ntoken)
		self.init_weights()
		self.rnn_type = rnn_type
		self.nhid = nhid
		self.ntoken = ntoken
		self.criterion = nn.CrossEntropyLoss()
		self.device = torch.device("cuda" if torch.cuda.is_available() and not self.option.no_cuda else "cpu")

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, input, target):
		'''
		bs,15; bs,15
		'''
		batch_size = input.size(0)
		length = input.size(1)
		target = target.view(-1)
		emb = self.drop(self.encoder(input))
		c0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(self.device)
		h0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(self.device)
		output, hidden = self.rnn(emb, (c0,h0))
		output = self.drop(output).contiguous().view(batch_size*length,-1)
		decoded = self.decoder(output)
		loss = self.criterion(decoded, target)
		v,idx = torch.max(decoded,1)
		acc = torch.mean(torch.eq(idx,target).float())
		return loss,acc, decoded.view(batch_size, length, self.ntoken)

	def predict(self, input, s0, pos, sequence_length, forwardflag = True, topk = 20):
		'''
		bs,15; bs,15
		'''
		K = topk-self.length
		batch_size = input.size(0)
		length = input.size(1)
		valid_id = torch.gt(sequence_length-1,pos)
		if forwardflag:
			emb = self.encoder(input)
			c0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(self.device)
			h0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(self.device)
			output, hidden = self.rnn(emb, (c0,h0))
			decoded = nn.Softmax(2)(self.decoder(output)) # bs,v
			decoded[:,:,30000] = 0
			prob, word = torch.topk(decoded[:,pos-1,:],K)
			word = torch.cat([word, s0],1)
			ins_word = word.clone()
		else:
			input = reverse_input(input.cpu(), sequence_length.cpu()).to(self.device)
			emb = self.encoder(input)
			c0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(self.device)
			h0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(self.device)
			output, hidden = self.rnn(emb, (c0,h0))
			decoded = nn.Softmax(2)(self.decoder(output)) # bs,l,v
			decoded[:,:,30000] = 0

			revse_id =valid_id.long()*(sequence_length-pos-2)
			ids = torch.zeros(batch_size, self.length).to(self.device)
			ones = torch.ones(batch_size,1).to(self.device)
			ids = ids.scatter(1, revse_id,ones)
			decoded_column = decoded[ids>0.5]
			prob, word = torch.topk(decoded_column,K)
			word = torch.cat([word, s0],1)
            #for inserion
			revse_id =valid_id.long()*(sequence_length-pos-1)
			ids = torch.zeros(batch_size, self.length).to(self.device)
			ones = torch.ones(batch_size,1).to(self.device)
			ids = ids.scatter(1, revse_id,ones)
			decoded = decoded[ids>0.5]
			prob, ins_word = torch.topk(decoded,K)
			ins_word = torch.cat([ins_word, s0],1)


		return word, ins_word, valid_id

	def prod_prob(self, input):
		'''
		bs,l
		'''
		batch_size = input.size(0)
		length = input.size(1)

		emb = self.encoder(input)
		c0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(self.device)
		h0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(self.device)
		output, hidden = self.rnn(emb, (c0,h0))
		decoded = nn.Softmax(2)(self.decoder(output)) # bs,l,v
		padding = torch.ones(batch_size,1)*30003
		target = torch.cat([input[:,1:], padding.long().to(self.device)],1)
		probs = torch.gather(decoded,2, target.unsqueeze(2)) # probs
		return probs

class PredictingModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, option):
        super(PredictingModel, self).__init__()
        rnn_type = 'LSTM'
        self.option = option
        dropout = option.dropout
        ntoken = option.vocab_size+1
        ninp = option.emb_size
        nhid = option.hidden_size
        self.nlayers = option.num_layers
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, self.nlayers, dropout = dropout ,batch_first=True,\
                bidirectional=True )
        self.decoder = nn.Linear(nhid*2, ntoken)
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.ntoken = ntoken
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.option.no_cuda else "cpu")

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, target):
        '''
        bs,15; bs,15
        '''
        batch_size = input.size(0)
        length = input.size(1)

        ind = int(random.random()*length)

        target = input[:,ind].clone()
        input[:,ind] = self.ntoken-1

        emb = self.drop(self.encoder(input))
        c0 = torch.zeros(2*self.nlayers, batch_size, self.nhid).to(self.device)
        h0 = torch.zeros(2*self.nlayers, batch_size, self.nhid).to(self.device)
        output, hidden = self.rnn(emb, (c0,h0)) # batch, length,2h
        pooled = nn.MaxPool1d(length)(output.permute(0,2,1)) #batch,2h
        decoded = self.decoder(pooled.squeeze(2))
        loss = self.criterion(decoded, target)
        v,idx = torch.max(decoded,1)
        acc = torch.mean(torch.eq(idx,target).float())
        return loss,acc, decoded.view(batch_size, self.ntoken)

    def predict(self, input):
        '''
        bs,15; bs,15
        '''
        batch_size = input.size(0)
        length = input.size(1)

        emb = self.encoder(input)
        c0 = torch.zeros(2*self.nlayers, batch_size, self.nhid).to(self.device)
        h0 = torch.zeros(2*self.nlayers, batch_size, self.nhid).to(self.device)
        output, hidden = self.rnn(emb, (c0,h0)) # batch, length,2h
        pooled = nn.MaxPool1d(length)(output.permute(0,2,1)) #batch,2h
        decoded = nn.Softmax(1)(self.decoder(pooled.squeeze(2)))
        return decoded



    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
 
class UPRL1(nn.Module):
    def __init__(self, option):
        super(UPRL, self).__init__()
        rnn_type = 'LSTM'
        self.option = option
        dropout = option.dropout
        ntoken = option.vocab_size
        self.vocab_size = option.vocab_size
        ninp = option.emb_size
        nhid = option.hidden_size
        self.nlayers = option.num_layers
        self.drop = nn.Dropout(dropout)
        self.embedding= nn.Embedding(ntoken, ninp)

        self.rnn = nn.LSTM(ninp*2+2, nhid, self.nlayers, dropout = dropout ,batch_first=True,\
                bidirectional=True )

        self.repeat_size = 10
        self.decoder = nn.Linear(2*nhid, ntoken+1)
        self.n_token = ntoken
        self.nhid = nhid
        self.ntoken = ntoken
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.option.no_cuda else "cpu")

    def forward(self, input, key_pos, id2sen, emb_word):
        '''
        bs,15; bs,15
        '''

        print('====================')
        print(' '.join(id2sen(input[0])))
        print(key_pos)

        length = 15
        input = torch.tensor(input).long().view(1,length).repeat(self.repeat_size,1)
        key_pos_ = [key_pos for i in range(self.repeat_size)]
        key_pos = torch.tensor(key_pos_).float().view(self.repeat_size,length,1)
        N_step = 10
        st = input
        s0 = input
        N_step = 10
        pis = torch.zeros(self.repeat_size,N_step)
        actions = torch.zeros(self.repeat_size,N_step)
        rewards = torch.zeros(self.repeat_size,N_step)
        batch_size = self.repeat_size
        c0 = torch.zeros(2*self.nlayers, batch_size, self.nhid)
        h0 = torch.zeros(2*self.nlayers, batch_size, self.nhid)

        
        scores = self.f(st,s0, key_pos_,id2sen, emb_word)
        for i in range(N_step): 
            pos = i% length
            st,pi = self.step(st, s0, key_pos,pos, length,c0,h0)
            score_new = self.f(st, s0, key_pos_,id2sen, emb_word)
            reward  = score_new-scores
            
            #print(' '.join(id2sen(st[0].tolist())))
            #print(score_new, reward)

            pis[:,i:i+1] = pi
            rewards[:,i:i+1] = reward
            scores = score_new
            
        total_r = torch.sum(rewards,1)
        inc_flag = torch.gt(total_r, torch.mean(total_r)).float()
        rlloss =  -torch.log(pis.clamp(1e-6,1)) *rewards
        self.loss = torch.mean(rlloss,1)*inc_flag
        print(total_r, inc_flag)
        avg_rewards = torch.mean(rewards)
        return torch.mean(self.loss), avg_rewards
     

    def step(self, s_t_1, s0, key_pos, pos, length,c0,h0):
        # bs,L
        batch_size = key_pos.size(0)
        #print(s_t_1.size(), s0.size(), key_pos.size())
        pos_tensor = torch.zeros(self.repeat_size,length,1)
        pos_tensor[:,pos,:] = 1
        embt = self.embedding(s_t_1)
        emb0 = self.embedding(s0)
        emb = torch.cat([embt,emb0,pos_tensor,key_pos],2)
        output, hidden = self.rnn(emb, (c0,h0)) # batch, length,2h
        pooled = nn.MaxPool1d(length)(output.permute(0,2,1)) #batch,2h
        decoded = nn.Softmax(1)(self.decoder(pooled.squeeze(2))) # bs,V
        if self.training:
            action = decoded.multinomial(1)
        else: 
            values,action = torch.max(decoded,1)
        pi = torch.gather(decoded, 1, action) # (b_s,1), \pi for i,j
        
        replaceflag = torch.lt(action,self.n_token).long()
        st = torch.clone(s_t_1)
        st[:,pos:pos+1] = action*replaceflag +  (1-replaceflag)*s_t_1[:,pos:pos+1]
        return st, pi
    
    def f(self, st,s0, key_pos, id2sen, emb_word):
        xt = st.tolist()
        x0 = s0.tolist()
        sims =  similarity_batch(xt, x0, key_pos, id2sen, emb_word, self.option)
        return torch.tensor(sims,dtype=torch.float)

    def init_hidden(self, bsz):
    	weight = next(self.parameters())
    	if self.rnn_type == 'LSTM':
    		return (weight.new_zeros(self.nlayers, bsz, self.nhid),
    				weight.new_zeros(self.nlayers, bsz, self.nhid))
    	else:
    		return weight.new_zeros(self.nlayers, bsz, self.nhid)
  
class UPRL(nn.Module):
    def __init__(self, option):
        super(UPRL, self).__init__()
        rnn_type = 'LSTM'
        self.option = option
        self.length = 15
        dropout = option.dropout
        self.prate = 0.15
        self.vocab_size = option.vocab_size
        ninp = option.emb_size
        nhid = option.hidden_size
        self.nlayers = option.num_layers
        self.drop = nn.Dropout(dropout)
        self.embedding= nn.Embedding(self.vocab_size, ninp)

        self.rnn = nn.LSTM(ninp*2+1, nhid, self.nlayers, dropout = dropout ,batch_first=True,\
                bidirectional=True )

        self.repeat_size = option.repeat_size
        self.num_actions = 2*self.vocab_size+2
        self.decoder = nn.Linear(2*nhid+1+self.length, self.num_actions)
        self.nhid = nhid
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.option.no_cuda else "cpu")

    def forward(self, input, key_pos, sequence_length, forwardmodel=None):
        '''
        bs,15; bs,14
        '''

        N_step = 15
        self.batch_size = input.size(0)
        st = input.clone()
        s0 = input.clone()
        emb0 = self.embedding(s0)
        length_t = sequence_length.clone()
        pis = torch.zeros(self.batch_size,N_step).to(self.device)
        actions = torch.zeros(self.batch_size,N_step).to(self.device)
        c0 = torch.zeros(2*self.nlayers, self.batch_size, self.nhid).to(self.device)
        h0 = torch.zeros(2*self.nlayers, self.batch_size, self.nhid).to(self.device)
        self.switch_m = torch.FloatTensor(self.batch_size, 2).fill_(self.prate).to(self.device)
        self.switch_m[:,1] = 1- self.prate
        self.action_m = torch.FloatTensor(self.batch_size, self.num_actions).fill_(1.0/self.num_actions).to(self.device)


        for i in range(N_step): 
            pos = i% self.length
            st,pi, action, length_t , re= self.step(st, emb0, key_pos, pos, length_t,c0,h0)
            pis[:,i:i+1] = pi
            actions[:,i:i+1] = re

        step_reward = torch.sum(torch.eq(actions, 2*self.vocab_size+1).float(),1) * 0.01 # hold
        fv, sim, div, flu  = self.f(st,s0, key_pos, forwardmodel, length_t)
        reward = fv + step_reward

        reward = torch.sum(actions,1)*0.1
        reward = reward.detach() # bs,1
        reward =  reward.view(self.repeat_size, -1) 
        reward_adjust = nn.ReLU()(reward- torch.mean(reward,0, keepdim = True)) #rep,k
        
        sum_pi = torch.sum(-torch.log(torch.clamp(pis,1e-10,1-1e-10)),1, keepdim = True) # bs,1
        loss = sum_pi*(reward_adjust.view(-1,1)) # bs,1
        return loss, reward, st, flu.detach()

    def step(self, s_t_1, emb0, key_pos, pos, length_t,c0,h0):
        # bs,L; bs,l, emb; bs,l; int;
        pos_tensor = torch.zeros(self.batch_size,self.length,1).to(self.device)
        pos_tensor[:,pos,:] = 1
        embt = self.embedding(s_t_1)
        embt[:,pos,:] = embt[:,pos,:]*0
        emb = torch.cat([embt,emb0, pos_tensor],2)
        output, hidden = self.rnn(emb, (c0,h0)) # batch, length,2h
        #pooled = nn.MaxPool1d(self.length)(output.permute(0,2,1)).squeeze(2) #batch,2h
        pooled = output[:,pos,:]
        representation = torch.cat([pooled,key_pos[:,pos:pos+1], pos_tensor.squeeze(2)],1)



        decoded = nn.Softmax(1)(self.decoder(representation)) # bs,V

        if self.training:
            action = decoded.multinomial(1)
            explorate_flag = self.switch_m.multinomial(1)  # as edge_predict
            action_explorate = self.action_m.multinomial(1)
            action1 = action*explorate_flag + action_explorate*(1-explorate_flag)
        else: 
            values,action1 = torch.max(decoded,1, keepdim=True)
        action = action1.detach()
        pi = torch.gather(decoded, 1, action) # (b_s,1), \pi for i,j
        
        replaceflag = torch.lt(action,self.vocab_size).long()
        insertflag = torch.ge(action,self.vocab_size) * torch.lt(action, 2*self.vocab_size)
        insertflag = (insertflag  * torch.le(length_t+insertflag.long(),15)).long()
        deleteflag = torch.eq(action,2*self.vocab_size)
        deleteflag = (deleteflag * torch.gt(length_t-deleteflag.long(),2)).long()

        holdflag = 1- replaceflag -insertflag - deleteflag

        rep = torch.cat([action, s_t_1[:,pos+1:]],1)
        ins = torch.cat([action%self.vocab_size, s_t_1[:,pos:-1]],1)
        padding = torch.ones(self.batch_size,1, device = ins.device, dtype = torch.long)*30001
        dele = torch.cat([s_t_1[:,pos+1:], padding],1)

        hol = s_t_1[:,pos:]
        st = torch.clone(s_t_1)
        st[:,pos:] =  (rep* replaceflag) + (ins * insertflag) +(dele*deleteflag )+ (hol * holdflag)
        length_tt = insertflag+length_t-deleteflag

        re = torch.eq(action,pos+2).long()
        return st, pi, action, length_tt, re
    
    def f(self, st,s0, key_pos, forwardmodel=None, sequence_length =None):
        # bs,l; bs,l; bs, l
        e=1e-50
        M_kw=2
        emb0 = self.embedding(s0)
        embt = self.embedding(st).permute(0,2,1)
        emb_mat = torch.bmm(emb0,embt) # K,l,l
        norm0 = 1/(torch.norm(emb0,p= 2,dim=2)+e) # K,l
        normt = 1/(torch.norm(embt,p= 2,dim=1)+e) # K,l
        norm0 = torch.diag_embed(norm0) # K,15,15
        normt = torch.diag_embed(normt) # k,l,l
        sim_mat = torch.bmm(torch.bmm(norm0, emb_mat), normt) # K,l,l
        sim_vec,_ = torch.max(sim_mat,2)  # K,l
        sim,_ = torch.min(sim_vec*key_pos+ (1-key_pos),1)
        sim = sim**M_kw

        M_repeat = 1
        align_t = st* torch.cat([st[:,1:],st[:,0:1]],1)
        align_0 = s0* torch.cat([s0[:,1:],s0[:,0:1]],1)
        align = torch.eq(align_t,align_0).float()
        expression_diversity = torch.clamp(1-torch.mean(align,1),0,1)

        if forwardmodel is not None:
            prod_prob = forwardmodel.prod_prob(st).squeeze(2) # bs,l, 1
            fluency =\
                    -torch.sum(torch.log(torch.clamp(prod_prob,1e-50)),1)/sequence_length.squeeze(1).float()
            fluency = torch.clamp(fluency, 0,100) 
            fluencyflag = torch.lt(fluency,5).float()
            res = sim * expression_diversity * fluencyflag *((1/fluency)) +\
                     sim * expression_diversity * (1-fluencyflag)*0.01
        else:
            res = sim * expression_diversity
        return res, sim, expression_diversity, fluency


    def init_hidden(self, bsz):
    	weight = next(self.parameters())
    	if self.rnn_type == 'LSTM':
    		return (weight.new_zeros(self.nlayers, bsz, self.nhid),
    				weight.new_zeros(self.nlayers, bsz, self.nhid))
    	else:
    		return weight.new_zeros(self.nlayers, bsz, self.nhid)
   
class UPRL_LM(nn.Module):
    def __init__(self, option):
        super(UPRL_LM, self).__init__()
        rnn_type = 'LSTM'
        self.option = option
        self.length = 15
        self.topk = option.topk
        self.num_edits = option.num_edits
        self.step_reward = option.step_reward
        dropout = option.dropout
        self.prate = 0.1
        self.M_kw = option.M_kw
        self.M_flu = option.M_flu
        self.vocab_size = option.vocab_size
        ninp = option.emb_size
        nhid = option.hidden_size
        self.nlayers = option.num_layers
        self.drop = nn.Dropout(dropout)
        self.embedding= nn.Embedding(self.vocab_size, ninp)

        self.rnn = nn.LSTM(ninp*2+1, nhid, self.nlayers, dropout = dropout ,batch_first=True,\
                bidirectional=True )

        self.del_vec = nn.Parameter(torch.rand(1,1, ninp))
        self.hold_vec = nn.Parameter(torch.rand(1,1, ninp))
        self.repeat_size = option.repeat_size
        self.num_actions = 300 #2*self.vocab_size+2
        self.mlp = nn.Linear(2*nhid+1+self.length, 100)
        self.mlp1 = nn.Linear(ninp+4*self.topk+2,100)
        self.nhid = nhid
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.option.no_cuda else "cpu")

    def forward(self, input, key_pos, sequence_length, forwardmodel, backwardmodel,embmodel, id2sen=None):
        '''
        bs,15; bs,14
        '''

        N_step = self.num_edits
        self.batch_size = input.size(0)
        st = input.clone()
        s0 = input.clone()
        emb0 = self.embedding(s0)
        length_t = sequence_length.clone()
        pis = torch.zeros(self.batch_size,N_step).to(self.device)
        actions = torch.zeros(self.batch_size,N_step).to(self.device)
        c0 = torch.zeros(2*self.nlayers, self.batch_size, self.nhid).to(self.device)
        h0 = torch.zeros(2*self.nlayers, self.batch_size, self.nhid).to(self.device)
        self.switch_m = torch.FloatTensor(self.batch_size, 2).fill_(self.prate).to(self.device)
        self.switch_m[:,1] = 1- self.prate
        self.action_m = torch.FloatTensor(self.batch_size, 4*self.topk+2).fill_(1.0/(4*self.topk+2)).to(self.device)

        for i in range(N_step): 
            pos = i% (self.length-3)+2
            st,pi, action, length_t , re= self.step(st,s0, emb0, key_pos, pos, length_t,c0,h0,\
                    forwardmodel, backwardmodel)
            pis[:,i:i+1] = pi
            actions[:,i:i+1] = action
            #print('-----pos', pos)
            #print(id2sen(st.cpu().numpy()[0]))
            #print(id2sen(re.cpu().numpy()[0]))
        step_reward = torch.sum(torch.eq(actions, 4*self.topk+1).float(),1) * self.step_reward # hold
        fv0, sim, div, flu  = self.f(s0,s0, key_pos, forwardmodel, backwardmodel, embmodel, sequence_length, sequence_length,
                origin=True)
        fvt, sim, div, flu  = self.f(st,s0, key_pos, forwardmodel, backwardmodel, embmodel, length_t, sequence_length)
        improve_flag = fvt/fv0
        reward = (1.0/10)*improve_flag* torch.gt(improve_flag,1).float() + step_reward
        #reward = torch.sum(actions,1)*0.1
        reward = reward.detach() # bs,1
        reward =  reward.view(self.repeat_size, -1) 
        reward_adjust = nn.ReLU()(reward- torch.mean(reward,0, keepdim = True)) #rep,k
        
        sum_pi = torch.sum(-torch.log(torch.clamp(pis,1e-10,1-1e-10)),1, keepdim = True) # bs,1
        loss = sum_pi*(reward_adjust.view(-1,1)) # bs,1
        return loss, reward, st, flu.detach()

    def step(self, s_t_1, s0, emb0, key_pos, pos, length_t,c0,h0, forwardmodel, backwardmodel):
        # bs,L; bs,l, emb; bs,l; int;
        pos_tensor = torch.zeros(self.batch_size,self.length,1).to(self.device)
        pos_tensor[:,pos,:] = 1
        embt = self.embedding(s_t_1)
        embt = embt* (1-pos_tensor)
        emb = torch.cat([embt,emb0, pos_tensor],2)
        output, hidden = self.rnn(emb, (c0,h0)) # batch, length,2h
        #pooled = nn.MaxPool1d(self.length)(output.permute(0,2,1)).squeeze(2) #batch,2h
        pooled = output[:,pos,:]
        # 2*hid+1 +length
        representation = torch.cat([pooled,key_pos[:,pos:pos+1], pos_tensor.squeeze(2)],1)

        rep_f_word, ins_f_word, _ = forwardmodel.predict(s_t_1,s0, pos, length_t, topk=self.topk) # bs,k
        rep_b_word, ins_b_word, valid_id = backwardmodel.predict(s_t_1, s0, pos, length_t, forwardflag =
            False, topk = self.topk) # bs,k
        K = 4*self.topk+2
        tag = torch.arange(K).view(-1,1).long()
        onehot = torch.zeros(K,K).scatter_(1,tag,int(1)).to(self.device)
        # bs,80
        rep_words = torch.cat([rep_f_word, rep_b_word, ins_f_word, ins_b_word], 1)
        word_emb = forwardmodel.encoder(rep_words).detach()

        emb =  torch.cat([word_emb, self.del_vec.repeat(self.batch_size,1,1),\
                self.hold_vec.repeat(self.batch_size,1,1)],1) # bs,82,300
        emb = torch.cat([emb, onehot.unsqueeze(0).repeat(self.batch_size,1,1)],2) #bs,82,382
        representation = self.mlp(representation).unsqueeze(2)
        emb = self.mlp1(emb)
        score = torch.bmm(emb, representation).squeeze(2) # bs,82,382; bs,382,1-> bs,82,1
        policy = nn.Softmax(1)(score) # bs,V

        if self.training:
            action = policy.multinomial(1)
            explorate_flag = self.switch_m.multinomial(1)  # as edge_predict
            action_explorate = self.action_m.multinomial(1)
            action1 = action*explorate_flag + action_explorate*(1-explorate_flag)
        else: 
            values,action1 = torch.max(policy,1, keepdim=True)
        action = action1.detach()
        pi = torch.gather(policy, 1, action) # (b_s,1), \pi for i,j
        
        N_split = 2*self.topk
        replaceflag = (torch.lt(action, N_split)*valid_id).long()
        insertflag = torch.ge(action,N_split) * torch.lt(action, 2*N_split)
        insertflag = (insertflag  * valid_id* torch.le(length_t+insertflag.long(),15)).long()
        deleteflag = torch.eq(action,2*N_split)
        deleteflag = (deleteflag *valid_id* torch.gt(length_t-deleteflag.long(),2)).long()
        holdflag = 1- replaceflag -insertflag - deleteflag
        
        action_words = torch.gather(rep_words, 1, torch.clamp(action, 0,N_split-1).long())
        rep = torch.cat([action_words, s_t_1[:,pos+1:]],1)
        ins = torch.cat([action_words, s_t_1[:,pos:-1]],1)
        padding = torch.ones(self.batch_size,1, device = ins.device, dtype = torch.long)*30003
        dele = torch.cat([s_t_1[:,pos+1:], padding],1)
        #print('rep, ins, del',pos, length_t, replaceflag, insertflag, deleteflag)

        hol = s_t_1[:,pos:]
        st = torch.clone(s_t_1)
        st[:,pos:] =  (rep* replaceflag) + (ins * insertflag) +(dele*deleteflag )+ (hol * holdflag)
        length_tt = insertflag+length_t-deleteflag

        return st, pi, action, length_tt, rep_words
    
    def f(self, st,s0, key_pos, forwardmodel=None, backwardmodel=None,  embmodel=None, sequence_length =None, original_length=None,
            origin = False):
        # bs,l; bs,l; bs, l
        e=1e-50
        emb0 = embmodel.encoder(s0)
        embt = embmodel.encoder(st).permute(0,2,1)
        emb_mat = torch.bmm(emb0,embt) # K,l,l
        norm0 = 1/(torch.norm(emb0,p= 2,dim=2)+e) # K,l
        normt = 1/(torch.norm(embt,p= 2,dim=1)+e) # K,l
        norm0 = torch.diag_embed(norm0) # K,15,15
        normt = torch.diag_embed(normt) # k,l,l
        sim_mat = torch.bmm(torch.bmm(norm0, emb_mat), normt) # K,l,l
        sim_vec,_ = torch.max(sim_mat,2)  # K,l
        sim,_ = torch.min(sim_vec*key_pos+ (1-key_pos),1)
        sim = sim**self.M_kw

        stt = st+1
        s00 = s0+1

        align_t = stt* torch.cat([stt[:,1:],stt[:,0:1]],1) # bs,l
        align_0 = s00* torch.cat([s00[:,1:],s00[:,0:1]],1) #bs,l
        align_t = align_t +  torch.gt(align_t,900000000).long()*1e9
        align_t = align_t.unsqueeze(1).repeat(1,self.length,1)
        align_0 = align_0.unsqueeze(2).repeat(1,1,self.length)
        align = torch.sum(torch.sum(torch.eq(align_t,align_0).float(),1),1)
        expression_diversity = torch.clamp(1-(align/original_length.float().view(-1)),0.05,1)

        if True:
            prod_prob = forwardmodel.prod_prob(st).squeeze(2) # bs,l, 1
            fluency =\
                    -torch.sum(torch.log(torch.clamp(prod_prob,1e-50)),1)/sequence_length.squeeze(1).float()

            st_back = reverse_input(st.cpu(), sequence_length.cpu()).to(self.device)
            prod_prob_back = backwardmodel.prod_prob(st_back).squeeze(2) # bs,l, 1
            fluency_back =\
                    -torch.sum(torch.log(torch.clamp(prod_prob_back,1e-50)),1)/sequence_length.squeeze(1).float()
            fluency = (1/torch.clamp(fluency+fluency_back, 0,100) )**self.M_flu
            if origin:
                fluencyflag = torch.lt(fluency,1e20).float()
            else:
                fluencyflag = torch.lt(fluency,10).float()
            res = sim * expression_diversity * fluencyflag *fluency +\
                     sim * expression_diversity * (1-fluencyflag)*0.01*fluency
        else:
            res = sim * expression_diversity
        return res, sim, expression_diversity, fluency


    def init_hidden(self, bsz):
    	weight = next(self.parameters())
    	if self.rnn_type == 'LSTM':
    		return (weight.new_zeros(self.nlayers, bsz, self.nhid),
    				weight.new_zeros(self.nlayers, bsz, self.nhid))
    	else:
    		return weight.new_zeros(self.nlayers, bsz, self.nhid)
    
