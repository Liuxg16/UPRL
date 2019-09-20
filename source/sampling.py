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

def simulatedAnnealing_std(config):
    option = config
    with tf.name_scope("forward_train"):
        with tf.variable_scope("forward", reuse=None):
            m_forward = PTBModel(is_training=True, config = config)
    with tf.name_scope("forward_test"):
        with tf.variable_scope("forward", reuse=True):
            mtest_forward = PTBModel(is_training=False, config = config)
    var=tf.trainable_variables()
    var_forward=[x for x in var if x.name.startswith('forward')]
    saver_forward=tf.train.Saver(var_forward, max_to_keep=1)
    with tf.name_scope("backward_train"):
        with tf.variable_scope("backward", reuse=None):
            m_backward = PTBModel(is_training=True, config = config)
    with tf.name_scope("backward_test"):
        with tf.variable_scope("backward", reuse=True):
            mtest_backward = PTBModel(is_training=False, config = config)
    var=tf.trainable_variables()
    var_backward=[x for x in var if x.name.startswith('backward')]
    saver_backward=tf.train.Saver(var_backward, max_to_keep=1)
    
    init = tf.global_variables_initializer()
  
    dataclass = data.Data(config)       

    session = tf.Session()
    session.run(init)
    saver_forward.restore(session, option.forward_save_path)
    saver_backward.restore(session, option.backward_save_path)

    if option.mode == 'kw-bleu':
        similarity = similarity_keyword_bleu
    else:
        similarity = similarity_keyword
    similaritymodel = None

    tfflag = True

    fileobj = open(option.emb_path,'r')
    emb_word,emb_id=pkl.load(StrToBytes(fileobj), encoding='latin1')
    fileobj.close()
    sim=option.sim
    sta_vec=list(np.zeros([option.num_steps-1]))

    use_data, sta_vec_list = read_data_use(option, dataclass.sen2id)
    id2sen = dataclass.id2sen
    generateset = []
    C = 0.05
    temperatures =  C*(1.0/100)*np.array(list(range(option.sample_time+1,1,-1)))
    print(temperatures)
    
    for sen_id in range(use_data.length):
        sta_vec=sta_vec_list[sen_id]
        input, sequence_length, _=use_data(1, sen_id)
        input_original=input[0]
        sta_vec_original = [x for x in sta_vec]
        # for i in range(1,option.num_steps):
        #   if input[0][i]>option.rare_since and  input[0][i]<option.dict_size:
        #     sta_vec[i-1]=1
        pos=0
        print('----------------')
        print(' '.join(id2sen(input[0])))
        print(sta_vec)
        calibrated_set = [x for x in input[0]]
        for iter in range(option.sample_time):
            temperature = temperatures[iter]
            ind=pos%(sequence_length[0]-1)
            action=choose_action(option.action_prob)
            calibrated_set = list(set(calibrated_set))
            if action==0: # word replacement (action: 0)
                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input, sequence_length,\
                            mode='use')[0]
                else:
                    prob_old= output_p(input, forwardmodel) #15,K
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word,
                          option, similaritymodel)[0]
                    prob_old_prob*=similarity_old
                else:
                    similarity_old=-1

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)
                if tfflag:
                    prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
                    prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                else:
                    prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                    prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
                         calibrated_set=calibrated_set)

                prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')
                prob_candidate=[]
                for i in range(len(input_candidate)):
                  tem=1
                  for j in range(sequence_length[0]-1):
                    tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                  tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                  prob_candidate.append(tem)
          
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_candidate=prob_candidate*similarity_candidate

                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                
                prob_candidate_prob=prob_candidate[prob_candidate_ind]
                
                V_new = math.log(max(np.power(prob_candidate_prob,1.0/sequence_length),1e-200))
                V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))
                alphat = min(1,math.exp(min((V_new-V_old)/temperature,100)))
                
                if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size:
                    input1=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    if np.sum(input1[0])==np.sum(input[0]):
                        pass
                    else:
                        calibrated_set.append(input[0][ind+1])
                        input= input1
                        print('ind, action,oldprob,vold, vnew, alpha,simold, simnew', ind, action,prob_old_prob,V_old,\
                                        V_new,alphat,similarity_old,similarity_candidate[prob_candidate_ind])
                        print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))

            elif action==1: # word insert
                if sequence_length[0]>=option.num_steps:
                    pos += 1
                    continue
                    # break

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)

                if tfflag:
                    prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
                    prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                else:
                    prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                    prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]

                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
                    calibrated_set=calibrated_set)

                if tfflag:
                    prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')
                else:
                    prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003
                prob_candidate=[]
                #for i in range(option.search_size):
                for i in range(len(input_candidate)):
                    tem=1
                    for j in range(sequence_length_candidate[0]-1):
                        tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                    tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                    prob_candidate.append(tem)
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]
                similarity_new = similarity_candidate[prob_candidate_ind]

                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input,\
                            sequence_length,mode='use')[0]
                else:
                    prob_old = output_p(input, forwardmodel) # 100,15,300003

                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1
                V_new = math.log(max(np.power(prob_candidate_prob,1.0/sequence_length_candidate[0]),1e-200))
                V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))

                alphat = min(1,math.exp(min((V_new-V_old)/temperature,200)))
                if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size:
                    input=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    sequence_length+=1

                    pos+=1
                    # sta_vec.insert(ind, 0.0)
                    # del(sta_vec[-1])
                    print('ind, action,oldprob,vold, vnew, alpha,simold, simnew', ind, action,prob_old_prob,V_old,\
                            V_new,alphat,similarity_old,similarity_new)

                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))


            elif action==2: # word delete
                if sequence_length[0]<=2 or ind==0:
                    pos += 1
                    continue
                if tfflag:
                    prob_old=run_epoch(session, mtest_forward, input, sequence_length,\
                            mode='use')[0]
                else:
                    prob_old= output_p(input, forwardmodel) #15,K
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1

                input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                        sequence_length, ind, None, option.search_size, option,\
                        mode=action,calibrated_set=calibrated_set)

                # delete sentence
                if tfflag:
                    prob_new=run_epoch(session, mtest_forward, input_candidate,\
                            sequence_length_candidate,mode='use')[0]
                else:
                    prob_new = output_p(input_candidate, forwardmodel)


                tem=1
                for j in range(sequence_length_candidate[0]-1):
                    tem*=prob_new[j][input_candidate[0][j+1]]
                tem*=prob_new[j+1][option.dict_size+1]
                prob_new_prob=tem
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)[0]
                    prob_new_prob=prob_new_prob*similarity_candidate
                
                #alpha is acceptance ratio of current proposal
                if input[0] in input_candidate:
                    for candidate_ind in range(len(input_candidate)):
                        if input[0] in input_candidate[candidate_ind: candidate_ind+1]:
                            break
                        pass
                    V_new = math.log(max(np.power(prob_new_prob,1.0/sequence_length_candidate[0]),1e-200))
                    V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))

                    alphat = min(1,math.exp((V_new-V_old)/temperature))
                else:
                    alphat=0
             
                if choose_action([alphat, 1-alphat])==0:
                    calibrated_set.append(input[0][ind])
                    input=np.concatenate([input[:,:ind+1], input[:,ind+2:], input[:,:1]*0+option.dict_size+1], axis=1)
                    sequence_length-=1
                    # del(sta_vec[ind])
                    # sta_vec.append(0)
                    pos -= 1

                    print('ind, action,oldprob,vold, vnew, alpha,simold, simnew',ind, action,prob_old_prob,V_old,\
                                V_new,alphat,similarity_old,similarity_candidate)
                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))


            pos += 1
        generateset.append(id2sen(input[0]))
        appendtext(id2sen(input[0]), option.save_path)

def simulatedAnnealing(config):
    option = config
    with tf.name_scope("forward_train"):
        with tf.variable_scope("forward", reuse=None):
            m_forward = PTBModel(is_training=True, config = config)
    with tf.name_scope("forward_test"):
        with tf.variable_scope("forward", reuse=True):
            mtest_forward = PTBModel(is_training=False, config = config)
    var=tf.trainable_variables()
    var_forward=[x for x in var if x.name.startswith('forward')]
    saver_forward=tf.train.Saver(var_forward, max_to_keep=1)
    with tf.name_scope("backward_train"):
        with tf.variable_scope("backward", reuse=None):
            m_backward = PTBModel(is_training=True, config = config)
    with tf.name_scope("backward_test"):
        with tf.variable_scope("backward", reuse=True):
            mtest_backward = PTBModel(is_training=False, config = config)
    var=tf.trainable_variables()
    var_backward=[x for x in var if x.name.startswith('backward')]
    saver_backward=tf.train.Saver(var_backward, max_to_keep=1)
    
    init = tf.global_variables_initializer()
  
    dataclass = data.Data(config)       

    session = tf.Session()
    session.run(init)
    saver_forward.restore(session, option.forward_save_path)
    saver_backward.restore(session, option.backward_save_path)


    tfflag = True

    fileobj = open(option.emb_path,'r')
    emb_word,emb_id=pkl.load(StrToBytes(fileobj), encoding='latin1')
    fileobj.close()
    sim=option.sim
    sta_vec=list(np.zeros([option.num_steps-1]))

    use_data, sta_vec_list = read_data_use(option, dataclass.sen2id)
    id2sen = dataclass.id2sen
    generateset = []
    
    temperatures =  option.C*(1.0/100)*np.array(list(range(option.sample_time+1,1,-1)))
    print(temperatures)
    option.temperatures = temperatures
    
    for sen_id in range(use_data.length):
        sta_vec=sta_vec_list[sen_id]
        input, sequence_length, _=use_data(1, sen_id)
        print('----------------')
        print(' '.join(id2sen(input[0])))
        print(sta_vec)
        maxV = -30
        for k in range(option.N_repeat):
            sen, V = sa(input, sequence_length, sta_vec, id2sen, emb_word,session, mtest_forward, mtest_backward,option)
            print(sen,V)
            if maxV<V:
                sampledsen = sen
                maxV = V
        print('best',sampledsen, maxV)
        appendtext(sampledsen, option.save_path)

def sa(input, sequence_length, sta_vec, id2sen, emb_word, session, mtest_forward, mtest_backward, option):
    if option.mode == 'kw-bleu':
        similarity = similarity_keyword_bleu
    else:
        similarity = similarity_keyword
    sim = similarity
    similaritymodel = None
    pos=0
    input_original=input[0]
    sta_vec_original = [x for x in sta_vec]
    calibrated_set = [x for x in input[0]]
    for iter in range(option.sample_time):
        temperature = option.temperatures[iter]
        ind=pos%(sequence_length[0]-1)
        action=choose_action(option.action_prob)
        calibrated_set = list(set(calibrated_set))
        if action==0: # word replacement (action: 0)
            prob_old=run_epoch(session, mtest_forward, input, sequence_length,\
                        mode='use')[0]
            tem=1
            for j in range(sequence_length[0]-1):
                tem*=prob_old[j][input[0][j+1]]
            tem*=prob_old[j+1][option.dict_size+1]
            prob_old_prob=tem
            if sim!=None:
                similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word,
                      option, similaritymodel)[0]
                prob_old_prob*=similarity_old
            else:
                similarity_old=-1

            input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                    cut_from_point(input, sequence_length, ind, option, mode=action)
            prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
            prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
            prob_mul=(prob_forward*prob_backward)

            input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                    sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
                     calibrated_set=calibrated_set)

            prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
                        sequence_length_candidate,mode='use')
            prob_candidate=[]
            for i in range(len(input_candidate)):
              tem=1
              for j in range(sequence_length[0]-1):
                tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
              tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
              prob_candidate.append(tem)
      
            prob_candidate=np.array(prob_candidate)
            if sim!=None:
                similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                        id2sen, emb_word, option, similaritymodel)
                prob_candidate=prob_candidate*similarity_candidate

            prob_candidate_norm=normalize(prob_candidate)
            prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
            
            prob_candidate_prob=prob_candidate[prob_candidate_ind]
            
            V_new = math.log(max(np.power(prob_candidate_prob,1.0/sequence_length),1e-200))
            V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))
            alphat = min(1,math.exp(min((V_new-V_old)/temperature,100)))
            
            if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size:
                input1=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                if np.sum(input1[0])==np.sum(input[0]):
                    pass
                else:
                    calibrated_set.append(input[0][ind+1])
                    input= input1
                    print('ind, action,oldprob,vold, vnew, alpha,simold, simnew', ind, action,prob_old_prob,V_old,\
                                    V_new,alphat,similarity_old,similarity_candidate[prob_candidate_ind])
                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))

        elif action==1: # word insert
            if sequence_length[0]>=option.num_steps:
                pos += 1
                continue
                # break

            input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                    cut_from_point(input, sequence_length, ind, option, mode=action)

            prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
            prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]

            prob_mul=(prob_forward*prob_backward)

            input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                    sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
                calibrated_set=calibrated_set)

            prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
                        sequence_length_candidate,mode='use')
            prob_candidate=[]
            for i in range(len(input_candidate)):
                tem=1
                for j in range(sequence_length_candidate[0]-1):
                    tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                prob_candidate.append(tem)
            prob_candidate=np.array(prob_candidate)
            if sim!=None:
                similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                        id2sen, emb_word, option, similaritymodel)
                prob_candidate=prob_candidate*similarity_candidate
            prob_candidate_norm=normalize(prob_candidate)
            prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
            prob_candidate_prob=prob_candidate[prob_candidate_ind]
            similarity_new = similarity_candidate[prob_candidate_ind]

            prob_old=run_epoch(session, mtest_forward, input,\
                        sequence_length,mode='use')[0]

            tem=1
            for j in range(sequence_length[0]-1):
                tem*=prob_old[j][input[0][j+1]]
            tem*=prob_old[j+1][option.dict_size+1]
            prob_old_prob=tem
            if sim!=None:
                similarity_old=similarity(input, input_original,sta_vec,\
                        id2sen, emb_word, option, similaritymodel)[0]
                prob_old_prob=prob_old_prob*similarity_old
            else:
                similarity_old=-1
            V_new = math.log(max(np.power(prob_candidate_prob,1.0/sequence_length_candidate[0]),1e-200))
            V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))

            alphat = min(1,math.exp(min((V_new-V_old)/temperature,200)))
            if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size:
                input=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                sequence_length+=1

                pos+=1
                # sta_vec.insert(ind, 0.0)
                # del(sta_vec[-1])
                print('ind, action,oldprob,vold, vnew, alpha,simold, simnew', ind, action,prob_old_prob,V_old,\
                        V_new,alphat,similarity_old,similarity_new)

                print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))


        elif action==2: # word delete
            if sequence_length[0]<=2 or ind==0:
                pos += 1
                continue
            prob_old=run_epoch(session, mtest_forward, input, sequence_length,\
                        mode='use')[0]
            tem=1
            for j in range(sequence_length[0]-1):
                tem*=prob_old[j][input[0][j+1]]
            tem*=prob_old[j+1][option.dict_size+1]
            prob_old_prob=tem
            if sim!=None:
                similarity_old=similarity(input, input_original,sta_vec,\
                        id2sen, emb_word, option, similaritymodel)[0]
                prob_old_prob=prob_old_prob*similarity_old
            else:
                similarity_old=-1

            input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                    sequence_length, ind, None, option.search_size, option,\
                    mode=action,calibrated_set=calibrated_set)

            # delete sentence
            prob_new=run_epoch(session, mtest_forward, input_candidate,\
                        sequence_length_candidate,mode='use')[0]
            

            tem=1
            for j in range(sequence_length_candidate[0]-1):
                tem*=prob_new[j][input_candidate[0][j+1]]
            tem*=prob_new[j+1][option.dict_size+1]
            prob_new_prob=tem
            if sim!=None:
                similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                        id2sen, emb_word, option, similaritymodel)[0]
                prob_new_prob=prob_new_prob*similarity_candidate
            
            #alpha is acceptance ratio of current proposal
            if input[0] in input_candidate:
                for candidate_ind in range(len(input_candidate)):
                    if input[0] in input_candidate[candidate_ind: candidate_ind+1]:
                        break
                    pass
                V_new = math.log(max(np.power(prob_new_prob,1.0/sequence_length_candidate[0]),1e-200))
                V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))

                alphat = min(1,math.exp((V_new-V_old)/temperature))
            else:
                alphat=0
         
            if choose_action([alphat, 1-alphat])==0:
                calibrated_set.append(input[0][ind])
                input=np.concatenate([input[:,:ind+1], input[:,ind+2:], input[:,:1]*0+option.dict_size+1], axis=1)
                sequence_length-=1
                # del(sta_vec[ind])
                # sta_vec.append(0)
                pos -= 1

                print('ind, action,oldprob,vold, vnew, alpha,simold, simnew',ind, action,prob_old_prob,V_old,\
                            V_new,alphat,similarity_old,similarity_candidate)
                print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))


        pos += 1
    return ' '.join(id2sen(input[0])),V_old

def output_p(sent, model):
    # list
    with torch.no_grad():
        sent = torch.tensor(sent, dtype=torch.long).cuda()
        output = model.predict(sent) # 1,15,300003
        # res = output.cpu().numpy()
        return output
        return output.cpu().numpy()

def outputp(sent, model, length, option, lm=True):
    # list K,l
    # return K,
    N = len(sent)
    with torch.no_grad():
        sent = torch.tensor(sent, dtype=torch.long).cuda()
        output = model.predict(sent) # 1,15,300003
        if lm:
            sent_target = torch.cat([sent[:,1:],sent[:,:1]], 1)
            probs = torch.gather(output.view(output.size(0)*output.size(1),output.size(2)),\
                    1,sent_target.view(-1,1)) # K,l
            probs = torch.prod(probs.view(N,-1)[:,:length[0]],1, dtype=torch.float64)
            return probs.cpu().numpy()
        else:
            return output

def sa_batch(input, sequence_length, sta_vec, id2sen, emb_word, forwardmodel, backwardmodel, option):
    if option.mode == 'kw-bleu':
        similarityfun = similarity_keyword_bleu_tensor
    else:
        similarityfun= similarity_keyword_tensor
    sim = similarityfun
    similaritymodel = None

    generate_candidate = generate_candidate_input_batch
    pos=0
    input_original= copy(input)
    sta_vec_original = [x for x in sta_vec]
    calibrated_set =[ [x for x in inp] for inp in input]
    N_input = len(input)
    for iter in range(option.sample_time):
        temperature = option.temperatures[iter]
        ind=pos%(np.max(sequence_length-1))
        action=choose_action(option.action_prob)
        if action==0: 
            prob_old = output_p(input, forwardmodel).cpu().numpy() #k,l,vocab
            prob_old_prob = getp(prob_old,input, sequence_length, option) # K,
            input_ = [[x] for x in input]
            similarity_old=similarity_batch(input_, input_original, sta_vec, id2sen, emb_word,
                      option, similarityfun, similaritymodel) #K,
            prob_old_prob = prob_old_prob*similarity_old #K,

            input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                    cut_from_point(input, sequence_length, ind, option, mode=action)

            prob_forward = output_p(input_forward, forwardmodel)[:, ind%(sequence_length[0]-1),:]#k,l,vocab
            prob_backward = output_p(input_backward, backwardmodel)[:, sequence_length[0]-1-ind%(sequence_length[0]-1),:]#k,l,vocab
            prob_mul=(prob_forward*prob_backward).cpu().numpy() #K,vocab
            input_candidate, sequence_length_candidate=generate_candidate(input,\
                    sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
                     calibrated_set=calibrated_set) # K,100,15
            input_candidate_flat = input_candidate.reshape(-1,option.num_steps)
            sequence_length_candidate_flat = sequence_length_candidate.reshape(-1)

            prob_candidate_pre = output_p(input_candidate_flat, forwardmodel).cpu().numpy() #k*100,l,vocab
            prob_candidate = getp(prob_candidate_pre,
                    input_candidate_flat,sequence_length_candidate_flat, option) # K*100
            
            prob_candidate = np.array(prob_candidate).reshape(N_input,-1) # K,100
            similarity_candidate=similarity_batch(input_candidate, input_original,sta_vec,\
                        id2sen, emb_word, option ,similarityfun, similaritymodel) #  K,100
            similarity_candidate =similarity_candidate.reshape(N_input,-1)
            prob_candidate=prob_candidate*similarity_candidate # K,100

            # sampling
            prob_candidate_norm=\
                prob_candidate/(np.maximum(prob_candidate.sum(1,keepdims=True),1e-50))
            prob_candidate_ind=samplep(prob_candidate_norm)
            id_sample = torch.tensor(prob_candidate_ind,dtype=torch.long).view(N_input,1)
            prob_candidate_prob= torch.gather(torch.tensor(prob_candidate,dtype=torch.float),1,id_sample) # 5,1
            prob_candidate_prob = prob_candidate_prob.squeeze().numpy() 
            V_new = np.log(np.maximum(np.power(prob_candidate_prob,1.0/sequence_length),1e-200))
            V_old = np.log(np.maximum(np.power(prob_old_prob, 1.0/sequence_length),1e-200))

            alphat = np.minimum(1,np.exp(np.minimum((V_new-V_old)/temperature,100)))
            for i,inp in enumerate(input):                
                if ind>=sequence_length[i]-1:
                    continue
                alpha = alphat[i]
                chooseind = prob_candidate_ind[i]
                if choose_action([alpha, 1-alpha])==0:
                    input1=input_candidate[i][chooseind]
                    if np.sum(input1)==np.sum(inp):
                        pass
                    else:
                        input[i] = input1
                        print('vold, vnew, alpha,simold, simnew', V_old[i],\
                                    V_new[i],alpha,similarity_old[i],similarity_candidate[i][chooseind])
                        print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[i])))

        elif action==1: # word insert
            
            input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                    cut_from_point(input, sequence_length, ind, option, mode=action)

            prob_forward = output_p(input_forward, forwardmodel)[:, ind%(sequence_length[0]-1),:]#k,l,vocab
            prob_backward = output_p(input_backward, backwardmodel)[:, sequence_length[0]-1-ind%(sequence_length[0]-1),:]#k,l,vocab
            prob_mul=(prob_forward*prob_backward).cpu().numpy() #K,vocab

            input_candidate, sequence_length_candidate=generate_candidate(input,\
                    sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
                     calibrated_set=calibrated_set) # K,100,15
            input_candidate_flat = input_candidate.reshape(-1,option.num_steps)
            sequence_length_candidate_flat = sequence_length_candidate.reshape(-1)

            prob_candidate_pre = output_p(input_candidate_flat, forwardmodel).cpu().numpy()#k*100,l,vocab
            prob_candidate = getp(prob_candidate_pre,
                    input_candidate_flat,sequence_length_candidate_flat, option) # K*100
            
            prob_candidate = np.array(prob_candidate).reshape(N_input,-1) # K,100
            similarity_candidate=similarity_batch(input_candidate, input_original,sta_vec,\
                        id2sen, emb_word, option, similarityfun,similaritymodel) #  K,100
            similarity_candidate =similarity_candidate.reshape(N_input,-1)
            prob_candidate=prob_candidate*similarity_candidate # K,100
            prob_candidate_norm=\
                prob_candidate/(np.maximum(prob_candidate.sum(1,keepdims=True),1e-50))
            prob_candidate_ind=samplep(prob_candidate_norm)
            id_sample = torch.tensor(prob_candidate_ind,dtype=torch.long).view(N_input,1)
            prob_candidate_prob= torch.gather(torch.tensor(prob_candidate,dtype=torch.float),1,id_sample) # 5,1
            prob_candidate_prob = prob_candidate_prob.squeeze().numpy() 
            V_new = np.log(np.maximum(np.power(prob_candidate_prob,1.0/(sequence_length+1)),1e-200))

            prob_old = output_p(input, forwardmodel).cpu().numpy() #k,l,vocab
            prob_old_prob = getp(prob_old,input, sequence_length, option) # K,
            input_ = [[x] for x in input]
            similarity_old=similarity_batch(input_, input_original, sta_vec, id2sen, emb_word,
                      option, similarityfun,similaritymodel) #K,
            prob_old_prob = prob_old_prob*similarity_old #K,
            V_old = np.log(np.maximum(np.power(prob_old_prob, 1.0/sequence_length),1e-200))

            alphat = np.minimum(1,np.exp(np.minimum((V_new-V_old)/temperature,100)))
            for i,inp in enumerate(input):                
                if ind>=sequence_length[i]-1 or sequence_length[i]>=option.num_steps:
                    continue
                alpha = alphat[i]
                chooseind = prob_candidate_ind[i]
                if choose_action([alpha, 1-alpha])==0:
                    input1=input_candidate[i][chooseind]
                    input[i] = input1
                    sequence_length[i]  = sequence_length[i]+1
                    print('vold, vnew, alpha,simold, simnew', V_old[i],\
                                    V_new[i],alpha,similarity_old[i],similarity_candidate[i][chooseind])
                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[i])))

        elif action==2: # word delete
            prob_old = output_p(input, forwardmodel).cpu().numpy() #k,l,vocab
            prob_old_prob = getp(prob_old,input, sequence_length, option) # K,
            input_ = [[x] for x in input]
            similarity_old=similarity_batch(input_, input_original, sta_vec, id2sen, emb_word,
                      option, similarityfun,similaritymodel) #K,
            prob_old_prob = prob_old_prob*similarity_old #K,

            input_candidate, sequence_length_candidate=generate_candidate(input,\
                    sequence_length, ind, None, option.search_size, option, mode=action,\
                     calibrated_set=calibrated_set) # K,100,15
            input_candidate = input_candidate[:,0,:]
            sequence_length_candidate = sequence_length_candidate[:,0]
            prob_new = output_p(input_candidate, forwardmodel).cpu().numpy() #k,l,vocab
            prob_new = getp(prob_new, input_candidate,sequence_length_candidate, option) # K

            input_candidate = [[x] for x in input_candidate]
            similarity_new=similarity_batch(input_candidate, input_original,sta_vec,\
                        id2sen, emb_word, option,similarityfun, similaritymodel) #  K,
            prob_new_prob = prob_new* similarity_new #K,

            V_new = np.log(np.maximum(np.power(prob_new_prob,1.0/(sequence_length-1)),1e-200))
            V_old = np.log(np.maximum(np.power(prob_old_prob, 1.0/sequence_length),1e-200))

            alphat = np.minimum(1,np.exp(np.minimum((V_new-V_old)/temperature,100)))
            for i,inp in enumerate(input):                
                if ind>=sequence_length[i]-1 or sequence_length[i]<=3 or ind==0:
                    continue
                alpha = alphat[i]
                if choose_action([alpha, 1-alpha])==0:
                    input1=input_candidate[i][0]
                    input[i] = input1
                        # calibrated_set.append(input[i][ind])
                    sequence_length[i]  = sequence_length[i]-1
                    print('vold, vnew, alpha,simold, simnew', V_old[i],\
                                    V_new[i],alpha,similarity_old[i],similarity_new[i])
                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[i])))

        pos += 1
    return input,V_old

def sa_predicting(input, sequence_length, sta_vec, id2sen, emb_word, forwardmodel, predictingmodel, option):
    if option.mode == 'kw-bleu':
        similarityfun = similarity_keyword_bleu_tensor
    else:
        similarityfun= similarity_keyword_tensor
    sim = similarityfun
    similaritymodel = None

    generate_candidate = generate_candidate_input_update
    pos=0
    input_original= copy(input)
    sta_vec_original = [x for x in sta_vec]
    calibrated_set =[ [x for x in inp] for inp in input]
    N_input = len(input)
    np.random.seed(111)
    for iter in range(option.sample_time):
        temperature = option.temperatures[iter]
        ind=pos%(np.max(sequence_length-1))
        action=choose_action(option.action_prob)
        # print('=================')
        # for inp, lent in zip(input,sequence_length):
        #     print(inp, lent)
        if action==0: 
            prob_old = output_p(input, forwardmodel) #k,l
            prob_old_prob = getp(prob_old,input, sequence_length, option) # K,
            input_ = [[x] for x in input]
            similarity_old=similarity_batch(input_, input_original, sta_vec, id2sen, emb_word,
                      option, similarityfun, similaritymodel) #K,
            prob_old_prob = prob_old_prob*similarity_old #K,
            input_new, input_sequence_length_new =\
                    mask_at_point(input, sequence_length, ind, option, mode=action)
            prob_mul = output_p(input_new, predictingmodel)#k,vocab
            input_candidate, sequence_length_candidate=generate_candidate(input,\
                    sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
                     calibrated_set=calibrated_set) # K,100,15
            # for x in input_candidate[0]:
            #     print(' '.join(id2sen(x)))
            input_candidate_flat = input_candidate.reshape(-1,option.num_steps)
            sequence_length_candidate_flat = sequence_length_candidate.reshape(-1)
             
            prob_candidate_pre = output_p(input_candidate_flat, forwardmodel)#k*100,l,vocab
            prob_candidate = getp(prob_candidate_pre,
                    input_candidate_flat,sequence_length_candidate_flat, option) # K*100
            prob_candidate = np.array(prob_candidate).reshape(N_input,-1) # K,100
            similarity_candidate=similarity_batch(input_candidate, input_original,sta_vec,\
                        id2sen, emb_word, option ,similarityfun, similaritymodel) #  K,100
            similarity_candidate =similarity_candidate.reshape(N_input,-1)
            prob_candidate=prob_candidate*similarity_candidate # K,100
            
            # sampling
            prob_candidate_norm=\
                prob_candidate/(np.maximum(prob_candidate.sum(1,keepdims=True),1e-50))
            prob_candidate_ind=samplep(prob_candidate_norm)
            id_sample = torch.tensor(prob_candidate_ind,dtype=torch.long).view(N_input,1)
            prob_candidate_prob= torch.gather(torch.tensor(prob_candidate,dtype=torch.float),1,id_sample) # 5,1
            prob_candidate_prob = prob_candidate_prob.squeeze().numpy() 
            V_new = np.log(np.maximum(np.power(prob_candidate_prob,1.0/sequence_length),1e-200))
            V_old = np.log(np.maximum(np.power(prob_old_prob, 1.0/sequence_length),1e-200))

            alphat = np.minimum(1,np.exp(np.minimum((V_new-V_old)/temperature,100)))
            for i,inp in enumerate(input):                
                if ind>=sequence_length[i]-1:
                    continue
                alpha = alphat[i]
                chooseind = prob_candidate_ind[i]
                if choose_action([alpha, 1-alpha])==0:
                    input1=input_candidate[i][chooseind]
                    if np.sum(input1)==np.sum(inp):
                        pass
                    else:
                        input[i] = input1
                        print('vold, vnew, alpha,simold, simnew', V_old[i],\
                                    V_new[i],alpha,similarity_old[i],similarity_candidate[i][chooseind])
                        print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[i])))

        elif action==1: # word insert
            input_new, input_sequence_length_new =\
                    mask_at_point(input, sequence_length, ind, option, mode=action)
            prob_mul = output_p(input_new, predictingmodel)#k,vocab
            input_candidate, sequence_length_candidate=generate_candidate(input,\
                    sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
                     calibrated_set=calibrated_set) # K,100,15

            input_candidate_flat = input_candidate.reshape(-1,option.num_steps)
            sequence_length_candidate_flat = sequence_length_candidate.reshape(-1)

            prob_candidate_pre = output_p(input_candidate_flat, forwardmodel)#k*100,l,vocab
            prob_candidate = getp(prob_candidate_pre,
                    input_candidate_flat,sequence_length_candidate_flat, option) # K*100
            
            prob_candidate = np.array(prob_candidate).reshape(N_input,-1) # K,100
            similarity_candidate=similarity_batch(input_candidate, input_original,sta_vec,\
                        id2sen, emb_word, option, similarityfun,similaritymodel) #  K,100
            similarity_candidate =similarity_candidate.reshape(N_input,-1)
            prob_candidate=prob_candidate*similarity_candidate # K,100
            prob_candidate_norm=\
                prob_candidate/(np.maximum(prob_candidate.sum(1,keepdims=True),1e-50))
            prob_candidate_ind=samplep(prob_candidate_norm)
            id_sample = torch.tensor(prob_candidate_ind,dtype=torch.long).view(N_input,1)
            prob_candidate_prob= torch.gather(torch.tensor(prob_candidate,dtype=torch.float),1,id_sample) # 5,1
            prob_candidate_prob = prob_candidate_prob.squeeze().numpy() 
            V_new = np.log(np.maximum(np.power(prob_candidate_prob,1.0/(sequence_length+1)),1e-200))

            prob_old = output_p(input, forwardmodel) #k,l,vocab
            prob_old_prob = getp(prob_old,input, sequence_length, option) # K,
            input_ = [[x] for x in input]
            similarity_old=similarity_batch(input_, input_original, sta_vec, id2sen, emb_word,
                      option, similarityfun,similaritymodel) #K,
            prob_old_prob = prob_old_prob*similarity_old #K,
            V_old = np.log(np.maximum(np.power(prob_old_prob, 1.0/sequence_length),1e-200))

            alphat = np.minimum(1,np.exp(np.minimum((V_new-V_old)/temperature,100)))
            for i,inp in enumerate(input):                
                if ind>=sequence_length[i]-1 or sequence_length[i]>=option.num_steps:
                    continue
                alpha = alphat[i]
                chooseind = prob_candidate_ind[i]
                if choose_action([alpha, 1-alpha])==0:
                    input1=input_candidate[i][chooseind]
                    input[i] = input1
                    sequence_length[i]  = sequence_length[i]+1
                    print('vold, vnew, alpha,simold, simnew', V_old[i],\
                                    V_new[i],alpha,similarity_old[i],similarity_candidate[i][chooseind])
                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[i])))

        elif action==2: # word delete
            prob_old = output_p(input, forwardmodel) #k,l,vocab
            prob_old_prob = getp(prob_old,input, sequence_length, option) # K,
            input_ = [[x] for x in input]
            similarity_old=similarity_batch(input_, input_original, sta_vec, id2sen, emb_word,
                      option, similarityfun,similaritymodel) #K,
            prob_old_prob = prob_old_prob*similarity_old #K,

            input_candidate, sequence_length_candidate=generate_candidate(input,\
                    sequence_length, ind, None, option.search_size, option, mode=action,\
                     calibrated_set=calibrated_set) # K,100,15
            input_candidate = input_candidate[:,0,:]
            sequence_length_candidate = sequence_length_candidate[:,0]
            prob_new = output_p(input_candidate, forwardmodel) #k,l,vocab
            prob_new = getp(prob_new, input_candidate,sequence_length_candidate, option) # K

            input_candidate = [[x] for x in input_candidate]
            similarity_new=similarity_batch(input_candidate, input_original,sta_vec,\
                        id2sen, emb_word, option,similarityfun, similaritymodel) #  K,
            prob_new_prob = prob_new * similarity_new #K,

            V_new = np.log(np.maximum(np.power(prob_new_prob,1.0/(sequence_length-1)),1e-200))
            V_old = np.log(np.maximum(np.power(prob_old_prob, 1.0/sequence_length),1e-200))

            alphat = np.minimum(1,np.exp(np.minimum((V_new-V_old)/temperature,100)))
            for i,inp in enumerate(input):                
                if ind>=sequence_length[i]-1 or sequence_length[i]<=3 or ind==0:
                    continue
                alpha = alphat[i]
                if choose_action([alpha, 1-alpha])==0:
                    input1=input_candidate[i][0]
                    input[i] = input1
                    sequence_length[i]  = sequence_length[i]-1
                    print('vold, vnew, alpha,simold, simnew', V_old[i],\
                                    V_new[i],alpha,similarity_old[i],similarity_new[i])
                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[i])))

        pos += 1
    return input,V_old

def sa_single(input, sequence_length, sta_vec, id2sen, emb_word, forwardmodel, backwardmodel, option):
    if option.mode == 'kw-bleu':
        # similarity = similarity_keyword_bleu
        similarity = similarity_keyword_bleu_tensor
    else:
        similarity = similarity_keyword
    sim = similarity
    similaritymodel = None
    print(' '.join(id2sen(input[0])), sequence_length)
    print(sta_vec)

    pos=0
    input_original=input[0]
    sta_vec_original = [x for x in sta_vec]
    calibrated_set = [x for x in input[0] if x< option.dict_size]
    for iter in range(option.sample_time):
        temperature = option.temperatures[iter]
        ind=pos%(sequence_length[0]-1)
        action=choose_action(option.action_prob)
        calibrated_set = list(set(calibrated_set))
        if action==0: 
            prob_old_prob = outputp(input, forwardmodel, sequence_length, option)[0]
            similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word,
                      option, similaritymodel)[0]
            prob_old_prob*=similarity_old
            input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                    cut_from_point(input, sequence_length, ind, option, mode=action)
            prob_forward =  outputp(input_forward, forwardmodel, sequence_length_forward, \
                    option, lm=False)[0, ind%(sequence_length[0]-1),:]
            
            prob_backward =  outputp(input_backward, backwardmodel, sequence_length_backward,\
                    option, lm=False)[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
            prob_mul=(prob_forward*prob_backward).cpu().numpy()
            input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                    sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
                     calibrated_set=calibrated_set)

            prob_candidate = outputp(input_candidate, forwardmodel, sequence_length_candidate, option) # k,
            similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                    id2sen, emb_word, option, similaritymodel)
            prob_candidate=prob_candidate*similarity_candidate
            prob_candidate_norm=normalize(prob_candidate)
            prob_candidate_ind= choose_an_action(prob_candidate_norm)
            prob_candidate_prob=prob_candidate[prob_candidate_ind]
            V_new = math.log(max(np.power(prob_candidate_prob,1.0/sequence_length_candidate[0]),1e-200))
            V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length[0]),1e-200))
            alphat = min(1,math.exp(min((V_new-V_old)/temperature,100)))
            
            if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size:
                input1=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                if np.sum(input1[0])==np.sum(input[0]):
                    pass
                else:
                    if input[0][ind+1]<option.dict_size:
                        calibrated_set.append(input[0][ind+1])
                    input= input1
                    print('ind, action,oldprob,vold, vnew, alpha,simold, simnew', ind, action,prob_old_prob,V_old,\
                                    V_new,alphat,similarity_old,similarity_candidate[prob_candidate_ind])
                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])),
                            sequence_length)

        elif action==1: # word insert
            if sequence_length[0]>=option.num_steps:
                pos += 1
                continue
                # break

            prob_old_prob = outputp(input, forwardmodel, sequence_length, option)[0]
            similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word,
                      option, similaritymodel)[0]
            prob_old_prob*=similarity_old

            input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                    cut_from_point(input, sequence_length, ind, option, mode=action)
            prob_forward =  outputp(input_forward, forwardmodel, sequence_length_forward, \
                    option, lm=False)[0, ind%(sequence_length[0]-1),:]
            prob_backward =  outputp(input_backward, backwardmodel, sequence_length_backward,\
                    option, lm=False)[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
            prob_mul=(prob_forward*prob_backward).cpu().numpy()
            input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                    sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
                     calibrated_set=calibrated_set)
            prob_candidate = outputp(input_candidate, forwardmodel, sequence_length_candidate, option) # k,
            similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                    id2sen, emb_word, option, similaritymodel)
            prob_candidate=prob_candidate*similarity_candidate
            prob_candidate_norm=normalize(prob_candidate)
            prob_candidate_ind= choose_an_action(prob_candidate_norm)
            prob_candidate_prob=prob_candidate[prob_candidate_ind]
            similarity_new = similarity_candidate[prob_candidate_ind]

            V_new = math.log(max(np.power(prob_candidate_prob,1.0/sequence_length_candidate[0]),1e-200))
            V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length[0]),1e-200))

            alphat = min(1,math.exp(min((V_new-V_old)/temperature,200)))
            if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size:
                input=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                sequence_length+=1
                pos+=1
                print('ind, action,oldprob,vold, vnew, alpha,simold, simnew', ind, action,prob_old_prob,V_old,\
                        V_new,alphat,similarity_old,similarity_new)
                print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])),
                        sequence_length)


        elif action==2: # word delete
            if sequence_length[0]<=2:
                pos += 1
                continue
            prob_old_prob = outputp(input, forwardmodel, sequence_length, option)[0]
            similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word,
                      option, similaritymodel)[0]
            prob_old_prob*=similarity_old

            input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
                    sequence_length, ind, None, option.search_size, option,\
                    mode=action,calibrated_set=calibrated_set)
            prob_candidate = outputp(input_candidate, forwardmodel, sequence_length_candidate,
                    option)[0]
            similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                    id2sen, emb_word, option, similaritymodel)[0]
            prob_new_prob=prob_candidate*similarity_candidate
            
            V_new = math.log(max(np.power(prob_new_prob,1.0/sequence_length_candidate[0]),1e-200))
            V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))
            alphat = min(1,math.exp((V_new-V_old)/temperature))
            if choose_action([alphat, 1-alphat])==0:
                if input[0][ind]<option.dict_size:
                    calibrated_set.append(input[0][ind])
                input=np.concatenate([input[:,:ind+1], input[:,ind+2:], input[:,:1]*0+option.dict_size+1], axis=1)
                sequence_length-=1
                pos -= 1
                print('ind, action,oldprob,vold, vnew, alpha,simold, simnew',ind, action,prob_old_prob,V_old,\
                            V_new,alphat,similarity_old,similarity_candidate)
                print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])),
                        sequence_length[0])


        pos += 1
    return ' '.join(id2sen(input[0])),V_old

def simulatedAnnealing_batch(option, dataclass, forwardmodel = None):
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
    for i in range(0,1000000):
        sen_id = i% (use_data.length//batch_size)
        sta_vec=sta_vec_list[sen_id*batch_size:sen_id*batch_size+batch_size]
        sta_vec = np.array(sta_vec)
        inp, sequence_length, _=use_data(batch_size, sen_id)
        assert len(inp)==len(sequence_length)
        batch_size = len(inp)

        if i % 50 == 0:
            agent.eval()
        else:
            agent.train()
        input = torch.tensor(inp).long().view(1, batch_size, -1).repeat(option.repeat_size,1,1)
        poskeys = torch.tensor(sta_vec).float().view(1, batch_size, -1).repeat(option.repeat_size,1,1)
        sequence_length = torch.tensor(sequence_length).long().view(1, batch_size).repeat(option.repeat_size,1)

        input = input.view(option.repeat_size*batch_size,-1).to(device)
        poskeys = poskeys.view(option.repeat_size*batch_size,-1).to(device)
        sequence_length = sequence_length.view(option.repeat_size*batch_size,-1).to(device)

        loss, rewards, st , temp = agent(input, poskeys, sequence_length, forwardmodel) # bs,15; bs,steps
        if i%50==0 and rewards.mean().item()>maxvalue:
            maxvalue = rewards.mean().item()
            with open('modelbest.pkl', 'wb') as f:
                torch.save(agent.state_dict(), f)

        if i % 100 == 0:
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
            
            
            print(' '.join(id2sen(inp[3])))
            print('key words ', sta_vec[3])
            print('generated:  '+' '.join(id2sen(st.cpu().numpy()[2,3])))
            print('generated:  ', st.cpu().numpy()[2,3])
            print(rewards.cpu().numpy()[2,3])


        #print('generated:  '+' '.join(id2sen(st.cpu().numpy()[3,1])))
        #print(rewards[3,1])
        loss = torch.mean(loss)
        a_rewards =  torch.mean(rewards).item()
        if i % 50 != 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm(agent.parameters(), 5)
            optimizer.step()
            avg_rewards.append(a_rewards)
            print('sentences: {}, avg reward {}, loss {}'.format(i, a_rewards,loss.item()))
        else:
            print('average: {}, avg reward {}, loss {}'.format(i, np.mean(avg_rewards),loss.item()))
            print('Testing: {}, avg reward {}, loss {}'.format(i, a_rewards,loss.item()))
            avg_rewards = []

def testing(option, dataclass, forwardmodel = None):
    option = option
    similarityfun = similarity_keyword_bleu_tensor

    device = torch.device("cuda" if torch.cuda.is_available() and not option.no_cuda else "cpu")
    agent = UPRL(option)
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
    for i in range(0,100):
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

        loss, rewards, st , temp = agent(input, poskeys, sequence_length, forwardmodel) # bs,15; bs,steps

        st = st.view(option.repeat_size,batch_size, -1)
        rewards = rewards.view(option.repeat_size, batch_size)
        temp = temp.view(option.repeat_size, batch_size).detach()
        print(' '.join(id2sen(inp[1])))
        print(inp[1])
        print('length ', sequence_length[1])
        print('key words ', sta_vec[1])
        print('generated:  '+' '.join(id2sen(st.cpu().numpy()[2,1])))
        print('generated:  ', st.cpu().numpy()[2,1])
        print('reward', rewards.cpu().numpy()[2,1])
        print(temp.cpu().numpy()[2,1])
            
            



