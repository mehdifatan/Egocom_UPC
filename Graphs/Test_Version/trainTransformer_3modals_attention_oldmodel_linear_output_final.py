import torch
import pandas as pd
import pickle
import numpy as np
from transformer.batch import subsequent_mask
from Models2 import Transformer
import torch.nn.functional as F
import torch.nn as nn
import datetime
import gc
import matplotlib.pyplot as plt
from Prepare_data import prepare_multiclass_data_from_preprocessed_hdf5
from torch.nn import MultiheadAttention 
import individual_TF

from sklearn.model_selection import train_test_split

from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayers import Norm
import copy


import argparse
parser = argparse.ArgumentParser(description='EgoCom Turn-Taking Prediction')
parser.add_argument('--future-pred', default=3, type=int,
                    help='Specifies on how long in the future we are going'
                         'to predict. Default use of predicting 1 sec in the'
                         'future. Other values to be used are 1,3,5,10')
parser.add_argument('--history-sec', default=5, type=int,
                    help='Specifies on how many seconds of the past we are'
                         'using as a history feature. Default value is 4sec'
                         'but also can be used 4,10,30.')

parser.add_argument('--include-prior', default=None, type=str,
                    help='By default (None) this will run train both a model'
                         'with a prior and a model without a prior.'
                         'Set to "true" to include the label of the current'
                         'speaker when predicting who will be speaking in the'
                         'the future. Set to "false" to not include prior label'
                         'information. You can think of this as a prior on'
                         'the person speaking, since the person who will be'
                         'speaking is highly related to the person who is'
                         'currently speaking.')
parser.add_argument('--sequence-len', default=10, type=int,
                    help='Determine the sequence length of the transformer.'
                         'Could be any integer and has a default vale 10.')






class MyEnsemble(nn.Module):
    def __init__(self, model1, model2,model3,att_layer):
        super(MyEnsemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.att_layer=att_layer

                                            
    def forward(self, src1,src2,src3,src_att1,src_att2,src_att3,target_input,trg_att):

        pred1=self.model1(src1, target_input,src_att1, trg_att)
        pred2=self.model2(src2, target_input,src_att2, trg_att)
        pred3=self.model3(src3, target_input,src_att3, trg_att)

        pred, attn_output_weights = self.att_layer(pred1, pred2, pred3)

        return pred




class NoamOpt:
	"Optim wrapper that implements rate."
	def __init__(self, model_size, factor, warmup, optimizer):
		self.optimizer = optimizer
		self._step = 0
		self.warmup = warmup
		self.factor = factor
		self.model_size = model_size
		self._rate = 0

	def step(self):
		"Update parameters and rate"
		self._step += 1
		rate = self.rate()
		for p in self.optimizer.param_groups:
			p['lr'] = rate
		self._rate = rate
		self.optimizer.step()

	def rate(self, step = None):
		"Implement `lrate` above"
		if step is None:
			step = self._step
		return self.factor * \
			(self.model_size ** (-0.5) *
			min(step ** (-0.5), step * self.warmup ** (-1.5)))





#Make sequences of shifted timestamps (x0,x1,x2),(x1,x2,x3),(x2,x3,x4)
def shift(data, slices ,step):
	grouped_dataset=[]
	for k in data:
		grouped_data=[]
		for i in range(k.shape[0]-slices+1):
			grouped_data.append(k[i*step:i*step+slices])
		grouped_dataset.append(grouped_data)
	return grouped_dataset




#After splitting the data into sequences of the same video, concat them ignoring the video id
def concat_the_sequences(videos):
	seq=[]
	for video in videos:
		for sequence in video:
			seq.append(sequence)
	return seq





def all_idx(idx, axis):
	grid = np.ogrid[tuple(map(slice, idx.shape))]
	grid.insert(axis, idx)
	return tuple(grid)





def onehot_initialization(a):
	ncols = 4
	out = np.zeros(a.shape + (ncols,), dtype=float)
	out[all_idx(a, axis=2)] =  1
	return out





def train_epoch(model, x_train, y_train, optim1, device):
        
        #y_train1=y_train.clone()
	#y_train2=y_train.Clone()
        model.train()
        epoch_loss, n_correct, n_total= 0,0,0
        inp =  iter(x_train)
        
        for id_b,batch in enumerate(y_train): 
                #optim1.zero_grad()  
                
                optim1.optimizer.zero_grad()
		#optim2.optimizer.zero_grad()
		#optim3.optimizer.zero_grad()
                
                #Input features
                src = next(inp).to(device)
                
                src1=src[:,:,:899].clone()
                src2=src[:,:,900:7043].clone()
                src3=src[:,:,7044:].clone()
                
                #Target starting point
                start_of_sequence = torch.tensor(np.zeros((1,4))).repeat(batch.shape[0],1,1)
                trg =  torch.cat((start_of_sequence, batch), 1)
                
                #Shift target by one for next token prediction
                target_input = trg[:,:-1,:].to(device)
                target = trg[:,1:,:].to(device)
                _, target = target.max(2) 
                
                #Make the mask for the next speaker sequences
                src_att1 = torch.ones((src1.shape[0], 1,src1.shape[1])).to(device)
                src_att2 = torch.ones((src2.shape[0], 1,src2.shape[1])).to(device)
                src_att3 = torch.ones((src3.shape[0], 1,src3.shape[1])).to(device)
                
                sequence_length = target_input.size(1)
                
                
                trg_att=subsequent_mask(sequence_length).repeat(target.shape[0],1,1).to(device)
                
                
                pred=model(src1.float(),src2.float(),src3.float(),src_att1,src_att2,src_att3,target_input.float(),trg_att)
                print(pred)
                print(target)
                
                
                
                
                
                pred=torch.transpose(pred,1,2)

                loss1 =criterion1(pred,target)
                loss1.backward()
                optim1.step()
                
                
                epoch_loss += loss1.item()
                
		
                
                _, predicted = pred.max(1)
                
                
		
                n_correct += predicted.eq(target).sum().item()
                n_total += target.size(0)*target.size(1)
                
		
                
                
        time_elapsed = datetime.datetime.now() - start_time     
        print("Time elapsed:", str(time_elapsed).split(".")[0])
        
        
        accuracy=100.*n_correct/n_total
	
        
        
        
        loss_per_epoch = epoch_loss/len(x_train)
        return loss_per_epoch, accuracy










def eval_epoch(model, x_val, y_val, device):

	

	model.eval()



	epoch_loss, n_correct, n_total = 0, 0, 0
	inp = iter(x_val)
	with torch.no_grad():
		for id_b, batch in enumerate(y_val):
			src = next(inp).to(device)

			src1=src[:,:,:899].clone()
			src2=src[:,:,900:7043].clone()
			src3=src[:,:,7044:].clone()



			#Target starting point
			start_of_sequence = torch.tensor(np.zeros((1,4))).repeat(batch.shape[0],1,1)
			trg =  torch.cat((start_of_sequence, batch), 1)

			#Shift target by one for next token prediction
			target_input = trg[:,:-1,:].to(device)
			target = trg[:,1:,:].to(device)
			_, target = target.max(2) 

			#Make the mask for the next speaker sequences
			src_att1 = torch.ones((src1.shape[0], 1,src1.shape[1])).to(device)
			src_att2 = torch.ones((src2.shape[0], 1,src2.shape[1])).to(device)
			src_att3 = torch.ones((src3.shape[0], 1,src3.shape[1])).to(device)



			sequence_length = target_input.size(1)
			trg_att=subsequent_mask(sequence_length).repeat(target.shape[0],1,1).to(device)


			
			pred=model(src1.float(),src2.float(),src3.float(),src_att1,src_att2,src_att3,target_input.float(),trg_att)
			
			
			pred = torch.transpose(pred,1,2)

			_, predicted = pred.max(1)




			loss1 =criterion1(pred,target)
			epoch_loss += loss1.item()



			n_correct += predicted.eq(target).sum().item()
			n_total += target.size(0)*target.size(1)

			


	accuracy=100.*n_correct/n_total 
	loss_per_epoch = epoch_loss/len(x_val)
	return loss_per_epoch, accuracy 











# Extract argument flags
args = parser.parse_args()
future_pred = args.future_pred
history_sec = args.history_sec
sequence_len = args.sequence_len



if args.include_prior is None:
    include_prior_list = [True, False]
elif args.include_prior.lower() == 'true':
    include_prior_list = [True]
elif args.include_prior.lower() == 'false':
    include_prior_list = [False]
else:
    raise ValueError('--include prior should be None, "true", or "false')


val = True



#--------------------------------
modals = ['text_video_voxaudio'] 
include_prior_list = [True, False ]
layers = 6
emb_size=512
heads=8
N=903
warmup = 10
dropout = 0.1
N1=899
N2=6143
N3=1536
heads2=4
max_epoch=20

#device=torch.device("cuda:1")
#cpu='store_true'
#if cpu or not torch.cuda.is_available():
#    device=torch.device("cpu")
verbose=True
val_size=0
#N = x_val.shape[2]
device=torch.device("cuda:2")
#----------------------------------
                                          
#print(moodel)
for modal in modals:
    for include_prior in include_prior_list:
       # del model
        #gc.collect()
        #torch.cuda.empty_cache()
        if not include_prior and modal=='text_video_voxaudio':
            flag_settings = {
                'future_pred': future_pred,
                'history_sec': history_sec,
                'modals': modal, #modals
                'include_prior': include_prior, #list
                'sequence_len' : sequence_len,
            }
            print('Running with settings:', flag_settings)


            x_train,y_train, x_val, y_val, x_test, y_test = prepare_multiclass_data_from_preprocessed_hdf5(
                modal,
                history_sec,
                future_pred,
                include_prior)            
            
			


            (x_train, x_test, y_train, y_test) = train_test_split(x_train, y_train, test_size=0.40, random_state=42)
            (x_test, x_val, y_test, y_val) = train_test_split(x_test, y_test, test_size=0.40, random_state=42)





            x_train = concat_the_sequences(shift(x_train,sequence_len,1))
            y_train = concat_the_sequences(shift(y_train,sequence_len,1))
            x_val = concat_the_sequences(shift(x_val,sequence_len,1))
            y_val = concat_the_sequences(shift(y_val,sequence_len,1))
            x_test = concat_the_sequences(shift(x_test,sequence_len,1))
            y_test = concat_the_sequences(shift(y_test,sequence_len,1))


            x_train = np.stack(x_train)
            
            
            y_train = onehot_initialization(np.stack(y_train))




            print(x_train.shape, y_train.shape)
            x_val = np.stack(x_val)
            y_val = onehot_initialization(np.stack(y_val))
            print(x_val.shape , y_val.shape)
            x_test = np.stack(x_test)
            y_test = onehot_initialization(np.stack(y_test))
            print(x_test.shape, y_test.shape)

            x_tr_dl = torch.utils.data.DataLoader(x_train,  batch_size=25)
            y_tr_dl = torch.utils.data.DataLoader(y_train,  batch_size=25)
            x_val_dl = torch.utils.data.DataLoader(x_val, batch_size=25)
            y_val_dl = torch.utils.data.DataLoader(y_val, batch_size=25)
            x_test_dl = torch.utils.data.DataLoader(x_test, batch_size=25)
            y_test_dl = torch.utils.data.DataLoader(y_test, batch_size=25)


            #optim =  torch.optim.SGD(model.parameters(), lr=0.01)
            criterion1 = nn.CrossEntropyLoss()
            #criterion2 = nn.CrossEntropyLoss()
            #criterion3 = nn.CrossEntropyLoss()

            loss_values, loss_val_values, train_acc_values, val_acc_values, test_acc_values = [] , [] , [] , [], []
            #max_epoch=100
            start_time = datetime.datetime.now()


				

            model1=individual_TF.IndividualTF(N1, 4, 4, N=layers,                                     
                                             d_model=emb_size, d_ff=2048, h=heads, dropout=dropout).float().to(device)
            
            model2=individual_TF.IndividualTF(N2, 4, 4, N=layers,
        
                                             d_model=emb_size, d_ff=2048, h=heads, dropout=dropout).float().to(device)

            model3=individual_TF.IndividualTF(N3, 4, 4, N=layers,                                 
                                             d_model=emb_size, d_ff=2048, h=heads, dropout=dropout).float().to(device)

           


            att_layer=torch.nn.MultiheadAttention(4, heads2, dropout=0,batch_first=True).float().to(device)



            model = MyEnsemble(model1, model2,model3,att_layer)



            optim1 = NoamOpt(512, 1., len(x_tr_dl)*warmup,
                                                    torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9))

           

            #optim1 =torch.optim.Adam(model1.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)


            
            
            #print(max_epoch)
            best_val_acc = 0.0
            best_test_acc= 0.0
            start_time = datetime.datetime.now()
            for epoch in range(max_epoch): 
                train_loss, train_accuracy = train_epoch(model, x_tr_dl, y_tr_dl,optim1,device)
                loss_values.append(train_loss)
                train_acc_values.append(train_accuracy)
                print('Epoch: %03i/%03i | Train Loss: %.3f | Accuracy: %.3f'%(epoch+1, max_epoch, train_loss ,train_accuracy))
                val_loss, val_accuracy = eval_epoch(model, x_val_dl, y_val_dl, device)
                loss_val_values.append(val_loss)
                val_acc_values.append(val_accuracy)
                print('Epoch: %03i/%03i | Val Loss : %.3f | Val Accuracy %.3f'%(epoch+1, max_epoch, val_loss , val_accuracy))
                test_loss,test_accuracy = eval_epoch(model, x_test_dl, y_test_dl, device)
                test_acc_values.append(test_accuracy)
                print('Epoch: %03i/%03i | Test Accuracy %.3f'%(epoch+1, max_epoch, test_accuracy))
                
                #print("The best test accuracy so far is: ",test_accuracy)

                if val_accuracy>best_val_acc:
                    best_val_acc = val_accuracy
                    best_test_acc = test_accuracy
                    print("The best test accuracy so far is: ",best_test_acc)
                else:
                    print("The best test accuracy so far is: ",best_test_acc)
                    




            plt.figure()
            plt.plot(loss_values, label = "train loss")
            plt.plot(loss_val_values , label ="val loss")
            leg = plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.savefig('Graphs/Test_Version/Loss_'+ 'Linear' +'.png')
            plt.show()
            plt.figure().clear()
            plt.close()
            plt.cla()
            plt.clf()

            plt.figure()
            plt.plot(train_acc_values, label = "train acc")
            plt.plot(val_acc_values, label = "val acc")
            plt.plot(test_acc_values, label = "test acc")
            leg = plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy %")
            plt.savefig('Graphs/Test_Version/Accuracy_'+ 'Linear' +'.png')
            plt.show()
            plt.figure().clear()
            plt.close()
            plt.cla()
            plt.clf()

