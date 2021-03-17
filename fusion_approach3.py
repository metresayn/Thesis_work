import time
import copy
import os
import numpy as np
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from sklearn import preprocessing
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import model.vgg_vd_face_fer_dag as vgg_model
import model.alexnet_face_fer_bn_dag as alex_model


###global_variables
batch_size = 4
num_epochs = 10
classes={
    "happy":1,
    "disgust":2,
    "fear":3,
    "surprise":4,
    "anger":5,
    "sadness":6
}
lb = preprocessing.LabelBinarizer()
lb = lb.fit_transform([1,2,3,4,5,6])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


val_df = os.listdir("Torch_dataset/Val")
train_df = os.listdir("Torch_dataset/Train")
#test_df = os.listdir("orch_dataset/Test")
print ('The no of validation and training videos respectively: ')
print (len(val_df), len(train_df))
#len(test_df))

class KVIEDataset(data.Dataset):    
    def __init__(self,fused_image_list=None,phase=None):
        self.phase = phase
        self.fused_image_info = fused_image_list
        self.rgb_mean = torch.tensor([0.70, 0.44, 0.36])
        self.rgb_std = torch.tensor([0.34, 0.37, 0.26])
        self.thermal_mean = torch.tensor([0.34, 0.37, 0.48])
        self.thermal_std = torch.tensor([0.18, 0.17, 0.24])
        self.rgb_mean = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.rgb_mean,0),2),3)
        self.rgb_std = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.rgb_std,0),2),3)
        self.thermal_mean = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.thermal_mean,0),2),3)
        self.thermal_std = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.thermal_std,0),2),3)
        if phase == 'train':
            print ("No of training videos: "+str(len(self.fused_image_info)))
        elif phase == 'val':
          print ("No of validation videos: "+str(len(self.fused_image_info)))
        else:
            print ("No of testing videos: "+str(len(self.fused_image_info)))
         
    def __len__(self):
        return len(self.fused_image_info)
    
    def __getitem__(self, index):
        if self.phase == 'train':
          fused_info = torch.load("Torch_dataset/Train/"+self.fused_image_info[index])
        elif self.phase == 'val':
          fused_info = torch.load("Torch_dataset/Val/"+self.fused_image_info[index])
        else:
          fused_info = torch.load("Torch_dataset/Test/"+self.fused_image_info[index])
        rgb_images = fused_info['rgb']
        thermal_images = fused_info['thm']
        rgb_images = rgb_images.to(dtype=torch.float32)
        #print (rgb_images.shape)
        thermal_images = thermal_images.to(dtype=torch.float)
        rgb_images = rgb_images/255.0
        thermal_images = thermal_images/255.0
        rgb_images = (rgb_images - self.rgb_mean)/(self.rgb_std)
        thermal_images = (thermal_images - self.thermal_mean)/(self.thermal_std)
        label = fused_info['lbl']
        return rgb_images, thermal_images, label

train_dataset = KVIEDataset(fused_image_list = train_df, phase="train")
val_dataset = KVIEDataset(fused_image_list = val_df, phase="val")

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

dataloaders = {}
dataloaders['train'] = train_loader
dataloaders['val'] = val_loader
dataset_sizes = {}
dataset_sizes['train'] = len(train_loader.dataset)
dataset_sizes['val'] = len(val_loader.dataset)


## Loading thermal and visible models
def vgg_vd_face_fer_dag(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = vgg_model.Vgg_vd_face_fer_dag()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model
    
def alexnet_face_fer_bn_dag(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = alex_model.Alexnet_face_fer_bn_dag()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model

class Conv_LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, bidirectional):
        super(Conv_LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        #self.seq_length = seq_length
        self.dropout = nn.Dropout(p=0.2)
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional = self.bidirectional)
        
        #self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if self.bidirectional == False:
          h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
          c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        else:
          h_0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
          c_0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Propagate input through LSTM
        out, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        #h_out = h_out.view(-1, self.hidden_size)
        
        #out = self.fc(h_out)
        #out = self.dropout(out)
        #print (out.shape)
        if self.hidden_size == 6:
          return self.dropout(h_out)
        else:
          return self.dropout(out)

def lstm(num_classes,input_size,hidden_size,num_layers,bidirectional):
  model = Conv_LSTM(num_classes,input_size,hidden_size,num_layers,bidirectional)
  return model


def init_weights(m):
    if type(m) == nn.LSTM:
        nn.init.orthogonal_(m.weight_ih_l0, gain=10*nn.init.calculate_gain('tanh'))
        nn.init.orthogonal_(m.weight_hh_l0, gain=10*nn.init.calculate_gain('tanh'))
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias, 0.0)
        
class desc(nn.Module):
	def __init__(self):
		counter_rgb,counter_thermal = 0,0
		super(desc, self).__init__()
		self.rgb_model = alexnet_face_fer_bn_dag(weights_path="pretrained_model/alexnet_face_fer_bn_dag.pth")
		self.thermal_model = vgg_vd_face_fer_dag(weights_path="pretrained_model/vgg_vd_face_fer_dag.pth")
		for params in self.rgb_model.parameters():
			counter_rgb +=1
			#print (params.shape)
			if counter_rgb <= 26:
				#print (params)
				params.requires_grad = False
		for params in self.thermal_model.parameters():
			#print (params.shape)
			counter_thermal +=1
			if counter_thermal <= 2:
				#print (params)
				params.requires_grad = False
		print (counter_rgb, counter_thermal)
		self.fc1 = nn.Sequential(nn.Linear(24, 6, bias=True),
														#nn.ReLU(inplace=True),
														#nn.Dropout(0.5,inplace=False),
														#nn.Linear(64,6, bias=True),
														nn.Sigmoid())
		'''self.fc2 = nn.Sequential(nn.Linear(8192, 512, bias=True),
														nn.ReLU(inplace=True),
														nn.Linear(512,6, bias=True),
														nn.Sigmoid())'''
		print (self.rgb_model)
		print (self.thermal_model)
		self.rnn_model_rgb_ini = lstm(num_classes = 6, input_size = 4096, hidden_size = 512, num_layers=1, bidirectional = True)
		self.rnn_model_rgb_final = lstm(num_classes = 6, input_size = 1024, hidden_size = 6, num_layers=1, bidirectional = True)
		self.rnn_model_thermal_ini = lstm(num_classes = 6, input_size = 4096, hidden_size = 512 , num_layers=1, bidirectional = True)
		self.rnn_model_thermal_final = lstm(num_classes = 6, input_size = 1024, hidden_size = 6, num_layers=1, bidirectional = True)
		print (self.rnn_model_rgb_ini)
		#self.rnn_model_rgb.apply(init_weights)
		#self.rnn_model_thermal.apply(init_weights)
		#self.fc2.apply(init_weights)
		self.fc1.apply(init_weights)

	def forward(self, thermal_img, rgb_img):
		thermal_features = self.thermal_model(thermal_img)
		rgb_features = self.rgb_model(rgb_img)
		rgb_lstm_input = rgb_features.view(4,15,4096)
		thermal_lstm_input = thermal_features.view(4,15,4096)
		rnn_out_rgb_1 = self.rnn_model_rgb_ini(rgb_lstm_input)
		rnn_out_rgb_2 = self.rnn_model_rgb_final(rnn_out_rgb_1)
		rnn_out_thm_1 = self.rnn_model_thermal_ini(thermal_lstm_input)
		rnn_out_thm_2 = self.rnn_model_thermal_final(rnn_out_thm_1)
		#print (rnn_out_1.shape, rnn_out_2.shape)
		combined_features = torch.cat((rnn_out_rgb_2,rnn_out_thm_2), dim =2)
		#print (combined_features.shape)
		#print (combined_features[1,0],combined_features[1,1])
		#lstm_input = combined_features.view(4,15,8192)
		#print (lstm_input[0,1,0],lstm_input[0,1,1])
		#rnn_out_1 = self.rnn_model_ini(lstm_input)
		#rnn_out_2 = self.rnn_model_final(rnn_out_1)
		#print (rnn_out_2.shape)
		img = torch.transpose(combined_features, 0, 1)
		img = torch.flatten(img, start_dim=1)
		#print (img.shape)
		img = self.fc1(img)
		return img
		

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

torch.cuda.empty_cache()
model_ft = desc()

model_ft = model_ft.to(device)


criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.000001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.5)


def train_model(model, criterion, optimizer, scheduler, batch_size, num_epochs, writer, sequence_length):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            #since = time.time()
            batch_counter = 0
            for thermal_inputs,rgb_inputs,labels in dataloaders[phase]:
                batch_counter +=1
                thermal_inputs = thermal_inputs.to(device)
                rgb_inputs = rgb_inputs.to(device)
                if (list(thermal_inputs.shape) == [batch_size,sequence_length,3,224,224]) and (list(rgb_inputs.shape) == [batch_size,sequence_length,3,224,224]):
                  thermal_inputs = torch.reshape(thermal_inputs,(batch_size*sequence_length,3,224,224))
                  rgb_inputs = torch.reshape(rgb_inputs,(batch_size*sequence_length,3,224,224))
                  labels = labels.to(device)
                  optimizer.zero_grad()

                  # forward
                  # track history if only in train
                  with torch.set_grad_enabled(phase == 'train'):
                      outputs = model(thermal_inputs, rgb_inputs)
                      #print (outputs)
                      _, y_pred_tags = torch.max(outputs, dim = 1)
                      loss = criterion(outputs, torch.max(labels, 1)[1])
                      # backward + optimize only if in training phase
                      if phase == 'train':
                          loss.backward()
                          optimizer.step()
                          
                      
              
                  # statistics
                  running_loss += loss.item()
                  running_corrects += torch.sum(y_pred_tags == torch.max(labels, 1)[1])
                  #print (running_corrects)
                  #break;
            #break;
            if phase == 'train':
              scheduler.step()
                

            epoch_loss = running_loss
            epoch_acc = running_corrects / dataset_sizes[phase]
            writer.add_scalar('Loss/'+phase, epoch_loss, epoch)
            # print(loss.item())
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts =  copy.deepcopy(model.state_dict())

            print()
            #break;
        #break;

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
    
writer = SummaryWriter(log_dir = "Stats/Fusion3")
model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, batch_size, num_epochs, writer, 15)
###optional
'''torch.save({'epoch': 10,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_ft.state_dict(),
            }, "data/models/fusion_approach2.pth")'''