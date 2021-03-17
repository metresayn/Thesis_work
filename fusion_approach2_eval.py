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


test_df = os.listdir("Torch_dataset/Test")

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

test_dataset = KVIEDataset(fused_image_list = test_df, phase="test")
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

dataloaders = {}
dataloaders['test'] = test_loader
dataset_sizes = {}
dataset_sizes['test'] = len(test_loader.dataset)

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
        #self.dropout = nn.Dropout(p=0.2)
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional = self.bidirectional)
        
        #self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if self.hidden_size == 6:
          h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
          c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        else:
          h_0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
          c_0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Propagate input through LSTM
        out, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        if self.hidden_size == 128:
          return h_out
        else:
          return out

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
		self.rgb_model = alexnet_face_fer_bn_dag()
		self.thermal_model = vgg_vd_face_fer_dag()
		#self.newmodel = torch.nn.Sequential(*(list(self.model.children())[:-2]))
		for params in self.rgb_model.parameters():
			counter_rgb +=1
			#print (params.shape)
			if counter_rgb <= 32:
				#print (params)
				params.requires_grad = False
		for params in self.thermal_model.parameters():
			#print (params.shape)
			counter_thermal +=1
			if counter_thermal <= 4:
				#print (params)
				params.requires_grad = False
		self.fc2 = nn.Sequential(nn.Linear(8192, 512, bias=True),
								nn.ReLU(inplace=True),
                            	nn.Dropout(p=0.5,inplace=False),
								nn.Linear(512,6, bias=True),
								nn.Sigmoid())
		print (self.rgb_model)
		print (self.thermal_model)
		self.fc2.apply(init_weights)

	def forward(self, thermal_img, rgb_img):
		thermal_features = self.thermal_model(thermal_img)
		rgb_features = self.rgb_model(rgb_img)
		combined_features = torch.cat((thermal_features,rgb_features), dim =1)
		lstm_input = combined_features.view(4,15,8192)

		img = self.fc2(lstm_input)
		img = torch.mean(img,1,True)
		img = torch.flatten(img, start_dim=1)
		return img


model_ft = desc()
model_ft = model_ft.to(device)

# to load
checkpoint = torch.load('trained_fusion_models/fusion_approach2.pth', map_location=torch.device('cpu'))
model_ft.load_state_dict(checkpoint['model_state_dict'])


correct = 0
#total = 0
nb_classes = 6
confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for thermal_inputs,rgb_inputs,labels in test_loader:
        thermal_inputs = thermal_inputs.to(device)
        rgb_inputs = rgb_inputs.to(device)
        #labels = labels.to(device)

        if (list(thermal_inputs.shape) == [batch_size,15,3,224,224]) and (list(rgb_inputs.shape) == [batch_size,15,3,224,224]):
                  thermal_inputs = torch.reshape(thermal_inputs,(batch_size*15,3,224,224))
                  rgb_inputs = torch.reshape(rgb_inputs,(batch_size*15,3,224,224))
                  labels = labels.to(device)
                  outputs = model_ft(thermal_inputs, rgb_inputs)
                  _, predicted = torch.max(outputs,1)
                  #total += labels.size(0)
                  correct += (predicted == torch.max(labels, 1)[1]).sum().item()
                  for t, p in zip(torch.max(labels, 1)[1], predicted.view(-1)):
                          confusion_matrix[t.long(), p.long()] += 1
print ("No of corrects: "+str(correct))
print ("Accuracy: "+str(correct/dataset_sizes['test']))
print(confusion_matrix)
print(confusion_matrix.diag()/confusion_matrix.sum(1))