import time
import copy
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import os
from sklearn import preprocessing
import pandas as pd
import cv2
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import model.vgg_vd_face_fer_dag as vgg_model
import model.alexnet_face_fer_bn_dag as alex_model

###global_variables
batch_size = 16
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
#print(device)


train_data_path = "data/Train/"
train_label_path = "data/Labels/"
#test_data_path = "/content/drive/"'My Drive'"/Dataset/Test/"
val_data_path = "data/Val/"
#test_labels = "/content/drive/"'My Drive'"/Dataset/Train_labels/"

def read_img_from_file(data_path, label_path, data_type):
        img_list= []
        img_file_list =  [f for f in os.listdir(data_path) if not f.startswith('.')]
        for file in img_file_list:
            usr = file
            img_type_list = [f for f in os.listdir(data_path+usr) if not f.startswith('.')]
            if (data_type in img_type_list):
                img_directory_list = [f for f in os.listdir(data_path+usr+"/"+data_type+"/") if not f.startswith('.')]
                #img_directory_list.remove('onset')
                #img_directory_list.remove('apex')
                for categories in img_directory_list:
                    images = [f for f in os.listdir(os.path.join(data_path,usr,data_type,categories,"")) if (f.endswith('.png') and not f.startswith('.'))]
                    for img in images:
                        lbl = read_label_for_img(label_path, categories, usr)
                        if lbl != -1:
                            if os.path.getsize(os.path.join(data_path,usr,data_type,categories,img)) > 0:
                                img_list.append({
                                    'img': os.path.join(data_path,usr,data_type,categories,img),
                                    'label':lbl,
                                    'seq': categories,
                                    'usr': usr
                                })
            else:
                pass
        return img_list
    
def read_label_for_img(path, category, usr):
        label_file_list = [f for f in os.listdir(path) if not f.startswith('.')]
        label_file_list.sort()
        #print (label_file_list)
        label_df = pd.read_csv(path+label_file_list[0], sep = ",")
        try:
            if category[-1] == '1':
                label = label_df.loc[label_df[category[-1]]==int(usr),["happy","disgust","fear","surprise","anger","sadness"]].idxmax(axis=1)
                final_lbl = label.values[0]
            else:
                label = label_df.loc[label_df[category[-1]]==int(usr),[
                    "happy"+"."+str(int(category[-1])-1),"disgust"+"."+str(int(category[-1])-1),"fear"+"."+str(int(category[-1])-1),"surprise"+"."+str(int(category[-1])-1),"anger"+"."+str(int(category[-1])-1),"sadness"+"."+str(int(category[-1])-1)]].idxmax(axis=1)
                final_lbl = label.values[0][:-2]
            return final_lbl
        except:
            return -1

####Loading ImageFile Locations
train_image_list_infrared = read_img_from_file(train_data_path, train_label_path, 'infrared')
val_image_list_infrared = read_img_from_file(val_data_path, train_label_path, 'infrared')
#test_image_list_infrared = read_img_from_file(test_data_path, train_label_path, 'infrared')
train_image_list_visible = read_img_from_file(train_data_path, train_label_path, 'visible')
val_image_list_visible = read_img_from_file(val_data_path, train_label_path, 'visible')
#test_image_list_visible = read_img_from_file(test_data_path, train_label_path, 'visible')
print ("The no. of training infrared images and validation infrared images are: "+str(len(train_image_list_infrared))+" "+str(len(val_image_list_infrared)))
print ("The no. of training visible images and validation visible images are: "+str(len(train_image_list_visible))+" "+str(len(val_image_list_visible)))


#Validations for empty labels
def validations(image_list):
  temp_list = []
  labelset = set()
  for items in image_list:
    if items["label"] == items["label"]:
      temp_list.append(items)
  for items in temp_list:
    labelset.add(items["label"])
  #print (len(temp_list),labelset)
  return temp_list
train_image_list_infrared = validations(train_image_list_infrared)
val_image_list_infrared = validations(val_image_list_infrared)
train_image_list_visible = validations(train_image_list_visible)
val_image_list_visible = validations(val_image_list_visible)


#@title Fusion of Image Types
def fusion(imglist1,imglist2):
  fused = []
  for item1 in imglist1:
    for item2 in imglist2:
      if (item1['usr'] == item2['usr']) and (item1['seq'] == item2['seq']) and (item1['img'].split('/')[-1] == item2['img'].split('/')[-1]):
        #print (item1['img'].split('/')[-1], item2['img'].split('/')[-1])
        fused.append({
            'thermal_img': item1['img'],
            'rgb_img': item2['img'],
            'label': item1['label'],
            'user': item1['usr'],
            'video': item1['seq']
        })
  return fused


train_image_list_fused = fusion(train_image_list_infrared, train_image_list_visible)
val_image_list_fused = fusion(val_image_list_infrared, val_image_list_visible)
print ("The no. of fused training image pairs are: "+str(len(train_image_list_fused)))
print("The no. of fused validation image pairs are: "+str(len(val_image_list_fused)))


#Face_detection
def face_detect(img):
  roi_color = img
  if img is None:
    return None
  face_cascade = cv2.CascadeClassifier("face_detect/haarcascade_frontalface_default.xml")
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
  for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_color = img[y:y+h, x:x+w]
  return roi_color

#dataset_class
class KVIEDataset(data.Dataset):    
    def __init__(self, train=True,fused_image_list=None,transform=None):
        self.transform = transform
        self.train = train
        self.fused_image_info = fused_image_list
        if train:
            print ("No of training/validation visible images: "+str(len(self.fused_image_info)))
        else:
            print ("No of testing images: "+str(len(self.fused_image_info)))
         
    def __len__(self):
        return len(self.fused_image_info)
    
    def __getitem__(self, index):
        fused_info = self.fused_image_info[index]
        #rgb_file_extract_and_face+detect+transform
        visible_img_file = fused_info['rgb_img']
        visible_img = cv2.imread(visible_img_file)
        face_visible_img = face_detect(visible_img)
        v_img = cv2.cvtColor(face_visible_img, cv2.COLOR_BGR2RGB)
        face_visible_img = Image.fromarray(v_img)
        if self.transform is not None:
            face_visible_img = self.transform[0](face_visible_img)
            #face_visible_img = torchv.transforms.normalize(mean=[0.70, 0.44, 0.36], std=[0.34, 0.37, 0.26])
        #face_visible_img = np.array(face_visible_img)
        label = torch.tensor(lb[classes[fused_info['label']]-1])
        
        #thermal_file_extract_and_face+detect+transform
        thermal_img_file = fused_info['thermal_img']
        thermal_img = cv2.imread(thermal_img_file)
        face_thermal_img = face_detect(thermal_img)
        t_img = cv2.cvtColor(face_thermal_img, cv2.COLOR_BGR2RGB)
        thermal_img = Image.fromarray(t_img)
        if self.transform is not None:
          face_thermal_img = self.transform[1](thermal_img)
          #face_thermal_img = .transforms.normalize(mean=[0.34, 0.37, 0.48], std=[0.18, 0.17, 0.24])
        #face_thermal_img = np.array(thermal_img)
        cv2.destroyAllWindows()
        return face_visible_img, face_thermal_img, label
        
def my_collate(batch):
    len_batch = batch_size
    batch = list(filter (lambda x:x is not None, batch)) # filter out all the Nones
    #print (len(batch))
    if len(batch) < batch_size: # if there are samples missing just use existing members, doesn't work if you reject every sample in a batch
      diff = len_batch - len(batch)
      #print (diff)
      batch += batch[:diff]
      #print (len(batch))
    return torch.utils.data.dataloader.default_collate(batch)
    
transform_data = [transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.CenterCrop(224),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.70, 0.44, 0.36], std=[0.34, 0.37, 0.26]),
]),
transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.CenterCrop(224),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.34, 0.37, 0.48], std=[0.18, 0.17, 0.24]),
])
]

##Dataset class initialization
train_dataset = KVIEDataset(train=True, fused_image_list = train_image_list_fused, transform=transform_data)
val_dataset = KVIEDataset(train=True, fused_image_list = val_image_list_fused,transform=transform_data)
#test_dataset = KVIEDataset(train=False, fused_image_list= test_image_list_fused ,transform=transform_data)

##Dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, collate_fn = my_collate, num_workers=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size, collate_fn = my_collate, num_workers=4, shuffle=True)
#test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size, collate_fn = my_collate, num_workers=4, shuffle=True)


###Build the training dataloaders, hyperparameters for training
dataloaders = {}
dataloaders['train'] = train_loader
dataloaders['val'] = val_loader
#dataloaders['test'] = test_loader

dataset_sizes = {}
dataset_sizes['train'] = len(train_loader.dataset)
dataset_sizes['val'] = len(val_loader.dataset)
#dataset_sizes['test'] = len(test_loader.dataset)

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
    

## Fusion model load and hyperparameter values
class desc(nn.Module):
	def __init__(self):
		count = 0
		super(desc, self).__init__()
		self.rgb_model = alexnet_face_fer_bn_dag(weights_path="pretrained_model/alexnet_face_fer_bn_dag.pth")
		self.thermal_model = vgg_vd_face_fer_dag(weights_path="pretrained_model/vgg_vd_face_fer_dag.pth")
		#self.newmodel = torch.nn.Sequential(*(list(self.model.children())[:-2]))
		for params in self.rgb_model.parameters():
			count +=1
			if count <= 26:
				params.requires_grad = False
		'''for params in self.thermal_model.parameters():
			count +=1
			if count <= 30:
				params.requires_grad = False'''
		self.fc = nn.Sequential(nn.Linear(8192, 512, bias=True),
								nn.ReLU(inplace=True),
								nn.Dropout(0.5,inplace=False),
								nn.Linear(512,6, bias=True),
								nn.Sigmoid())
		print (self.rgb_model)
		print (self.thermal_model)

	def forward(self, thermal_img, rgb_img):
		thermal_features = self.thermal_model(thermal_img)
		rgb_features = self.rgb_model(rgb_img)
		combined_features = torch.cat((thermal_features,rgb_features), dim =1)
		#print (combined_features.shape)
		#img = img.view(-1, (512*7*7))
		img = self.fc(combined_features)
		return img
		
torch.cuda.empty_cache()
model_ft = desc()

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.000001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.5)


###training model
def train_model(model, criterion, optimizer, scheduler, writer, num_epochs):
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
            for thermal_inputs,rgb_inputs,labels in dataloaders[phase]:
                thermal_inputs = thermal_inputs.to(device)
                rgb_inputs = rgb_inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(thermal_inputs, rgb_inputs)
                    _, y_pred_tags = torch.max(outputs, dim = 1)
                    loss = criterion(outputs, torch.max(labels, 1)[1])
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                    
            
                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(y_pred_tags == torch.max(labels, 1)[1])
                #break;
            #break;
            if phase == 'train':
              scheduler.step()
                

            epoch_loss = running_loss
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            writer.add_scalar('Loss/'+phase, epoch_loss, epoch)

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

writer = SummaryWriter(log_dir = "Stats/Fusion1")
model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, writer, num_epochs)

###optional
'''torch.save({'epoch': 10,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_ft.state_dict(),
            }, "data/models/fusion_approach1.pth")'''

