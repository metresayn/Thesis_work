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
#num_epochs = 10
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

test_data_path = "data/Test/"
test_labels = "data/Labels/"

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
            
test_image_list_infrared = read_img_from_file(test_data_path, test_labels, 'infrared')
test_image_list_visible = read_img_from_file(test_data_path, test_labels, 'visible')

print ("The no. of test infrared images and visible images are: "+str(len(test_image_list_infrared))+" "+str(len(test_image_list_visible)))


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
test_image_list_infrared = validations(test_image_list_infrared)
test_image_list_visible = validations(test_image_list_visible)


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


test_image_list_fused = fusion(test_image_list_infrared, test_image_list_visible)
print ("The no. of fused test image pairs are: "+str(len(test_image_list_fused)))

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

dataset_sizes = {}
test_dataset = KVIEDataset(train=False, fused_image_list= test_image_list_fused ,transform=transform_data)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size, collate_fn = my_collate, num_workers=4, shuffle=True)
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
    

## Fusion model load and hyperparameter values
class desc(nn.Module):
	def __init__(self):
		count = 0
		super(desc, self).__init__()
		self.rgb_model = alexnet_face_fer_bn_dag()
		self.thermal_model = vgg_vd_face_fer_dag()
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

# to load
checkpoint = torch.load('trained_fusion_models/fusion_approach1.pth', map_location=torch.device('cpu'))
model_ft.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer'])


correct = 0
#total = 0
nb_classes = 6
confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for thermal_inputs,rgb_inputs,labels in test_loader:
        thermal_inputs = thermal_inputs.to(device)
        rgb_inputs = rgb_inputs.to(device)
        labels = labels.to(device)
        outputs = model_ft(thermal_inputs,rgb_inputs)
        _, predicted = torch.max(outputs,1)
        #total += labels.size(0)
        correct += (predicted == torch.max(labels, 1)[1]).sum().item()
        for t, p in zip(torch.max(labels, 1)[1], predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
print ("No of corrects: "+str(correct))
print ("Accuracy: "+str(correct/dataset_sizes['test']))
print(confusion_matrix)
print(confusion_matrix.diag()/confusion_matrix.sum(1))



