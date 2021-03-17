import os
import pandas as pd
import math
import torch
from sklearn import preprocessing

train_data_path = "data/Train/"
train_label_path = "data/Labels/"
test_data_path = "data/Test/"
val_data_path = "data/Val/"

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

train_image_list_infrared = read_img_from_file(train_data_path, train_label_path, 'infrared')
val_image_list_infrared = read_img_from_file(val_data_path, train_label_path, 'infrared')
test_image_list_infrared = read_img_from_file(test_data_path, train_label_path, 'infrared')
train_image_list_visible = read_img_from_file(train_data_path, train_label_path, 'visible')
val_image_list_visible = read_img_from_file(val_data_path, train_label_path, 'visible')
test_image_list_visible = read_img_from_file(test_data_path, train_label_path, 'visible')

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
test_image_list_fused = fusion(test_image_list_infrared, test_image_list_visible)
print ('Length of fused images train, val, test respectively:')
print (len(train_image_list_fused), len(val_image_list_fused), len(test_image_list_fused))

train_df = pd.DataFrame.from_dict(train_image_list_fused)
val_df = pd.DataFrame.from_dict(val_image_list_fused)
test_df = pd.DataFrame.from_dict(test_image_list_fused)


#@title .pt converter code
import cv2
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

def read_preprocess_img(img_file, type_of_image):
  img = cv2.imread(img_file)
  face_visible_img = face_detect(img)
  v_img = cv2.cvtColor(face_visible_img, cv2.COLOR_BGR2RGB)
  v_img = torch.tensor(v_img, dtype=torch.uint8).permute(2,0,1)
  v_img = torch.unsqueeze(v_img,0)
  #print (v_img.shape)
  v_img = torch.nn.functional.interpolate(v_img,[224,224])
  #face_img = np.array(face_img)
  cv2.destroyAllWindows()
  return v_img 
    
def read_frame_sequence(fused_image_dict,phase):
  seq_list = []
  rgb_images, thermal_images = [],[]
  users = fused_image_dict['user'].unique()
  for usr in users:
    user_data = fused_image_dict.loc[fused_image_dict['user'] == usr]
    sequences = user_data['video'].unique()
    for seqs in sequences:
      img_data = user_data.loc[user_data['video'] == seqs]
      seq_list.append(img_data)
  
  for seqs in seq_list:
    fused_info = seqs
    #print (fused_info.head(5))
    usr = fused_info['user'].unique()[0]
    sequence = fused_info['video'].unique()[0]
    #print (sequence)
    if len(fused_info.index) >= 60:
      middle_row_no = math.floor(len(fused_info.index)/2)
      label = torch.tensor(lb[classes[fused_info['label'].iloc[middle_row_no]]-1])
      fused_info = fused_info.iloc[middle_row_no-30:middle_row_no+30]
      fused_info = fused_info.reset_index(drop=True)
    else:
      label = torch.tensor(lb[classes[fused_info['label'].iloc[0]]-1])
      diff_in_len = math.ceil((60 - len(fused_info.index))/2)
      df2 = pd.DataFrame(0, index=np.arange(diff_in_len), columns=fused_info.columns)
      new_df = df2.append(fused_info)
      #print (new_df)
      if (len(new_df.index) < 60):
        new_df = new_df.append(pd.DataFrame(0, index=np.arange(60-len(new_df.index)), columns=fused_info.columns))
      fused_info = new_df.reset_index(drop=True)
     #print(fused_info)
    for choice_of_row in range(4):
      #vid_dic = {}
      rgb_images, thermal_images = [],[]
      subset_fused_info = fused_info[(fused_info.index+1) % 4 == choice_of_row]
      for index, row in subset_fused_info.iterrows():
        if (row['rgb_img'] != 0) and (row['thermal_img'] != 0):
          #rgb_file_extract_and_face+detect+transform
          visible_img_file = row['rgb_img']
          face_visible_img = read_preprocess_img(visible_img_file,"RGB")
          face_visible_img = torch.squeeze(face_visible_img)
          rgb_images.append(face_visible_img)
          #thermal_file_extract_and_face+detect+transform
          thermal_img_file = row['thermal_img']
          face_thermal_img = read_preprocess_img(thermal_img_file,"THERMAL")
          face_thermal_img = torch.squeeze(face_thermal_img)
          thermal_images.append(face_thermal_img)
        else:
          rgb_images.append(torch.zeros((3,224,224)))
          thermal_images.append(torch.zeros((3,224,224)))
      rgb = torch.stack(rgb_images)
      thermal = torch.stack(thermal_images)
      vid_dic = {
          'rgb' : rgb,
          'thm' : thermal,
          'lbl' : label
      }
      torch.save(vid_dic,"Torch_dataset/"+phase+"/"+usr+'_'+sequence+'_'+str(choice_of_row)+".pt")
  return 1

read_frame_sequence(train_df, 'Train')
read_frame_sequence(val_df, 'Val')
read_frame_sequence(test_df, 'Test')





