import os
from PIL import Image

data_path = "data/raw_data/"
save_path = "data/processed_data/"

for dirpath, dirnames, filenames in os.walk(data_path):
    structure = os.path.join(save_path, dirpath[len(data_path):])
    if not os.path.isdir(structure):
        os.mkdir(structure)
    else:
        print("Folder already exists!")

usr_list =  [f for f in os.listdir(data_path) if not f.startswith('.')]
for usrs in usr_list:
	img_type_list = [f for f in os.listdir(os.path.join(data_path,usrs)) if not f.startswith('.')]
	for img_type in img_type_list:
		img_directory_list = [f for f in os.listdir(os.path.join(data_path,usrs,img_type)) if not f.startswith('.')]
		#img_directory_list.remove('onset')
		#img_directory_list.remove('apex')
		for categories in img_directory_list:
			counter = 0
			images = [f for f in os.listdir(os.path.join(data_path,usrs,img_type,categories,"")) if ((f.endswith('.bmp') or f.endswith('.BMP')) and not f.startswith('.'))]
			for img in images:
				if os.path.getsize(os.path.join(data_path,usrs,img_type,categories,img)) > 0:
					raw_img = Image.open(os.path.join(data_path,usrs,img_type,categories,img))
					new_img = raw_img.resize((256, 256))
					new_img.save(os.path.join(save_path,usrs,img_type,categories,str(counter)+'.png'), 'png')
					counter+=1
	print (usrs+" is done")
