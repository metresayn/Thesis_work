Steps to run the project:-
-------------------------------------------------------
- Install the required packages using the file -> requirements.txt

Downloading Data and Pre-processing
-----------------------------------
- Run the script preprocess\_rawdata.py.
- This script fetches data from the raw\_data folder and processes it. 
- An example of raw data is provided but in order to download the entire raw data please use the link:
 http://nvie.ustc.edu.cn/download/. The unzip password of database is "lab704". We have used the only the spontaneous part of the database with frontal lighting as part of our experiments.
- Remove the folders onset, apex manually wherever it is necessary.
- In the case of downloading raw data and processing, the partition between Train, Test and Val folders has to be created for subsequent scripts to run.
- The link to already processed Test, Train and Validation data : https://drive.google.com/drive/folders/1SS25V48bNi6RP82gqtx8BU2VvvTdnNVv?usp=sharing. Please
download this data and put them in exactly the same folder structure already created.

Download the pre-trained models
-------------------------------
This is needed as part of feature-extraction purposes. 
The download links for the models : https://drive.google.com/drive/folders/1JLQ5PlmmeUBMU5gyTfQepRrAjlZR-yy6?usp=sharing
Please download the models and put in the folder : pretrained\_model/

Download the trained fusion models:
-----------------------------------
This is needed for running an evaluation of the models.
The download links for the models: https://drive.google.com/drive/folders/1qm8GNi7lx105rlUzf8toR-Q0LWU6ajfT?usp=sharing
Please download these models and put them in the folder : trained\_fusion\_models/


Extra Steps needed to run for Fusion Approach 2 and Fusion Approach 3:
---------------------------------------------------------------------
- Make sure the data folder contains the processed data in the folders Train, Test and Val.
- Run the script torch\_datasets\_prep.py
- This will create a folder called Torch\_dataset with the following folders: Train, Test and Val containing tensor image sequences.


Evaluation Steps:
-----------------
- Run the fusion approaches using the following scripts:
	- fusion\_approach1\_eval.py
	- fusion\_approach2\_eval.py
	- fusion\_approach3\_eval.py


Training Steps:
--------------
- Run the following scripts the train the models:
	- fusion\_approach1.py
	- fusion\_approach2.py
	- fusion\_approach3.py

Visualising Training and Validation Loss Patterns:
--------------------------------------------------
- The training script generates tensorboard writer data in the folder Stats/Fusion'x', where 'x' in [1,2,3]
- Run the script plotting.py and provide the desired file location for plotting. For example : Stats/Fusion3/events.out.tfevents.1608549461.d19903a18cdb.1546.0
	
Assumptions made and changes needed as part of other machines:
--------------------------------------------------------------
- We made our evaluation scripts on a cpu machine, so small changes are needed to make the same script run on a gpu device. For example; torch.load(map)







