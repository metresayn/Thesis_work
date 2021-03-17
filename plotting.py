import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

train_loss = []
val_loss = []
step = []
file_location = input('Enter the file location of tensorboard file:')
for e in tf.train.summary_iterator(file_location):
	for v in e.summary.value:
		#print (v.tag, v.simple_value, e.step)
		if v.tag == 'Loss/train':
			train_loss.append(v.simple_value)
		else:
			val_loss.append(v.simple_value)
			step.append(e.step+1)
#print (train_loss, val_loss, step)
fig, axs = plt.subplots(2,sharex=True)
axs[0].plot(step, train_loss, 'tab:orange')
axs[0].set(ylabel='Training_loss')
axs[1].plot(step, val_loss)
axs[1].set(ylabel='Validation_loss', xlabel='Epoch No.')
plt.show()