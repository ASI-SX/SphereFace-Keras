import keras
from sphereface_20 import modeldef
from keras import backend as K
import tensorflow as tf

#Loss Variables
gamma = 0
iteration = 0
LambdaMin = 5.0
LambdaMax = 1500.0
lamb = 1500.0

def AngleLoss(self, input_format, target):
    
    global iteration
    iteration = iteration + 1
    cos_theta, phi_theta = input_format
    target = K.reshape(target,(-1,1)) #size=(B,1)

    index = K.zeros_like(cos_theta) #size=(B,Classnum)
    #index.scatter_(1,target.data.view(-1,1),1)
    #index = index.byte()
    #index = Variable(index)

    global lamb
    lamb = max(LambdaMin,LambdaMax/(1+0.1*iteration))
    output = cos_theta * 1.0 #size=(B,Classnum)
    output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
    output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

    logpt = tf.nn.log_softmax(output)
    logpt = tf.gather(logpt, target, axis=1)
    #logpt = logpt.view(-1)
    #pt = Variable(logpt.data.exp())

    loss = -1 * (1-pt)**gamma * logpt
    #loss = loss.mean()

    return loss


img_rows, img_cols = 112, 96 # Resolution of inputs
channel = 3
num_classes = 10572
batch_size = 128
nb_epoch = 1
nb_steps = 28000

model = modeldef(img_rows, img_cols, channel, num_classes)

