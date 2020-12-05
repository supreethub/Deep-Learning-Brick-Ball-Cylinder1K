
import numpy as np
import time
from fastai.vision import *



def pixel_accuracy(yhat, y):
    y_=y.squeeze(dim=1)
    yhat_=yhat.argmax(dim=1)
    return (y_==yhat_).sum().float()/y.numel()



class Model(object):
    def __init__(self, path='./sample_models', file='export.pkl'):
        
        self.learn=load_learner(path=path, file=file) #Load model
        self.class_names=['brick', 'ball', 'cylinder'] #Be careful here, labeled data uses this order, but fastai will use alphabetical by default!

    def predict(self, x):
        '''
        Input: x = block of input images, stored as Torch.Tensor of dimension (batch_sizex3xHxW), 
                   scaled between 0 and 1. 
        Returns: a tuple containing: 
            1. The final class predictions for each image (brick, ball, or cylinder) as a list of strings.
            2. Upper left and lower right bounding box coordinates (in pixels) for the brick ball 
            or cylinder in each image, as a 2d numpy array of dimension batch_size x 4.
            3. Segmentation mask for the image, as a 3d numpy array of dimension (batch_sizexHxW). Each value 
            in each segmentation mask should be either 0, 1, 2, or 3. Where 0=background, 1=brick, 
            2=ball, 3=cylinder. 
        '''

        #Normalize input data using the same mean and std used in training:
        x_norm=normalize(x, torch.tensor(self.learn.data.stats[0]), 
                            torch.tensor(self.learn.data.stats[1]))

        #Pass data into model:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            yhat = self.learn.model(x_norm.to(device))
            yhat = yhat.detach().cpu()

        class_prediction_indices=yhat.argmax(dim=1)

        class_pred = []
        for arr in class_prediction_indices:
          unique, counts = np.unique(arr, return_counts=True)
          arr_sorted = sorted(list(zip(unique, counts)), key = lambda x: x[1], reverse=True)
          if len(arr_sorted)<2:
            class_pred.append(2)
          else:
            class_pred.append(arr_sorted[1][0])

        class_predictions=[self.learn.data.classes[i] for i in class_pred]


        #Extract bounding box from mask:
        bboxes=np.zeros((x.shape[0], 4))
        for i in range(x.shape[0]):
            rows,cols= np.where(class_prediction_indices[i]!=0)
            if (rows != [] ) or (cols != []):
              bboxes[i, :] = np.array([rows.min(), cols.min(), rows.max(), cols.max()])
            else:
             bboxes[i, :] = np.array([175,80,180,85])

        mask = np.array([np.where(class_prediction_indices[i] > 0, class_pred[i], 0) for i in range(x.shape[0])])
        
        return (class_predictions, bboxes, mask)

