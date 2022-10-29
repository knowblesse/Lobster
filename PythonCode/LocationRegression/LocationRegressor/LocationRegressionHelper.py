import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

class dANN(nn.Module):
    def __init__(self, params):
        # params : device , input_size
        super(dANN, self).__init__()

        self.device = params['device']
        self.fc1 = nn.Linear(
            params['input_size'],
            100)
        self.dp1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(
            100,
            50)
        self.dp2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(
            50,
            25)
        self.fc4 = nn.Linear(
            25,
            params['output_node'])

    def forward(self, x):
        x = x.to(self.device)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.dp1(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.dp2(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)

        return x

    def init_weights(self):
        nn.init.normal_(self.fc1.weight, mean=0, std=0.2)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.2)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.2)
        nn.init.normal_(self.fc4.weight, mean=0, std=0.2)


class EarlyStopping():
    def __init__(self, model, model_control, tolerance=100, save_best=False):
        self.lowest_loss = np.inf
        self.tolerance = tolerance  # how many epochs to stay patient
        self.tolerance_counter = 0
        self.save_best = save_best
        self.model = model
        self.model_control = model_control
        self.model_ever_saved = False


    def __call__(self, loss):
        if loss >= self.lowest_loss:  # worse result
            self.tolerance_counter += 1
            if self.tolerance_counter > self.tolerance:
                return True
        else:  # better result
            self.lowest_loss = loss
            self.tolerance_counter = 0
            if self.save_best:
                torch.save(self.model.state_dict(), Path('./.tempModel'))
                torch.save(self.model_control.state_dict(), Path('./.tempControlModel'))
                self.model_ever_saved = True
        return False


    def loadBest(self):
        if not self.save_best:
            raise(BaseException("Earlystopping : 'save_best' was set as False"))
        if not self.model_ever_saved:
            raise(BaseException("Earlystopping : saved model does not exist"))
        self.model.load_state_dict(torch.load(Path('./.tempModel')))
        self.model_control.load_state_dict(torch.load(Path('./.tempControlModel')))
        self.model.eval()
        self.model_control.eval()


def correctRotationOffset(rotationData):
    # Correct Rotation data for further interpolation.
    # If the degree difference of two consecutive labeled data point is bigger than 180 degree,
    # it is more reasonable to think that the actual rotation is smaller than 180 degree, and
    # crossing the boarder between 0 and 360
    prev_head_direction = rotationData[0]
    degree_offset_value = np.zeros(rotationData.shape[0])
    for i in np.arange(1, rotationData.shape[0]):
        # if the degree change is more than a half rotation, use the smaller rotation value instead.
        if np.abs(rotationData[i] - prev_head_direction) > 180:
            if rotationData[i] > prev_head_direction:
                degree_offset_value[i:] -= 360
            else:
                degree_offset_value[i:] += 360
        prev_head_direction = rotationData[i]
    return rotationData + degree_offset_value

