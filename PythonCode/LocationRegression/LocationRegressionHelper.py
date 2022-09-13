import torch
import torch.nn as nn
import torch.nn.functional as F

class dANN(nn.Module):
    def __init__(self, params):
        # params : device , input_size
        super(dANN, self).__init__()

        self.device = params['device']
        self.fc1 = nn.Linear(
            params['input_size'],
            100)
        self.fc2 = nn.Linear(
            100,
            50)
        self.fc3 = nn.Linear(
            50,
            25)
        self.fc4 = nn.Linear(
            25,
            3)

    def forward(self, x):
        x = x.to(self.device)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)

        return x

    def init_weights(self):
        nn.init.normal_(self.fc1.weight, mean=0, std=0.2)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.2)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.2)
        nn.init.normal_(self.fc4.weight, mean=0, std=0.2)

def correctRotationOffset(rotationData):
    # Correct Rotation data for further interpolation.
    # If the degree difference of two consecutive labeld data point is bigger than 180 degree,
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

