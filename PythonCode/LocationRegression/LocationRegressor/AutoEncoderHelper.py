try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    print('pytorch is not installed. Using without it.')
import sys
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.io import loadmat
import platform
import warnings

if 'torch' in sys.modules:
    class dANN_AutoEncoder(nn.Module):
        def __init__(self, params):
            # params : device , input_size
            super(dANN, self).__init__()
            self.device = params['device']

            # Encoder
            self.co_fc1 = nn.Linear(
                params['input_size'],
                100)
            self.co_dp1 = nn.Dropout(0.2)
            self.co_fc2 = nn.Linear(
                100,
                50)
            self.co_dp2 = nn.Dropout(0.2)
            self.co_fc3 = nn.Linear(
                50,
                50)
            self.co_fc4 = nn.Linear(
                50,
                params['encoded_dimension'])

            # Decoder
            self.de_fc1 = nn.Linear(
                params['encoded_dimension'],
                50)
            self.de_fc2 = nn.Linear(
                50,
                50)
            self.de_dp1 = nn.Dropout(0.2)
            self.de_fc3 = nn.Linear(
                50,
                100)
            self.de_dp2 = nn.Dropout(0.2)
            self.de_fc4 = nn.Linear(
                100,
                params['input_size'])
    
        def forward(self, x):
            return self.decode(self.encode(x))

        def encode(self, x):
            x = x.to(self.device)

            # Encoder
            x = self.co_fc1(x)
            x = F.relu(x)

            x = self.co_dp1(x)

            x = self.co_fc2(x)
            x = F.relu(x)

            x = self.co_dp2(x)

            x = self.co_fc3(x)
            x = torch.tanh(x)

            x = self.co_fc4(x)
            x = torch.sigmoid(x)

            return x

        def decode(self,x):
            x = self.de_fc1(x)
            x = torch.tanh(x)

            x = self.de_fc2(x)
            x = F.relu(x)

            x = self.de_dp1(x)

            x = self.de_fc3(x)
            x = F.relu(x)

            x = self.de_dp2(x)

            x = self.de_fc4(x)

            return x

        def init_weights(self):
            nn.init.normal_(self.co_fc1.weight, mean=0, std=0.2)
            nn.init.normal_(self.co_fc2.weight, mean=0, std=0.2)
            nn.init.normal_(self.co_fc3.weight, mean=0, std=0.2)
            nn.init.normal_(self.co_fc4.weight, mean=0, std=0.2)
            nn.init.normal_(self.de_fc1.weight, mean=0, std=0.2)
            nn.init.normal_(self.de_fc2.weight, mean=0, std=0.2)
            nn.init.normal_(self.de_fc3.weight, mean=0, std=0.2)
            nn.init.normal_(self.de_fc4.weight, mean=0, std=0.2)

    class EarlyStopping():
        def __init__(self, model, tolerance=100, save_best=False):
            self.lowest_loss = np.inf
            self.tolerance = tolerance  # how many epochs to stay patient
            self.tolerance_counter = 0
            self.save_best = save_best
            self.model = model
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
                    torch.save(self.model.state_dict(), Path('./tempModel'))
                    self.model_ever_saved = True
            return False

        def loadBest(self):
            if not self.save_best:
                raise (BaseException("Earlystopping : 'save_best' was set as False"))
            if not self.model_ever_saved:
                raise (BaseException("Earlystopping : saved model does not exist"))
            self.model.load_state_dict(torch.load(Path('./tempModel')))
            self.model.eval()
