import numpy as np
import re
import sys
from Switching.SwitchingHelper import parseAllData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pickle
import platform
from pathlib import Path
from tqdm import tqdm

#############################################
# Setup Cuda                                #
#############################################
if 'torch' in sys.modules:
    import torch
    from torch import nn
    from torch.nn import functional as F
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
                params['output_node'])


        def forward(self, x):
            x = x.to(self.device)

            x = self.fc1(x)
            x = F.relu(x)

            x = self.fc2(x)
            x = F.relu(x)

            x = self.fc3(x)

            return x


        def init_weights(self):
            nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
            nn.init.normal_(self.fc2.weight, mean=0, std=0.1)
            nn.init.normal_(self.fc3.weight, mean=0, std=0.1)


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
                raise(BaseException("Earlystopping : 'save_best' was set as False"))
            if not self.model_ever_saved:
                raise(BaseException("Earlystopping : saved model does not exist"))
            self.model.load_state_dict(torch.load(Path('./tempModel')))
            self.model.eval()


class TorchRegressor():
    def __init__(self, init_lr=1e-3, train_epoch=5000):
        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        self.init_lr = init_lr
        self.train_epoch = train_epoch

    def train(self, X_train, Y_train):
        """
        Train network
        :param X_train:
        :param Y_train:
        :return:
        """
        X_train = torch.tensor(X, dtype=torch.float32, device=self.device, requires_grad=True)
        Y_train = torch.tensor(Y, dtype=torch.float32, device=self.device, requires_grad=False)

        params = {
            'input_size': X_train.shape[1],
            'device': self.device,
            'output_node': Y_train.shape[1]  # number of neurons
        }

        self.net = dANN(params).to(self.device)
        self.net.init_weights()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.init_lr, momentum=0.5, weight_decay=0.0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=100, cooldown=100)

        #earlyStopping = EarlyStopping(model=net_real, model_control=net_fake, tolerance=1000, save_best=True)
        earlyStopping = EarlyStopping(model=self.net, tolerance=100, save_best=True)

        # Train
        pbar = tqdm(np.arange(self.train_epoch))

        for e in pbar:
            # Update net_real
            self.net.train()
            loss = F.l1_loss(self.net.forward(X_train), Y_train)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 2)
            self.optimizer.step()

            # Get learning rate
            lr = [group['lr'] for group in self.optimizer.param_groups]

            # Update tqdm part
            self.net.eval()

            with torch.no_grad():
                loss = F.l1_loss(self.net.forward(X_train), Y_train)

            pbar.set_postfix_str(
                f'lr:{lr[0]:.0e} ' +
                f'loss:{torch.mean(loss).item():.4f}')
            self.scheduler.step(loss)

            # EarlyStopping
            if (earlyStopping(loss)):
                break

        earlyStopping.loadBest()

    def predict(self, X_test):
        """
        Predict
        :param X_test:
        :return:
        """
        self.net.eval()
        with torch.no_grad():
            return self.net.forward(X_test)

    def score(self, X_test, Y_test):
        """
        Score
        :param X_test:
        :param Y_test:
        :return:
        """
        self.net.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32, device=self.device, requires_grad=True)
        with torch.no_grad():
            Y_pred = self.net.forward(X_test).to('cpu').numpy()
            print(r2_score(Y_test, Y_pred))
            return r2_score(Y_test, Y_pred)



if platform.system() == 'Windows':
    BasePath = Path('D:\Data\Lobster')
else:
    BasePath = Path.home() / 'Data'


InputFolder = BasePath / 'FineDistanceDataset'
OutputFolder = BasePath / 'FineDistanceResult_syncFixed_June'

data_out = []
regressor_out = []

for i, tank in enumerate(sorted([p for p in InputFolder.glob('#*')])):
    print(f'{i:02} {tank}')
    tankPath =  tank

    # Load Tank
    tank_name = re.search('#.*', str(tankPath))[0]
    print(tank_name)

    # Load Data
    data = parseAllData(tank_name)
    Y =  data['neural_data']

    location_data = data['location_data'] # row col real-distance fake_result original_result

    distance = location_data[:, 2] # TODO: Normalize?
    in_AE = data['inAE'] # one or zero. whether in avoid or escape trial
    current_zone = data['zoneClass'] # in which zone
    mid_point_times = data['midPointTimes']

    X = np.vstack((distance, in_AE, current_zone, mid_point_times)).T
    normalizer = StandardScaler()
    X = normalizer.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=622)

    full_model = TorchRegressor()
    full_model.train(X_train, Y_train)
    full_model_score = full_model.score(X_test, Y_test) # R^2 for the full model

    #no_distance_model = MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(200,50), max_iter=1000, solver='sgd', alpha=1e-3, learning_rate='invscaling', learning_rate_init=1e-4, power_t=0.2)).fit(np.delete(X_train, 0, axis=1), Y_train)
    #no_distance_score = no_distance_model.score(np.delete(X_test, 0, axis=1), Y_test) # R^2 for the full model

    #no_AE_model = MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(200,50), max_iter=1000, solver='sgd', alpha=1e-3, learning_rate='invscaling', learning_rate_init=1e-4, power_t=0.2)).fit(np.delete(X_train, 1, axis=1), Y_train)
    #no_AE_score = no_AE_model.score(np.delete(X_test, 1, axis=1), Y_test) # R^2 for the full model

    #no_zone_model = MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(200,50), max_iter=1000, solver='sgd', alpha=1e-3, learning_rate='invscaling', learning_rate_init=1e-4, power_t=0.2)).fit(np.delete(X_train, 2, axis=1), Y_train)
    #no_zone_score = no_zone_model.score(np.delete(X_test, 2, axis=1), Y_test) # R^2 for the full model

    #no_time_model = MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(200,50), max_iter=4000, solver='sgd', alpha=1e-3, learning_rate='invscaling', learning_rate_init=1e-4, power_t=0.2)).fit(np.delete(X_train, 3, axis=1), Y_train)
    #no_time_score = no_time_model.score(np.delete(X_test, 3, axis=1), Y_test) # R^2 for the full model

    #distance_explained = full_model_score - no_distance_score
    #AE_explained = full_model_score - no_AE_score
    #zone_explained = full_model_score - no_zone_score
    #time_explained = full_model_score - no_time_score

    #print(f"[{i}] full {full_model_score:.2f} distance {distance_explained:.2f} AE {AE_explained:.2f} zone {zone_explained:.2f} time {time_explained:.2f}")

    #result = {
    #    'tank_name': tank_name,
    #    'full': full_model_score,
    #    'distance': distance_explained,
    #    'AE': AE_explained,
    #    'zone': zone_explained,
    #    'time': time_explained
    #}

    #regressor = {
    #    'X_train': X_train,
    #    'Y_train': Y_train,
    #    'X_test': X_test,
    #    'Y_test': Y_test,
    #    'full': full_model,
    #    'distance': no_distance_model,
    #    'AE': no_AE_model,
    #    'zone': no_zone_model,
    #    'time': no_time_model
    #}

    #data_out.append(result)
    #regressor_out.append(regressor)


out = {
    'regression_result': data_out,
    'regressor': regressor_out
}

with open(Path('C:/Users/knowb/Desktop/RegressionData.pkl'), 'wb') as f:
    pickle.dump(out, f)
