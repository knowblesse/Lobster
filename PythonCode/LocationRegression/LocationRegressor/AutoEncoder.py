import re
from scipy.io import savemat
from tqdm import tqdm
from LocationRegressionHelper import *
from AutoEncoderHelper import *
import time
from numpy.random import default_rng
import argparse
import platform
import numpy as np
from pathlib import Path

import requests

print("Code is running on : " + ("cuda" if torch.cuda.is_available else "cpu"))
time.sleep(1)

parser = argparse.ArgumentParser(prog='AutoEncoder')
parser.add_argument('regressor')
parser.add_argument('--removeNestingData', default='False', required=False)
parser.add_argument('--removeEncounterData', default='False', required=False)
parser.add_argument('--removeWanderData', default='False', required=False)
args = parser.parse_args()


def strinput2bool(str_input):
    if str_input in ('True', 'true', 'y', 'yes'):
        return True
    elif str_input in ('False', 'false', 'n', 'no'):
        return False
    else:
        raise BaseException('Wrong argument input')


# Convert to bool
args.removeNestingData = strinput2bool(args.removeNestingData)
args.removeEncounterData = strinput2bool(args.removeEncounterData)
args.removeWanderData = strinput2bool(args.removeWanderData)


def AutoEncoder(tankPath, outputPath, device, numReducedDimension, neural_data_rate, truncatedTime_s, train_epoch, init_lr,
                    numBin, removeNestingData, removeEncounterData, removeWanderData):
    rng = default_rng()
    # Load Tank
    tank_name = re.search('#.*', str(tankPath))[0]
    print(tank_name)

    # Load Data
    neural_data, y_r, y_c, y_deg, midPointTimes, zoneClass = loadData(tankPath, neural_data_rate, truncatedTime_s,
                                                                      removeNestingData, removeEncounterData,
                                                                      removeWanderData, stratifyData=False)

    # Dataset Prepared
    X = np.clip(neural_data, -5, 5)

    # Prepare array to store regression result from the test dataset
    WholeTestResult = np.zeros([X.shape[0], numReducedDimension + 2])  # num data x [row, col, numReducedDimension]
    WholeTestResult[:, :3] = np.hstack((y_r, y_c))

    # Prepare array to store test dataset from the unit shuffled test dataset
    numUnit = int(X.shape[1] / numBin)
    print(f'numUnit : {numUnit}')

    # Setup Testing set
    # The dataset present at timeline. so, extracting every 10th datapoint would
    # generate roughly equal distribution of the data between train/test
    train_index = np.arange(X.shape[0])
    test_index = np.arange(0, X.shape[0], 10)

    train_index = np.delete(train_index, test_index)

    # Log
    train_log = np.zeros((train_epoch, 3))  # lr train test

    ####################
    # Start training
    ####################

    X_train = torch.tensor(X[train_index, :], dtype=torch.float32, device=device, requires_grad=True)
    X_test = torch.tensor(X[test_index, :], dtype=torch.float32, device=device, requires_grad=False)

    params = {'input_size': X_train.shape[1], 'device': device, 'encoded_dimension': numReducedDimension}
    net_real = dANN_AutoEncoder(params).to(device)
    net_real.init_weights()
    optimizer_real = torch.optim.SGD(net_real.parameters(), lr=init_lr, momentum=0.3, weight_decay=0.0)
    scheduler_real = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_real, patience=300, cooldown=100)
    earlyStopping = EarlyStopping(model=net_real, tolerance=1000, save_best=True)

    # Train
    pbar = tqdm(np.arange(train_epoch))

    for e in pbar:
        # Update net_real
        net_real.train()
        loss_real = F.mse_loss(net_real.forward(X_train), y_train)
        optimizer_real.zero_grad()
        loss_real.backward()
        torch.nn.utils.clip_grad_norm_(net_real.parameters(), 5)
        optimizer_real.step()

        # Get learning rate
        lr = [group['lr'] for group in optimizer_real.param_groups]

        # Update tqdm part
        net_real.eval()
        with torch.no_grad():
            loss_train_real = F.mse_loss(net_real.forward(X_train), y_train)
            loss_test_real = F.mse_loss(net_real.forward(X_test), y_test)
            train_log[ e, :] = np.array([
                lr,
                loss_train_real.to('cpu'),
                loss_test_real.to('cpu')])

        pbar.set_postfix_str( \
            f'lr:{lr[0]:.0e} ' +
            f'pr:{torch.mean(loss_real).item():.2f} ' +
            f'pr(Test):{torch.mean(loss_test_real).item():.2f} ')
        scheduler_real.step(loss_real)

        # EarlyStopping
        if (earlyStopping(loss_test_real)):
            break

    earlyStopping.loadBest()

    with torch.no_grad():
        loss_test_real = F.mse_loss(net_real.forward(X_test), y_test)
    print(f'Loss : {torch.mean(loss_test_real).item():.2f}')

    # Generate Regression result for test data
    net_real.eval()
    with torch.no_grad():
        reducedData = net_real.encode(X)

    WholeTestResult[:, 2:numReducedDimension+3] = realFit.to('cpu').numpy()

    savemat(outputPath / f'{tank_name}result_{dataset}.mat', {
        'WholeTestResult': WholeTestResult,
        'midPointTimes': midPointTimes,
        'train_log': train_log})


device = torch.device("cuda" if torch.cuda.is_available else "cpu")

if platform.system() == 'Windows':
    BasePath = Path('D:\Data\Lobster')
else:
    BasePath = Path.home() / 'Data'

InputFolder = BasePath / 'FineDistanceDataset'
OutputFolder = BasePath / 'AutoEncoderResult'

for i, tank in enumerate(sorted([p for p in InputFolder.glob('#*')])):
    print(f'{i:02} {tank}')
    if args.removeWanderData:
        # check if the session has the nnb data
        NNBFolder = BasePath / 'NonNavigationalBehaviorData'
        if not ((NNBFolder / (tank.name + '_nnb.csv')).is_file()):
            continue
    AutoEncoder(
        tankPath=tank,
        outputPath=OutputFolder,
        device=device,
        numReducedDimension=4,
        neural_data_rate=20,
        truncatedTime_s=10,
        train_epoch=20000,
        init_lr=0.005,
        numBin=1,
        removeNestingData=args.removeNestingData,
        removeEncounterData=args.removeEncounterData,
        removeWanderData=args.removeWanderData
    )

