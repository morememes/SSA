import numpy as np
import torch
from torch import nn, optim
from sklearn.preprocessing import MinMaxScaler
from easydict import EasyDict as edict
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import copy

def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

def createValXSSA(ts, val_size, seq_length, scalerssa, ssa):
  xs = []
  ys = []

  
  for i in tqdm(range(val_size-seq_length-1)):
    res = applySSA(ts[:-val_size + (i+seq_length)+1], ssa)
    res = scalerssa.transform(np.expand_dims(res, axis=1)).squeeze()
    xs.append(res[(-seq_length-1):-1])
    ys.append(res[-1])
  return np.array(xs), np.array(ys)

def applySSA(ts, ssa):
  s = ssa.rssa.ssa(ts, kind='1d-ssa', L=ssa.L);
  res = ssa.rssa.reconstruct(s, ssa.ssa_groups)

  resD = dict(zip(res.names, list(res)))
  res = resD['F1']
  for key in list(resD.keys())[1:]:
    res += resD[key]
  return res

def createTestTsSSA(ts, size, ssa):
  res = []
  for i in tqdm(range(size)):
    sl = (-size + 1 + i)
    if sl == 0:
      t = ts
    else:
      t = ts[: sl ]
    r = applySSA(t, ssa)
    res.append(r[-1])
  return np.array(res)

def splitData(ts, seq_len, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, ssa, val_ssa = False):
  tsOrigin = ts

  scaler = MinMaxScaler()
  scaler = scaler.fit(np.expand_dims(ts, axis=1))
  ts = scaler.transform(np.expand_dims(ts, axis=1)).squeeze()

  dTrain = ts[:TRAIN_SIZE]
  dVal = ts[TRAIN_SIZE:(TRAIN_SIZE+VAL_SIZE)]
  dTest = ts[(TRAIN_SIZE+VAL_SIZE):]

  X_train, y_train = create_sequences(dTrain, seq_len)
  X_val, y_val = create_sequences(dVal, seq_len)
  X_test, y_test = create_sequences(dTest, seq_len)

  X_val_ssa = None
  y_val_ssa = None
  

  res = applySSA(tsOrigin[:TRAIN_SIZE], ssa)

  scalerssa = MinMaxScaler()
  scalerssa = scalerssa.fit(np.expand_dims(res, axis=1))
  res = scalerssa.transform(np.expand_dims(res, axis=1)).squeeze()
  
  if (val_ssa):
    X_val_ssa, y_val_ssa = createValXSSA(tsOrigin[:(TRAIN_SIZE + VAL_SIZE)], VAL_SIZE, seq_len, 
                                         scalerssa, ssa)
    X_val_ssa = torch.from_numpy(X_val_ssa).float()
    y_val_ssa = torch.from_numpy(y_val_ssa).float()

  ts_test_ssa = None
  if (val_ssa):
    ts_test_ssa = createTestTsSSA(tsOrigin, TEST_SIZE, ssa)

  X_train_ssa, y_train_ssa = create_sequences(res, seq_len)

  data = {
      'train' : {
        'ts' : dTrain,
        'X' : torch.from_numpy(X_train).float(),
        'X_ssa' : torch.from_numpy(X_train_ssa).float(),
        'y' : torch.from_numpy(y_train).float(),
        'y_ssa' : torch.from_numpy(y_train_ssa).float(),
        'size' : TRAIN_SIZE
      },
      'val' : {
        'ts' : dVal,
        'X' : torch.from_numpy(X_val).float(),
        'X_ssa' : X_val_ssa,
        'y' : torch.from_numpy(y_val).float(),
        'y_ssa' : y_val_ssa,
        'size' : VAL_SIZE
      },
      'test' : {
        'ts' : dTest,
        'X' : torch.from_numpy(X_test).float(),
        'y' : torch.from_numpy(y_test).float(),
        'size' : TEST_SIZE,
        'ts_ssa' : ts_test_ssa
      },
      'ts_scaled' : ts,
      'ts' : tsOrigin,
      'seq_len' : seq_len,
      'scaler' : scaler,
      'scaler_ssa' : scalerssa
  }
  data = edict(data)

  return data

def train_model(epochs, model, train_data, train_labels, test_data=None, test_labels=None, checkpoint = 10):
  loss_fn = torch.nn.MSELoss(reduction='mean')

  optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
  num_epochs = epochs + 1

  train_hist = np.zeros(num_epochs)
  test_hist = np.zeros(num_epochs)

  for t in range(num_epochs):

    y_pred = model(train_data).squeeze()

    loss = loss_fn(y_pred.float(), train_labels)

# evaluate model:
    if test_data is not None:
      model.eval()
      with torch.no_grad():
        y_test_pred = model(test_data).squeeze()
        test_loss = loss_fn(y_test_pred.float(), test_labels)
      test_hist[t] = test_loss.item()
      model.train()

      if (t) % checkpoint == 0:  
        print(f'Epoch {t} train loss: {loss.item()} val loss: {test_loss.item()}')
    elif (t) % checkpoint == 0:
      print(f'Epoch {t} train loss: {loss.item()}')

    train_hist[t] = loss.item()
    
    optimiser.zero_grad()

    loss.backward()

    optimiser.step()
  
  return model.eval(), train_hist, test_hist
def train(model, N_epochs, X_train, y_train, X_val, y_val, checkpoint):
    return train_model(N_epochs, model, X_train, y_train, X_val, y_val, checkpoint)


def plot3graphs(true_cases, predicted_cases_raw, predicted_cases_ssa):

  plt.plot(
    np.arange(len(true_cases)),
    true_cases,
    label='Real Data'
  )

  # raw
  plt.plot(
    np.arange(len(predicted_cases_raw)), 
    predicted_cases_raw, 
    label='ANN'
  )

  # ssa
  plt.plot(
    np.arange(len(predicted_cases_ssa)), 
    predicted_cases_ssa, 
    label='SSA-ANN'
  )

  plt.legend();

def start_train(iargs, mArgs, name):
  args = copy(iargs)
  args.model = None
  N_epochs = args.N_epochs
  # load model
  model = mArgs.model_fun(**mArgs.model_inits)
  args.update({'model' : model})
  print("\nStart to train {} model \n".format(name))
  # first train
  model, train_hist, val_hist = train(**args)
  # detect low loss
  mi = np.where(val_hist == val_hist.min())[0][0]
  del model
  # load model
  args.update({'model' : None})
  model = mArgs.model_fun(**mArgs.model_inits)
  args.update({'model' : model, 'N_epochs' : mi})
  print("\n Start to train {} model \n".format(name))
  # second train
  model, train_hist, val_hist = train(**args)
  return model, train_hist, val_hist


def plotNgraphs(cases, labels):
  for case, label in zip(cases, labels):
    plt.plot(list(range(len(case))), case, label=label)
  plt.legend();


def getPreds(model, seq, scaler, scaler_ssa, testSize, seq_length, ssa_ts, ssa):
  with torch.no_grad():
    if ssa_ts is None:
      test_sq = scaler.transform(np.expand_dims(seq, axis=1)).squeeze()
    else:
      test_sq = scaler_ssa.transform(np.expand_dims(seq, axis=1)).squeeze()

    test_seq = torch.from_numpy(test_sq).float()
    preds = []

    for _ in tqdm(range(testSize)):
      y_test_pred = model(test_seq)
      pred = torch.flatten(y_test_pred).item()
      preds.append(pred)
      if ssa_ts is None:
        new_seq = test_seq.numpy().flatten()
        new_seq = np.append(new_seq, [pred])
        new_seq = new_seq[1:]
      else:
        pred2Numpy = scaler.inverse_transform(np.expand_dims(np.array([pred]), axis=0)).flatten()
        ssa_ts = np.concatenate([ssa_ts, pred2Numpy], axis = 0)[1:]
        new_seq = applySSA(ssa_ts, ssa)
        new_seq = np.array(new_seq[-seq_length:])
        new_seq = scaler_ssa.transform(np.expand_dims(new_seq, axis=1) ).squeeze()
      test_seq = torch.as_tensor(new_seq).view(seq_length).float()
    return np.array(preds)

def getPredsTomorrow(model, seq, groundTruth, scaler, testSize, seq_length):
    assert len(groundTruth) == testSize, "wrong size of groundTruth"
    
    groundTruth = scaler.transform(np.expand_dims(groundTruth, axis=1)).squeeze()
    test_sq = scaler.transform(np.expand_dims(seq, axis=1)).squeeze()
    test_seq = torch.from_numpy(test_sq).float()
    preds = []

    for i in tqdm(range(testSize)):
      y_test_pred = model(test_seq)
      pred = torch.flatten(y_test_pred).item()
      preds.append(pred)
      new_seq = test_seq.numpy().flatten()
      new_seq = np.append(new_seq, groundTruth[i])
      new_seq = new_seq[1:]
      test_seq = torch.as_tensor(new_seq).view(seq_length).float()
    return np.array(preds)

def getPredCases(model, data, ssa_ts = None, ssa = None, tomorrow = None, y_ssa = True):

  if ssa is None:
    seq = data.ts[:(data.train.size + data.val.size)][-data.seq_len:]
  else:
    seq = applySSA(data.ts[:(data.train.size + data.val.size)], ssa)
    seq = seq[-data.seq_len:]

  if tomorrow is None:
    preds = getPreds(model, 
                   seq,
                   data.scaler,
                   data.scaler_ssa, 
                   testSize = data.test.size, 
                   seq_length = data.seq_len, 
                   ssa_ts = ssa_ts, 
                   ssa = ssa)
  
    predicted_cases = data.scaler.inverse_transform(
    np.expand_dims(preds, axis=0)
    ).flatten()
  elif tomorrow is True:
    if ssa is None:
      ts = data.ts[-data.test.size:]
      sc = data.scaler
    else:
      ts = data.test.ts_ssa
      sc = data.scaler_ssa
    preds = getPredsTomorrow(model, 
                             seq, 
                             ts,
                             sc,
                             testSize = data.test.size,
                             seq_length = data.seq_len
                             )
    if y_ssa is True:
      predicted_cases = data.scaler_ssa.inverse_transform(
        np.expand_dims(preds, axis=0)
        ).flatten()
    else:
      predicted_cases = data.scaler.inverse_transform(
        np.expand_dims(preds, axis=0)
        ).flatten()
  else:
    seq = tomorrow.ts[:(data.train.size + data.val.size)][-data.seq_len:]
    preds = getPredsTomorrow(model, 
                             seq, 
                             tomorrow.ts[-data.test.size:],
                             tomorrow.scaler,
                             testSize = data.test.size,
                             seq_length = data.seq_len
                             )
    if y_ssa is True:
      predicted_cases = data.scaler_ssa.inverse_transform(
        np.expand_dims(preds, axis=0)
        ).flatten()
    else:
      predicted_cases = data.scaler.inverse_transform(
        np.expand_dims(preds, axis=0)
        ).flatten()

  return predicted_cases

#################################################################
##########################    MODEL    ##########################
#################################################################


class RainModel(nn.Module):

  def __init__(self, inputSize, hiddenLayer, outputSize):
    super(RainModel, self).__init__()
    self.linear1 = torch.nn.Linear(inputSize, hiddenLayer)
    self.linear2 = torch.nn.Linear(hiddenLayer, outputSize)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.sigmoid1 = nn.Sigmoid()


  def forward(self, x):
    x = self.linear1(x)
    x = self.sigmoid(x)
    x = self.linear2(x)    
    out = x

    return out

#################################################################
####################    ONE STEP TRAINING    ####################
#################################################################

def createOneStepData(data):
  d = edict({
    'train' : {
        'ts' : data.train.ts,
        'X' : data.train.X,
        'X_ssa' : data.train.X_ssa,
        'y' : data.train.y,
        'size' : data.train.size
    },
    'val' : {
        'ts' : data.val.ts,
        'X' : data.val.X,
        'y' : data.val.y,
        'size' : data.val.size
    },
    'seq_len' : data.seq_len,
    'scaler' : data.scaler,
    'ts' : data.ts[:-data.test.size],
    'ts_scaled' : data.ts_scaled[:-data.test.size],
    'ts_ssa_scaled' : None
  })
  return d

def shiftTrain(data, ssa = None):
  if ssa is not None:
    res = applySSA(data.ts, ssa)
    res = data.scaler.transform(np.expand_dims(res, axis=1)).squeeze()
    X_train, _ = create_sequences(res[-data.train.size:], data.seq_len)
    data.ts_ssa_scaled = res
  else:
    X_train, _ = create_sequences(data.ts_scaled[-data.train.size:], data.seq_len)
  _, y_train = create_sequences(data.ts_scaled[-data.train.size:], data.seq_len)
  train_data = {
    'X_train' : torch.from_numpy(X_train).float(),
    'y_train' : torch.from_numpy(y_train).float(),
    'X_val' : None,
    'y_val' : None
    }
  return edict(train_data)

def shiftData(data, pred, ssa = None):
  data.ts = np.concatenate([data.ts[1:], pred], axis = 0)
  ts = data.scaler.transform(np.expand_dims(data.ts, axis=1)).squeeze()
  data.ts_scaled = ts

  dTrain = ts[:data.train.size]
  dVal = ts[data.train.size:(data.train.size+data.val.size)]
  X_train, y_train = create_sequences(dTrain, data.seq_len)
  X_val, y_val = create_sequences(dVal, data.seq_len)

  data.train.X_ssa = None
  if ssa is not None:
    res = applySSA(data.ts[:data.train.size], ssa)
    res = data.scaler.transform(np.expand_dims(res, axis=1)).squeeze()
    X_train_ssa, _ = create_sequences(res, data.seq_len)
    data.train.X_ssa = torch.from_numpy(X_train_ssa).float()

  data.train.X = torch.from_numpy(X_train).float()
  data.train.y = torch.from_numpy(y_train).float()
  data.val.X = torch.from_numpy(X_val).float()
  data.val.y = torch.from_numpy(y_val).float()
  return data

def updateFitParams(fitParams, data, ssa = None):
    if ssa is None:
      X_train = data.train.X
    else:
      X_train = data.train.X_ssa

    d = edict({
      'X_train' : X_train,
      'y_train' : data.train.y,
      'X_val' : data.val.X,
      'y_val' : data.val.y
    })

    fitParams.update(d)
    return fitParams

def getPrediction(model, seq, scaler):
  seq = torch.from_numpy(seq).float()
  y_pred = model(seq)
  pred = torch.flatten(y_pred).item()
  pred = scaler.inverse_transform(np.expand_dims(np.array([pred]), axis=0)).flatten()
  return pred

def trainAndPredictOneStep(N_epochs, model, data, testSize, ssa = None):
  predictions = []
  testSize

  fitParams = edict({
      'model' : model,
      'N_epochs' : N_epochs,
      'checkpoint' : N_epochs +2,
  })
  fitParams = updateFitParams(fitParams, data, ssa)

  for i in tqdm(range(testSize)):
      model, train_hist, val_hist = train(**fitParams)
      minEphs = np.where(val_hist == val_hist.min())[0][0]
      # переделать выборку: X_train_new, y_train_new = funcition(...)
      train_data = shiftTrain(data, ssa)
      # kwargs update
      fitParams.update(train_data)
      model, _, _ = train(**fitParams)
      # pred = model prediction
      if ssa is None:
        seq = data.ts_scaled[-data.seq_len:]
      else:
        seq = data.ts_ssa_scaled[-data.seq_len:]
      pred = getPrediction(model, seq, data.scaler)
      predictions.append(pred[0])
      data = shiftData(data, pred, ssa)
      fitParams = updateFitParams(fitParams, data, ssa)
  return predictions