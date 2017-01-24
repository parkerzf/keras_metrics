import numpy as np
from keras.models import Model
from keras.layers import Input

preds = np.load(open('preds.npy'))
y_regression = np.load(open('y_regression.npy'))

print 'manual result: mse=%f' % np.mean(np.square(y_regression - preds))
x = Input(preds.shape[1:])
m = Model(x, x)
m.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mape'])
scores = m.evaluate(preds, y_regression, batch_size=32)

print '\nevaluate result: mse=%f' % scores[0]
