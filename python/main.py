from LSTM import *
import matplotlib.pyplot as plt

path = '/home/hrituraj/Desktop/InOrProject/AirPassengers.csv'
epochs  = 1
seq_len = 10

print('> Loading data... ')

X_train, y_train, X_test, y_test, nb_modes = load_data(path, seq_len, True)

print('> Data Loaded. Compiling...')

Models = []

print "X_train Shape =%s" % (X_train.shape,)
print "Y_train Shape =%s" % (y_train.shape,)

for mode in range(nb_modes):
    model = build_model([1, 50, 100, 1])

    model.fit(
	       X_train[mode,:,:,:],
	       y_train[mode, :],
	       batch_size=5,
	       nb_epoch=epochs,
	       validation_split=0.10)

    Models.append(model)

predictions = predict_point_by_point(Models, X_train, True)

plt.plot(predictions)
plt.show()

