import os
import numpy as np

uex = lambda x: np.sin(x[:,0])*np.sin(x[:,1])

train_size = 2**20
test_size = 2**10

X_train = np.random.rand(train_size,2)
u_train = uex(X_train)

X_test = np.random.rand(test_size,2)
u_test = uex(X_test)

data_dir = './pinndata'
if not os.path.exists(data_dir):
	os.makedirs(data_dir)
np.savez(os.path.join(data_dir, 'data.npz'),X_train=X_train,u_train=u_train,X_test=X_test,u_test=u_test)
