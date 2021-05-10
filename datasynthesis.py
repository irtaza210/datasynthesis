#irtaza's code starts from line 40 - 49 and from line 283 - 376
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fmnist_train = datasets.FashionMNIST("../data", train=True, download=True, transform=transforms.ToTensor())
fmnist_test = datasets.FashionMNIST("../data", train=False, download=True, transform=transforms.ToTensor())
binarization_train_batch = len(fmnist_train)
binarization_test_batch = len(fmnist_test)
binarization_train_loader = DataLoader(fmnist_train, batch_size=binarization_train_batch, shuffle=True)
binarization_test_loader = DataLoader(fmnist_test, batch_size=binarization_test_batch, shuffle=False)
temp_train_batch = len(fmnist_train)
temp_test_batch = len(fmnist_test)
temp_train_loader = DataLoader(fmnist_train, batch_size=temp_train_batch, shuffle=True)
temp_test_loader = DataLoader(fmnist_test, batch_size=temp_test_batch, shuffle=False)
train_batch_size = 30

import random
#lucy's code
def distribution_matrix(X, num_thresholds):
    # given p is an integer of power 2, and num_thresholds = p-1,
    # find p-1 thresholds value for each feature so that the probability that a feature of an image falls between two consecutive thresholds values is 1/p
    threshold_value = np.ndarray(shape=(num_thresholds, X.shape[1], X.shape[2], X.shape[3]))
    for color in range(X.shape[1]):
        for i in range(X.shape[2]):
            for j in range(X.shape[3]):
                dist = []
                for image in X:
                    dist.append(image[color][i][j])
                for threshold_idx in range(num_thresholds):
                    threshold_value[threshold_idx][color][i][j] = np.quantile(dist, (threshold_idx + 1) / (
                            num_thresholds + 1))
    return threshold_value
#irtaza's code
def generateKey(n):
    key = np.zeros(28*28*n)
    for i in range(28*28*n):
        randomnum= random.random()
        if (randomnum>0.5):
            key[i]= -1
        else:
            key[i] = 1
    return key.reshape((1, -1))
#lucy's code
def generateHmatrix(length):
    # recursively construct the hadamard matrix of size 2 to the power length
    H = np.ones(shape=(np.power(2, length), np.power(2, length)))
    i1 = 1
    while i1 < np.power(2, length):
        for i2 in range(i1):
            for i3 in range(i1):
                H[i2 + i1][i3] = H[i2][i3]
                H[i2][i3 + i1] = H[i2][i3]
                H[i2 + i1][i3 + i1] = -H[i2][i3]
        i1 += i1
    return H[:, 1:]
#lucy's code
def mapping(row):
    # convert a bits string into an array
    out = []
    for num in row:
        out.append(num)
    return out
#lucy's code
def generateMappingorder(X, length):
    # for every pixel in the image, randomly assign an order from 0 to 2 to the power length-1,
    # which represents which row of hadamard matrix a feature would be mapped into given which two thresholds values it lies between

    mapping = np.ndarray(shape=(X.shape[1], X.shape[2], X.shape[3]), dtype=object)
    for k in range(X.shape[1]):
        for i in range(X.shape[2]):
            for j in range(X.shape[3]):
                mapping_order = np.arange(np.power(2, length - 1))
                np.random.shuffle(mapping_order)
                mapping[k][i][j] = mapping_order
    return mapping
#lucy's code
def whitening(X):
    # decorrelate the features in the image
    images = X
    mean = images.mean(axis=0)
    images = images - mean
    images = images.reshape(images.shape[0], images.shape[1] * images.shape[2] * images.shape[3])
    cov = np.cov(images, rowvar=False)
    U, S, V = np.linalg.svd(cov)
    images = np.dot(images, U)
    images = images.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
    return images, U, mean

#lucy's code
def encoding(X, distribution, Z, length, H, mapping_order, num_inputs):
    # for each feature of every image, determine which two thresholds values it lies between,
    # map it to the corresponding row in the hadamard matrix,
    # perform bitwise xor with the key value assigned to that specific feature,
    # output the encoded result
    input_dataset = torch.empty(size=(X.shape[0], num_inputs))
    print("input_dataset ", input_dataset)
    for k in range(X.shape[0]):
        image = X[k]
        pixel_idx = 0
        for c in range(X.shape[1]):
            for i in range(X.shape[2]):
                for j in range(X.shape[3]):
                    dist_range = np.power(2, length - 1) - 2
                    while image[c][i][j] < distribution[dist_range][c][i][j] and dist_range > -1:
                        dist_range -= 1
                    coded = H[length - 1][mapping_order[c][i][j][dist_range + 1]]
                    coding = xor(mapping(coded), Z[0])

                    for bit_idx in range(len(coding)):
                        input_dataset[k][pixel_idx] = torch.tensor([coding[bit_idx]])
                        pixel_idx += 1
        del image
    print(input_dataset)
    return input_dataset
#lucy's code
def xor(s1, s2):
    ans = []
    for i in range(len(s1)):
        if s1[i] == s2[i]:
            ans.append(1)
        else:
            ans.append(-1)
    return ans
#lucy's code
def quantized_matrix(X, num_thresholds, distribution):
    quantized_value = np.ndarray(shape=(num_thresholds + 1, X.shape[1], X.shape[2], X.shape[3]))
    for c in range(X.shape[1]):
        for i in range(X.shape[2]):
            for j in range(X.shape[3]):
                for quantized_idx in range(num_thresholds + 1):
                    if quantized_idx == num_thresholds:
                        quantized_value[quantized_idx][c][i][j] = np.random.uniform(
                            distribution[quantized_idx - 1][c][i][j], 1)
                    elif quantized_idx == 0:
                        quantized_value[quantized_idx][c][i][j] = np.random.uniform(0,
                                                                                    distribution[quantized_idx][c][i][
                                                                                        j])
                    else:
                        quantized_value[quantized_idx][c][i][j] = np.random.uniform(
                            distribution[quantized_idx - 1][c][i][j], distribution[quantized_idx][c][i][j])
    return quantized_value
#lucy's code
def quantization(X, distribution, quant):
    input_dataset = torch.empty(size=(X.shape[0], X.shape[1], X.shape[2], X.shape[3]))
    for k in range(X.shape[0]):
        if k % 10000 == 1999:
            msg = 'quantization number: ' + str(k)
            print(msg)
        image = X[k]
        pixel_idx = 0
        for c in range(X.shape[1]):
            for i in range(X.shape[2]):
                for j in range(X.shape[3]):
                    dist_range = np.power(2, length - 1) - 2
                    while image[c][i][j] < distribution[dist_range][c][i][j] and dist_range > -1:
                        dist_range -= 1
                    input_dataset[k][c][i][j] = quant[dist_range + 1][c][i][j]
        del image
    return input_dataset

#lucy's code
H = []
length = 3
p_value = np.power(2, length - 1)
for i in range(length):
    H.append(generateHmatrix(i))
temp_iterator = iter(temp_train_loader)
X, y = next(temp_iterator)

#lucy's code
X, U, mean = whitening(X)
X = torch.from_numpy(X).float()
dist = distribution_matrix(X, p_value - 1)
print('dist done')
quant = quantized_matrix(X, p_value-1,dist)
print('quant done')
X = quantization(X,dist,quant)
mapping_order = generateMappingorder(X, length)
#irtaza's code
Z = generateKey(3)
num_inputs = (np.power(2, length - 1)-1) * X.shape[1] * X.shape[2] * X.shape[3]
X = encoding(X, dist, Z, length, H, mapping_order, num_inputs)

#lucy's code
encoded_dataset_train = data_utils.TensorDataset(X, y)

encoded_train_loader = data_utils.DataLoader(encoded_dataset_train, batch_size=train_batch_size, shuffle=True)
#lucy's code
temp_iterator = iter(temp_test_loader)
X, y = next(temp_iterator)
images = X
images = images - mean
images = images.reshape(images.shape[0], images.shape[1] * images.shape[2] * images.shape[3])
images = np.dot(images, U)
images = images.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
X = images
X = torch.from_numpy(X).float()
X = quantization(X,dist,quant)
X = encoding(X, dist, Z, length, H, mapping_order, num_inputs)

#lucy's code
encoded_dataset_test = data_utils.TensorDataset(X, y)
encoded_test_loader = data_utils.DataLoader(encoded_dataset_test, batch_size=len(fmnist_test), shuffle=False)

#lucy's code
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(784*3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )


    def forward(self, x):
        """Perform forward."""
        # x = self.init_layer(x)
        # x = x.view(x.size(0), 1, 28, 28)
        # x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
encoded_model = Net().to(device)

messages = []

#lucy's code
def epoch(loader, model, opt=None):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0., 0.
    # i = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

#lucy's code
opt = optim.SGD(encoded_model.parameters(), lr=0.001)
for t in range(200):
    if t == 40:
        for param_group in opt.param_groups:
            param_group["lr"] = 0.0002
    if t == 80:
        for param_group in opt.param_groups:
            param_group["lr"] = 0.0001
    if t == 120:
        for param_group in opt.param_groups:
            param_group["lr"] = 0.00003
    if t == 140:
        for param_group in opt.param_groups:
            param_group["lr"] = 0.00002
    train_err, train_loss = epoch(encoded_train_loader, encoded_model, opt)
    test_err, test_loss = epoch(encoded_test_loader, encoded_model)
    msg = str(t) + ':' + str(1-test_err)
    
#irtaza's code
import random
def random_encrypted_image():
    #this generates a random tensor of values between 0 and 1 and has same dimensions as an unencrypted MNIST image
    random_tensor = torch.rand((2,1,28,28))
    #encryption techniques applied
    random_tensor, U, mean = whitening(random_tensor)
    dist = distribution_matrix(random_tensor, p_value - 1)
    quant = quantized_matrix(random_tensor, p_value-1,dist)
    random_tensor = quantization(random_tensor,dist,quant)
    mapping_order = generateMappingorder(random_tensor, length)
    #a key is generated which has 28x28xn values of -1 and 1 placed randomly. It takes in a parameter value of
    # 3 because we want our key to have dimensions 28x28x3
    Z = generateKey(3)
    num_inputs = (np.power(2, length - 1)-1) * random_tensor.shape[1] * random_tensor.shape[2] * random_tensor.shape[3]
    #encoding process where all the values of the image will now be either -1 or 1 after doing bitwise xor
    random_tensor, mappingofmatrix = encoding(random_tensor, dist, Z, length, H, mapping_order, num_inputs)
    #a new encrypted image is returned along with the key and mapped hadamard matrix which we will be using for key extraction
    return np.array(random_tensor), Z, mappingofmatrix

#irtaza's code
def features_randomizer(x, k):
    #k number of indexes generated at random
    indexes = np.random.randint(0, x.shape[0], size=k)
    #we go to each index and check what value is at that index
    for i in range(len(indexes)):
        #if the value is 1, we make it -1
        if (x[indexes[i]]) == 1:
            x[indexes[i]] = -1
        #if value is -1, we make it 1
        else:
            x[indexes[i]] = 1
    #we returned the changed image
    return x

#irtaza's code
def synthesize(selectedclass, k_max):
    image, key, mappingofmatrix = random_encrypted_image() #initializing a random record
    initial_y_c = 0 #called y*c in shokri's paper
    number_of_rejections = 0 #called j in shokri's paper
    maximum_number_of_rejections = 5 #called rejmax in shokri's paper
    k_min = 1
    k = k_max
    total_iterations = 1500
    probability_cutoff = 0.8 #called confmin in shokri's paper
    counter = 0
    for i in range(total_iterations):
        counter+=1
        #we query the model
        opt = optim.SGD(encoded_model.parameters(), lr=0.001)
        if total_iterations == 40:
            for param_group in opt.param_groups:
                param_group["lr"] = 0.0002
        if total_iterations == 80:
            for param_group in opt.param_groups:
                param_group["lr"] = 0.0001
        train_err, train_loss = epoch(encoded_train_loader, encoded_model, opt)
        test_err, test_loss = epoch(encoded_test_loader, encoded_model)
        y_c  = 1-test_err #checking the probability that the image belongs to the class
        if y_c >= initial_y_c:
            if (y_c > probability_cutoff and selectedclass >= np.argmax(opt)):
                #image is accepted because it increases the hill climbing objective (probability generated by model is greater than the cutoff)
                return image,key,mappingofmatrix
            new_image = image
            initial_y_c = y_c
            number_of_rejections = 0
        else:
            number_of_rejections += 1
            if number_of_rejections > maximum_number_of_rejections:
                k = max(k_min, int(np.ceil(k/2)))
                n_rejects = 0
        #changing k randomly selected features of the image
        image = features_randomizer(new_image, k)
    return False

#irtaza's code
def keyextraction(selectedclass, k):
    image,key,mappingofmatrix = synthesize(selectedclass,k)
    mappingofmatrix = np.array(mappingofmatrix)
    potentialkey = xor(mappingofmatrix, image)
    if np.array_equal(key[0],potentialkey):
        return True
    else:
        return False
#to check if the keys are the same we run this method which takes in two parameters, the class that we want
#to test for (0-9) and the value of k which is used in the data synthesis algorithm and is defined as the maximum number
# of randomly selected features we change
keyextraction(6,3)
#if we want to simply synthesize data without checking if the keys are the same we run this statement
# the method returns 3 things, the synthesized image, the key, and the mapping of matrix. If we only
# care about the synthesized image, that will be at x[0]. (Method returns 3 things for x. x[0] is synthesized image,
# x[1] is key, and x[2] is mapping of matrix.
x = synthesize(6,3)
print(x[0])





