import torch 
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F

torch.manual_seed(43)
np.random.seed(43)
random.seed(43)


class Classifier_v1(nn.Module):
    def __init__(self):
        super(Classifier_v1, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 5)
        self.conv2 = torch.nn.Conv2d(4, 8, 5)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(8 * 13 * 13, 20)
        self.fc2 = torch.nn.Linear(20, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    


class Classifier_v2(nn.Module):
    def __init__(self):
        super(Classifier_v2, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5)
        self.conv2 = torch.nn.Conv2d(16, 32, 5)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(32 * 13 * 13, 20)
        self.fc2 = torch.nn.Linear(20, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    


class Classifier_v3(nn.Module):
    def __init__(self):
        super(Classifier_v3, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 5)
        self.conv2 = torch.nn.Conv2d(64, 128, 5)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(128 * 13 * 13, 20)
        self.fc2 = torch.nn.Linear(20, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    


class Classifier_v4(nn.Module):
    def __init__(self):
        super(Classifier_v4, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5)
        self.conv2 = torch.nn.Conv2d(16, 32, 5)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(32 * 13 * 13, 100)
        self.fc2 = torch.nn.Linear(100, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    


class Classifier_v5(nn.Module):
    def __init__(self):
        super(Classifier_v5, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5)
        self.conv2 = torch.nn.Conv2d(16, 32, 5)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(32 * 13 * 13, 200)
        self.fc2 = torch.nn.Linear(200, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    

class Classifier_v6(nn.Module):
    def __init__(self):
        super(Classifier_v6, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 5)
        self.conv2 = torch.nn.Conv2d(128,256, 5)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(256 * 13 * 13, 20)
        self.fc2 = torch.nn.Linear(20, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    
class Classifier_v7(nn.Module):
    def __init__(self):
        super(Classifier_v7, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 256, 5)
        self.conv2 = torch.nn.Conv2d(256, 512, 5)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(512 * 13 * 13, 20)
        self.fc2 = torch.nn.Linear(20, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x



class Classifier_v8(nn.Module):
    def __init__(self):
        super(Classifier_v8, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 5, stride=2)
        self.conv2 = torch.nn.Conv2d(64, 128, 5, stride=2)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(128 * 3 * 3, 20)
        self.fc2 = torch.nn.Linear(20, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    

class Classifier_v9(nn.Module):
    def __init__(self):
        super(Classifier_v9, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3)
        self.conv2 = torch.nn.Conv2d(64, 128, 3)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(128 * 14 * 14, 20)
        self.fc2 = torch.nn.Linear(20, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    

class Classifier_v10(nn.Module):
    def __init__(self):
        super(Classifier_v10, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 7)
        self.conv2 = torch.nn.Conv2d(64, 128, 7)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(128 * 11 * 11, 20)
        self.fc2 = torch.nn.Linear(20, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    

class Classifier_v11(nn.Module):
    def __init__(self):
        super(Classifier_v11, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5)
        self.conv2 = torch.nn.Conv2d(16, 32, 5)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(32 * 13 * 13, 2)
        # self.fc2 = torch.nn.Linear(100, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.softmax(self.fc1(x))
        # x = self.softmax(self.fc2(x))

        return x

class Classifier_v12(nn.Module):
    def __init__(self):
        super(Classifier_v12, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5)
        self.conv2 = torch.nn.Conv2d(16, 32, 5)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(32 * 13 * 13, 20)
        self.fc2 = torch.nn.Linear(20, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    
class Classifier_v13(nn.Module):
    def __init__(self):
        super(Classifier_v13, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5)
        self.conv2 = torch.nn.Conv2d(16, 32, 5)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(32 * 13 * 13, 50)
        self.fc2 = torch.nn.Linear(50, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    
class Classifier_v14(nn.Module):
    def __init__(self):
        super(Classifier_v14, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5)
        self.conv2 = torch.nn.Conv2d(16, 32, 5)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(32 * 13 * 13, 200)
        self.fc2 = torch.nn.Linear(200, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    


class Classifier_v15(nn.Module):
    def __init__(self):
        super(Classifier_v15, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 9)
        self.conv2 = torch.nn.Conv2d(64, 128, 9)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(128 * 10 * 10, 20)
        self.fc2 = torch.nn.Linear(20, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    


class Classifier_v16(nn.Module):
    def __init__(self):
        super(Classifier_v16, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 11)
        self.conv2 = torch.nn.Conv2d(64, 128, 11)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(128 * 8 * 8, 20)
        self.fc2 = torch.nn.Linear(20, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    


class Classifier_v17(nn.Module):
    def __init__(self):
        super(Classifier_v17, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 13)
        self.conv2 = torch.nn.Conv2d(64, 128, 13)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(128 * 7 * 7, 20)
        self.fc2 = torch.nn.Linear(20, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    


class Classifier_v18(nn.Module):
    def __init__(self):
        super(Classifier_v18, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 15)
        self.conv2 = torch.nn.Conv2d(64, 128, 15)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(128 * 5 * 5, 20)
        self.fc2 = torch.nn.Linear(20, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    
class Classifier_v19(nn.Module):
    def __init__(self):
        super(Classifier_v19, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 13)
        self.conv2 = torch.nn.Conv2d(64, 128, 13)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(128 * 7 * 7, 50)
        self.fc2 = torch.nn.Linear(50, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    

class Classifier_v20(nn.Module):
    def __init__(self):
        super(Classifier_v20, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 13)
        self.conv2 = torch.nn.Conv2d(8, 16, 13)
        self.conv3 = torch.nn.Conv2d(16, 32, 3)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(32 * 2 * 2, 50)
        self.fc2 = torch.nn.Linear(50, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x

class Classifier_v21(nn.Module):
    def __init__(self):
        super(Classifier_v21, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 13)
        self.conv2 = torch.nn.Conv2d(16, 32, 13)
        self.conv3 = torch.nn.Conv2d(32, 64, 3)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(64 * 2 * 2, 50)
        self.fc2 = torch.nn.Linear(50, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    
class Classifier_v22(nn.Module):
    def __init__(self):
        super(Classifier_v22, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 13)
        self.conv2 = torch.nn.Conv2d(32, 64, 13)
        self.conv3 = torch.nn.Conv2d(64, 128, 3)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(128 * 2 * 2, 50)
        self.fc2 = torch.nn.Linear(50, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x

class Classifier_v23(nn.Module):
    def __init__(self):
        super(Classifier_v23, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 13)


        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(8 * 26 * 26, 50)
        self.fc2 = torch.nn.Linear(50, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    
class Classifier_v24(nn.Module):
    def __init__(self):
        super(Classifier_v24, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 13)


        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(16 * 26 * 26, 50)
        self.fc2 = torch.nn.Linear(50, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x
    
class Classifier_v25(nn.Module):
    def __init__(self):
        super(Classifier_v25, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 13)


        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(32 * 26 * 26, 50)
        self.fc2 = torch.nn.Linear(50, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x


class ClassifierHyperparam_v1(nn.Module):
    def __init__(self):
        super(ClassifierHyperparam_v1, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,6,9)
        self.conv2 = torch.nn.Conv2d(6, 10, 7)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.2)

        self.fc1 = torch.nn.Linear(10 * 11 * 11, 62)
        self.fc2 = torch.nn.Linear(62, 17)
        self.fc3 = torch.nn.Linear(17, 2)

        self.batchnorm1 = torch.nn.BatchNorm2d(6)
        self.batchnorm2 = torch.nn.BatchNorm2d(10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.batchnorm1(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.batchnorm2(x)

        # x = x.view(x.size(0), -1)
        x = torch.nn.Flatten()(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        # x = self.softmax(self.fc3(x))
        x = self.softmax(self.fc3(x))
        # print(type(output))
        # print(output.shape)

        return x


class ClassifierHyperparam_v2(nn.Module):
    def __init__(self):
        super(ClassifierHyperparam_v2, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,18,9)
        self.conv2 = torch.nn.Conv2d(18, 19, 5)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.dropout = torch.nn.Dropout(0.10)

        self.fc1 = torch.nn.Linear(19 * 12 * 12, 74)
        self.fc2 = torch.nn.Linear(74, 73)
        self.fc3 = torch.nn.Linear(73, 2)

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        # x = x.view(x.size(0), -1)
        x = torch.nn.Flatten()(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.softmax(self.fc3(x))

        return x


class ClassifierHyperparam_v3(nn.Module):
    def __init__(self):
        super(ClassifierHyperparam_v3, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,229,9)
        self.conv2 = torch.nn.Conv2d(229, 166, 3)
        self.conv3 = torch.nn.Conv2d(166, 201, 5)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.dropout = torch.nn.Dropout(0.50)

        self.fc1 = torch.nn.Linear(201 * 4 * 4, 204)
        self.fc2 = torch.nn.Linear(204, 131)
        self.fc3 = torch.nn.Linear(131, 112)
        self.fc4 = torch.nn.Linear(112, 2)

        self.bn1 = torch.nn.BatchNorm2d(229)
        self.bn2 = torch.nn.BatchNorm2d(166)
        self.bn3 = torch.nn.BatchNorm2d(201)


    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.pool(self.relu(self.conv3(x)))
        x = self.bn3(x)

        # x = x.view(x.size(0), -1)
        x = torch.nn.Flatten()(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.softmax(self.fc4(x))

        return x
