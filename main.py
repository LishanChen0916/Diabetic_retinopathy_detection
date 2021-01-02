import numpy as np
from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torchvision.models
from sklearn.metrics import confusion_matrix
import itertools

# Without 1X1 convolution
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, downsampling=None):
        super(BasicBlock, self).__init__()

        # For zero padding
        padding_size = kernel_size // 2
        self.activation = nn.ReLU(inplace=True)
        self.basicBlock = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                padding=padding_size, 
                stride=stride, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                padding=padding_size, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.downsampling = downsampling

    def forward(self, x):
        residual = x
        output = self.basicBlock(x)

        if self.downsampling is not None:
            residual = self.downsampling(x)

        output += residual

        return self.activation(output)

# Add 1X1 convolution at the start and end of network
class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, downsampling=None):
        super(BottleneckBlock, self).__init__()

        # For zero padding
        padding_size = kernel_size // 2
        self.activation = nn.ReLU(inplace=True)
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=1, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            self.activation,

            nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                stride=stride,
                padding=padding_size, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            self.activation,

            nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels*self.expansion, 
                kernel_size=1, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels*self.expansion),
        )
        self.downsampling = downsampling

    def forward(self, x):
        residual = x
        
        output = self.bottleneckBlock(x)

        if self.downsampling is not None:
            residual = self.downsampling(x)

        output += residual

        return self.activation(output)

class ResNet(nn.Module):
    def __init__(self, block, layers_sizes):
        super(ResNet, self).__init__()

        self.current_in_channels = 64

        # Determine the block is BasicBlock or BottleneckBlock
        self.block = block

        # Determine the sizes of layers
        self.layers_sizes = layers_sizes

        self.conv0 = nn.Sequential(
            nn.Conv2d(
                # R, G, B
                in_channels=3, 
                out_channels=self.current_in_channels, 
                kernel_size=7,
                stride=2, 
                # Kernel_size // 2 (For zero padding)
                padding=3, 
                bias=False
            ),
            nn.BatchNorm2d(self.current_in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3, 
                stride=2, 
                padding=1
            ),
        )
        
        channels = self.current_in_channels
        for i, l in enumerate(self.layers_sizes):
            self.add_module("conv" + str(i+1),
                nn.Sequential(
                    *self._make_layers(l, channels, stride=(2 if i!=0 else 1))
                )
            )
            channels *= 2
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classify = nn.Linear(
                in_features=self.current_in_channels, 
                out_features=5, 
        )

    def _make_layers(self, num_layers, in_channels, stride=1):
        downsampling = None
        if stride != 1 or self.current_in_channels != in_channels * self.block.expansion:
            downsampling = nn.Sequential(
                nn.Conv2d(
                    self.current_in_channels, in_channels * self.block.expansion,
                    kernel_size = 1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(in_channels * self.block.expansion)
            )

        layers = []
        layers.append(self.block(self.current_in_channels, in_channels, stride=stride, downsampling=downsampling))
        self.current_in_channels = in_channels * self.block.expansion
        for i in range(1, num_layers):
            layers.append(self.block(self.current_in_channels, in_channels))

        return layers

    def forward(self, input):
        x = self.conv0(input)

        for convID in range(1, len(self.layers_sizes)+1):
            x = getattr(self, 'conv'+str(convID))(x)
        
        x = self.avgpool(x)

        # Flatten
        x = x.view(x.size(0), -1)
        self.output = self.classify(x)

        return self.output

class PretrainResNet(nn.Module):
    def __init__(self, num_classes, num_layers):
        super(PretrainResNet, self).__init__()
        
        pretrained_model = torchvision.models.__dict__[
            'resnet{}'.format(num_layers)](pretrained=True)
        
        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classify = nn.Linear(
            pretrained_model._modules['fc'].in_features, num_classes
        )
        
        del pretrained_model
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        
        return x

def trainAndTestModels(train_dataset, test_dataset, models, 
    epochs, batch_size, learning_rate, optimizer=optim.SGD, loss_function=nn.CrossEntropyLoss()):
    
    accuracyOfTrainingPerModels = {}

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    for key, model in models.items():
        # Initialize the trainAcc and testAcc arrays as zeros
        trainAcc = np.zeros(epochs)
        testAcc = np.zeros(epochs)
        test_y = []
        pred_y = []

        for epoch in range(epochs):
            # train the models
            model.train()
            for train_inputs, train_labels in train_loader:
                train_inputs = train_inputs.to(device)
                train_labels = train_labels.to(device).long().view(-1)

                optimizer(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4).zero_grad()
                outputs = model.forward(train_inputs)

                loss = loss_function(outputs, train_labels)
                loss.backward()

                # Update the parameters
                optimizer(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4).step()

                # Pick the maximum value of each row and return its index
                trainAcc[epoch] += (torch.max(outputs, 1)[1] == train_labels).sum().item()
            
            print("Epochs[%3d/%3d] Loss : %f" % (epoch, epochs, loss))
            trainAcc[epoch] = trainAcc[epoch] * 100 / len(train_dataset)
            accuracyOfTrainingPerModels.update([(key+'_train', trainAcc)])
            
            # test the models
            model.eval()
            with torch.no_grad():
                for test_inputs, test_labels in test_loader:
                    test_inputs = test_inputs.to(device)
                    test_labels = test_labels.to(device).long().view(-1)

                    outputs = model.forward(test_inputs)
                    testAcc[epoch] += (torch.max(outputs, 1)[1] == test_labels).sum().item()

                    test_y = np.append(test_y, test_labels.to(torch.device('cpu')).numpy())
                    pred_y = np.append(pred_y, torch.max(outputs, 1)[1].to(torch.device('cpu')).numpy())

            testAcc[epoch] = testAcc[epoch] * 100 / len(test_dataset)
            accuracyOfTrainingPerModels.update([(key+'_test', testAcc)])
        
            # Plot confusion matrix at the end of epoch
            if epoch == epochs - 1:
                plotConfusionMatrix(key, test_y, pred_y)

        print(accuracyOfTrainingPerModels)

        # Free cache
        torch.cuda.empty_cache()

    return accuracyOfTrainingPerModels

def plotTheResult(network, accuracy):
    plt.figure(figsize=(8,4.5))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.ylim(60, 100)

    for key, value in accuracy.items():
        plt.plot(
            value, 
            '--' if 'test' in key else '-',
            label=key
        )
        plt.legend(loc='best')
        plt.title(network)
        plt.xticks(range(len(value)))

    #plt.show()
    filename = network + "_Accuracy"
    plt.savefig(filename + ".png")

def plotConfusionMatrix(title, test_labels, test_pred, normalize=True):
    cm = confusion_matrix(test_labels, test_pred)
    np.set_printoptions(precision=2)
    
    plt.figure()
    classes = ['0', '1', '2', '3', '4']

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    #plt.show()
    filename = title + "_ConfusionMatrix"
    plt.savefig(filename + ".png")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    augmentation = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ]

    # Train with data augmentation
    train_dataset = RetinopathyLoader('data', 'train', augmentation=augmentation)
    test_dataset = RetinopathyLoader('data', 'test')

    # ResNet18
    models = {"ResNet18" : ResNet(BasicBlock, [2, 2, 2 ,2]).to(device),
        "ResNet18_Pretrained" : PretrainResNet(5, 18).to(device)
    }

    ResNet18Acc = trainAndTestModels(train_dataset, test_dataset, models,
        epochs=10, batch_size=16, learning_rate=1e-3)

    for key, value in ResNet18Acc.items():
    	print(key, np.amax(value))

    plotTheResult("ResNet18", ResNet18Acc)


    # ResNet50
    models = {"ResNet50" : ResNet(BottleneckBlock, [3, 4, 6, 3]).to(device), 
        "ResNet50_Pretrained" : PretrainResNet(5, 50).to(device)
    }

    ResNet50Acc = trainAndTestModels(train_dataset, test_dataset, models, 
        epochs=10, batch_size=8, learning_rate=1e-3)

    for key, value in ResNet50Acc.items():
    	print(key, np.amax(value))

    plotTheResult("ResNet50", ResNet50Acc)