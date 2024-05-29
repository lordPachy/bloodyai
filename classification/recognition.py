from d2l import torch as d2l
import numpy as np
from torch import nn
import torchvision
from torchvision.transforms import v2
import torch
import numpy as np
import torch.utils
import torch.utils.data


# -------------------
# CODE OF THE MODEL

class Residual(nn.Module):
    """Residual block: it is employ in class ResNet"""

    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = nn.functional.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return nn.functional.relu(Y)


class ResNet(d2l.Classifier):
    """The generic ResNet class, with hyperparameters to be specified"""

    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i + 2}', self.block(*b, first_block=(i == 0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)
        ))

        self.net.apply(d2l.init_cnn)

    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def block(self, num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels))

        return nn.Sequential(*blk)

    def configure_optimizers(self):
        return torch.optim.Adagrad(self.parameters(), lr=self.lr)

    def accuracy(self, Y_hat, Y, averaged=True):
        Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
        compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
        self.validation_results['correct'].append(d2l.reshape(Y, -1))
        self.validation_results['predict'].append(preds)
        return d2l.reduce_mean(compare) if averaged else compare


class ResNet18(ResNet):
    """The ResNet18 class with its own hyperparameters."""
    validation_results = {'correct': [], 'predict': []}

    def __init__(self, lr=0.1, num_classes=10):
        super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)), lr, num_classes)


# ----------------------
# CODE FOR THE INFERENCE

# THIS IS VITAL!!!! KEEP IT IN THIS ORDER.
labels = ['Artifact', 'Burst', 'Eosinophil', 'Large Lymph', 'Monocyte', 'Neutrophil', 'Small Lymph']


# 1. RUN PREPARE_MODEL BEFORE ANY INFERENCE
def prepare_ResNet18(path_to_parameters):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = ResNet18()
    model.load_state_dict(torch.load(path_to_parameters, map_location = torch.device(device)))
    model = model.to(device)
    model.eval()

    return model


# 2. RUN THIS METHOD TO DO INFERENCE
def inference_with_ResNet18(model, labels, img):
    transf = v2.Compose([v2.Resize((224, 224)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    x = torch.from_numpy(img)
    c, h, w = x.shape
    x = torch.reshape(input=x, shape=(1, c, h, w))
    x = transf(x)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    x = x.to(device)
    preds = model.forward(x)
    preds = preds.to('cpu')
    i = np.argmax(preds.detach().numpy())
    return labels[i]


# 3. SUBSTITUTE HERE YOUR PATH TO THE WEIGHTS...
#model = prepare_ResNet18('./parameters/state_dict_model_split_scheduler.pt')
# ...AND THE TEST IMAGE
#print(inference_with_ResNet18(model, labels, torchvision.io.read_image('./test.jpg')))