import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import statistics

class Context2X(nn.Module):
    def __init__(self, cfg: dict, num_classes = 10, init_weights = True):
        super(Context2X, self).__init__()

        self.context1 = Context(cfg, 0)
        self.context2 = Context(cfg, 1)
        self.context3 = Context(cfg, 2)
        self.context4 = Context(cfg, 3)
        self.context5 = Context(cfg, 4)

        self.sequence = nn.Sequential(
            self.context1,
            self.context2,
            self.context3,
            self.context4,
            self.context5,
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 3 * 3 , 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

        if init_weights:
            self._initialize_weights()
        

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        out = self.sequence(x)
        out = out[0]
        print(out.shape)
        out = out.view(-1, 64 * 3 * 3)
        out = self.classifier(out)
        return out


class Context(nn.Module):
    def __init__(self, cfg, index):
        super(Context, self).__init__()

        self.kernels = [5, 3, 1]
        self.convs = []
        self.index = index
        
        self.make_convs(cfg, index)

        self.conv1 = self.convs[0]
        self.conv2 = self.convs[1]
        self.conv3 = self.convs[2]

        self.origin_conv = nn.Sequential(
            nn.Conv2d(3, cfg['channel'][index], kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(cfg['channel'][index]),
            nn.ReLU()
        )
        
        if index > 0:
            self.before_conv = nn.Sequential(
                nn.Conv2d(cfg['channel'][index], cfg['channel'][index], kernel_size = 1, stride = 1, padding = 0),
                nn.BatchNorm2d(cfg['channel'][index]),
                nn.ReLU()
            )
        
        if index == 0:
            self.conv4 = nn.Sequential(
                            nn.Conv2d(cfg['channel'][index] * 4, cfg['channel'][index + 1], kernel_size = 2, stride = 2, padding = 1),
                            nn.BatchNorm2d(cfg['channel'][index + 1]),
                            nn.ReLU(),
                        )
        else:
            self.conv4 = nn.Sequential(
                            nn.Conv2d(cfg['channel'][index] * 5, cfg['channel'][index + 1], kernel_size = 2, stride = 2, padding = 1),
                            nn.BatchNorm2d(cfg['channel'][index + 1]),
                            nn.ReLU(),
                        )


    def make_convs(self, cfg, index):

        if index == 0:

            for i, kernel in enumerate(self.kernels):
                self.convs.append(
                    nn.Sequential(
                        nn.Conv2d(cfg['channel'][index] * (i + 1), cfg['channel'][index], kernel, stride = 2, padding = 1),
                        nn.BatchNorm2d(cfg['channel'][index]),
                        nn.ReLU(),
                    ))

        else:

            for i, kernel in enumerate(self.kernels):
                self.convs.append(
                    nn.Sequential(
                        nn.Conv2d(cfg['channel'][index] * (i + 2), cfg['channel'][index], kernel, stride = 2, padding = 1),
                        nn.BatchNorm2d(cfg['channel'][index]),
                        nn.ReLU(),
                    ))


    def _catter(self, outs):
        median_sizes = []
        for out in outs:
            median_sizes.append(out.size(-1))
        median_size = int(statistics.median(median_sizes))
        
        go = []
        for out in outs:
            go.append(nn.AdaptiveMaxPool2d((median_size, median_size))(out))

        go = torch.cat(go, 1)

        return go
            

    def forward(self, x):

        if self.index == 0:
            outs = []
            origin = x

            #Append origin to outs
            out = x
            outs.append(self.origin_conv(origin))

            #Append first out to outs
            out = self.conv1(origin)
            outs.append(out)

            cat = self._catter(outs)

            out = self.conv2(cat)
            outs.append(out)

            cat = self._catter(outs)

            out = self.conv3(cat)
            outs.append(out)

            out = self._catter(outs)

            out = self.conv4(out)

            return out, origin
        else:
            outs = []

            origin = x[1]
            out = x[0]

            outs.append(self.origin_conv(origin))
            outs.append(self.before_conv(out))
            
            out = self._catter(outs)

            #Append first out to outs
            out = self.conv1(out)
            outs.append(out)

            cat = self._catter(outs)

            out = self.conv2(cat)
            outs.append(out)

            cat = self._catter(outs)

            out = self.conv3(cat)
            outs.append(out)

            out = self._catter(outs)

            out = self.conv4(out)

            return out, origin


cfg = {
    'channel': [3, 64, 128, 128, 128, 64]
}

def context2x(cfg = cfg, num_classes = 10, init_weights = True):
    return Context2X(cfg, num_classes, init_weights)