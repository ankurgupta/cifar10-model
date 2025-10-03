dropout_value = 0.1
# CNN Model
class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()

        self.convblock1 = nn.Sequential(
            #input size : 3x32x32
            # Block - 1, Layer - 1
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(), # ouptput: 32, RF: 3
            nn.Dropout(dropout_value),

             # #Layer - 3
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(), # output: 32, RF: 7
            nn.Dropout(dropout_value),

            # Layer 2 - depthwise separable convolution
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, groups=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(), #output: 32, RF: 5


            # Stride-2 Convolution - downsampling
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(), # output: 16, RF: 9
            nn.Dropout(dropout_value),
        )

        self.transition1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU() #output : 16, RF:9
        )
        self.convblock2 = nn.Sequential(

            #Block - 2, Layer - 1
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(), #output: 16, RF: 13
            nn.Dropout(dropout_value),

            # Block - 2, Layer - 2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(), #output: 16, RF: 17
            nn.Dropout(dropout_value),

            # stride = 2 convolution downsampling
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(), #output: 8, RF: 21
            nn.Dropout(dropout_value),
        )

        self.transition2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU() #output:8, RF:21
        )

        self.convblock3 = nn.Sequential(

            #Block - 3, Layer - 1

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(), #output: 8, RF: 29
            nn.Dropout(dropout_value),

            # Block - 3, Layer - 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(), #output: 8, RF: 37
            nn.Dropout(dropout_value),

             # Block - 3, Layer - 2 downsample here
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(), #output: 8, RF: 37
            nn.Dropout(dropout_value),

        )

        self.transition3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.convblock4 = nn.Sequential(

            #Block - 4, Layer Dilated Convolution
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(), #output: 4, RF: 59
        )

        self.outputblock = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            #nn.AvgPool2d(kernel_size=4),
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )



    def forward(self, x):
        x = self.convblock1(x)
        x = self.transition1(x)
        x = self.convblock2(x)
        x = self.transition2(x)
        x = self.convblock3(x)
        x = self.transition3(x)
        x = self.convblock4(x)
        x = self.outputblock(x)
        x = x.view(-1, 10)
        return x
