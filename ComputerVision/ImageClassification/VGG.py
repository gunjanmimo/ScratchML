# %%
import torch
import torch.nn as nn 


class VGG16(nn.Module):
    def __init__(self,n_channel:int,n_classes:int) :
        super(VGG16,self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=n_channel,out_channels=64,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        
        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        
        
        self.block_5 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.linear_block = nn.Sequential(
            nn.Linear(in_features=512*7*7,out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096,out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096,out_features=n_classes),
        )
        
    def forward(self,x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = x.view(x.size(0),-1)
        x = self.linear_block(x)
        return x

# %%
if __name__=="__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.randn(1,3,224,224)
    model = VGG16(n_channel=3,n_classes=10)
    # model
    
    predict  = model(image)
    #print(predict.shape)
# %%
