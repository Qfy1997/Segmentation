import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
import os
import time



Semantic_label_color_dict={0:(0,0,0),1:(176, 196, 222), 2:(255, 0, 255), 3:(30, 144, 255), 4:(250, 128, 114), 5:(238, 232, 170), 6:(255, 20, 147), 7:(123, 104, 238), 8:(255, 192, 203), 9:(105, 105, 105), 10:(85, 107, 47), 11:(205, 133, 63), 12:(0, 0, 128), 13:(50, 205, 50), 14:(127, 0, 127), 15:(176, 48, 96), 16:(128, 0, 0), 17:(72, 61, 139), 18:(0, 128, 0), 19:(60, 179, 113), 20:(0, 139, 139), 21:(255, 0, 0), 22:(255, 140, 0), 23:(255, 215, 0), 24:(0, 255, 0), 25:(148, 0, 211), 26:(0, 250, 154), 27:(220, 20, 60), 28:(0, 255, 255), 29:(0, 191, 255), 30:(0, 0, 255), 31:(173, 255, 47), 32:(218, 112, 214), 33:(60, 46, 70)}
# GEO_label_color_dict={-1:(0,0,0),1:(255, 0, 0),2:(0, 255, 0),3:(0, 0, 255)}
torch.manual_seed(66)
np.random.seed(0)



class UpResNet(nn.Module):
    def __init__(self,scale_factor=1,in_channels=3,out_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,64,9,padding=4)
        self.res_blocks = nn.Sequential(*[ResBlock(64) for _ in range(64)])
        self.upsample = nn.Sequential(
            nn.Conv2d(64,64*scale_factor**2,3,padding=1),
            nn.PixelShuffle(scale_factor)
        )
        
        self.conv2 = nn.Conv2d(64,4,9,padding=4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        feat = self.relu(self.conv1(x))
        feat = self.res_blocks(feat)
        feat = self.upsample(feat)
        out = self.sigmoid(self.conv2(feat))
        
        return out


class ResBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels,channels,3,padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels,channels,3,padding=1)
    
    def forward(self,x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out+residual


if __name__=='__main__':
    # train_path="your file path"
    # test_path = "your file path"
    # semantic_labels_path="your file path"
    files = os.listdir(train_path)
    model=UpResNet().to("cuda:0")
    optimizer = optim.Adam(model.parameters(),lr=0.00001)
    criterion = nn.BCELoss()

    with open("TestSet1.txt",'r') as f:
        data = f.readlines()
    test_list=[]
    train_list=[]
    for i in range(len(data)):
        test_list.append(data[i].strip().split('\\')[-1])
    print(len(test_list))
    for file in files:
        if file not in test_list:
            train_list.append(file)
        # print(file)
    # print(len(train_list))

    
    # for epoch in range(200):
    #     start=time.time()
    #     avg_loss=0
    #     for i in range(len(train_list)):
    #         pre_file=train_path+train_list[i]
    #         semanticBceLabel_path=semantic_labels_path+train_list[i][:-4]+".npy"
    #         semanticBceLabel=np.load(semanticBceLabel_path)
    #         semanticBceLabel=semanticBceLabel.astype(np.float32)
    #         image = cv2.imread(pre_file, cv2.IMREAD_COLOR)
    #         image_norm = np.zeros(image.shape, dtype=np.float32)
    #         cv2.normalize(image, image_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #         input_image = torch.from_numpy(image_norm)
    #         input_image = input_image.unsqueeze(0)
    #         input_image = input_image.permute(0,3,1,2)
    #         input_image = input_image.float()
    #         input_image = input_image.to("cuda:0")
    #         semanticLabel=torch.from_numpy(semanticBceLabel)
    #         semanticLabel = semanticLabel.float()
    #         print("smantic label shape:",semanticBceLabel.shape)
    #         semanticLabel = semanticLabel.to("cuda:0")
    #         optimizer.zero_grad()
    #         out = model(input_image)
    #         loss = criterion(out,semanticLabel)
    #         loss.backward()
    #         # print("batch:",i," loss:",loss.cpu().detach().numpy())
    #         optimizer.step()
    #         avg_loss+=loss
    #         # print(input_image.shape)
    #         # print(pre_file)
    #         break
    #     # avg_loss/=2488
    #     # end=time.time()
    #     # print("epoch:",epoch," loss:",avg_loss.cpu().detach().numpy()," time:",end-start,"s")
    #     break
    # torch.save(model,'UpResNet200Geo.pth')

    model=torch.load("UpResNet200Geo.pth")
    correct=0
    total=0
    
    pre_data_lis=[]
    for i in range(len(test_list)):
        pre_file=train_path+test_list[i]
        image = cv2.imread(pre_file, cv2.IMREAD_COLOR)
        # print("image shape:",image.shape)
        image_norm = np.zeros(image.shape, dtype=np.float32)
        cv2.normalize(image, image_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        input_image = torch.from_numpy(image_norm)
        input_image = input_image.unsqueeze(0)
        input_image = input_image.permute(0,3,1,2)
        input_image = input_image.float()
        input_image = input_image.to("cuda:0")
        semanticBceLabel_path=semantic_labels_path+test_list[i][:-4]+".npy"
        semanticBceLabel=np.load(semanticBceLabel_path)
        out = model(input_image)
        
        label_index = np.argmax(semanticBceLabel,axis=1)
        predict_index = np.argmax(out.cpu().detach().numpy(),axis=1)
        row_np = np.random.randint(low=0,high=255,size=(4),dtype='int')
        col_np = np.random.randint(low=0,high=255,size=(4),dtype='int')
        # print(label_index.shape)
        # print(predict_index.shape)
        
        # batch_index row col logits(34) predict label
        
        
        for row in range(4):
            for col in range(4):
                pre_data=[i,row_np[row],col_np[col]]
                for index in range(4):
                    pre_data.append(out.cpu().detach().numpy()[0][index][row_np[row]][col_np[col]])
                pre_data.append(predict_index[0][row_np[row]][col_np[col]])
                pre_data.append(label_index[0][row_np[row]][col_np[col]])
                pre_data_lis.append(pre_data)
                # print(pre_data)
                
                # break
            # break
        # print(out.shape)
        del out
        torch.cuda.empty_cache()
        print("batch:",i," predata length:",len(pre_data_lis))

        # acc
        # label_index = np.argmax(semanticBceLabel,axis=1)
        # predict_index = np.argmax(out.cpu().detach().numpy(),axis=1)
        # pre_acc_result = np.equal(predict_index, label_index)
        # correct+=pre_acc_result.sum()
        # total+=65536
        # break
    print(len(pre_data_lis))
    with open("geo_res.txt",'a') as f:
        for i in range(len(pre_data_lis)):
            for j in range(len(pre_data_lis[i])):
                f.write(str(pre_data_lis[i][j]))
                f.write(' ')
            f.write('\n')
    # print("correct:",correct)
    # print("total:",total)

   
