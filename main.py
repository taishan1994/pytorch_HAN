import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = "cpu"

adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask, my_data = load_data_dblp()
nb_nodes = fea_list[0].shape[0] #节点数目 3025
ft_size = fea_list[0].shape[1] #特征的维度 1870
nb_classes = y_train.shape[1]  #标签的数目 3

fea_list = [torch.transpose(torch.from_numpy(fea[np.newaxis]),2,1).to(device) for fea in fea_list]
#fea_list = torch.from_numpy(np.array(fea_list)).to(device)
adj_list = [adj[np.newaxis] for adj in adj_list]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
#train_mask = train_mask[np.newaxis]
#val_mask = val_mask[np.newaxis]
#test_mask = test_mask[np.newaxis]

my_labels = my_data['my_labels']
train_my_labels = my_data['train_my_labels']
val_my_labels = my_data['val_my_labels']
test_my_labels = my_data['test_my_labels']


biases_list = [torch.transpose(torch.from_numpy(adj_to_bias(adj, [nb_nodes], nhood=1)),2,1).to(device) for adj in adj_list]
print(len(biases_list))

dataset = 'acm'
featype = 'fea'
checkpt_file = 'pre_trained/{}/{}_allMP_multi_{}_.ckpt'.format(dataset, dataset, featype)
print('model: {}'.format(checkpt_file))
# training params
batch_size = 1
nb_epochs = 200
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
# numbers of hidden units per each attention head in each layer
hid_units = [8]
n_heads = [8, 1]  # additional entry for the output layer
residual = False
#inputs = torch.randn(1,1870,3025)
#ret = Attn_head(inputs,8,biases_list[0])
#print(ret(inputs))
print("fea_list[0].shape",fea_list[0].shape)
print("biases_list[0].shape:",biases_list[0].shape)
print(len(fea_list))
print(len(biases_list))

for inputs,biases in zip(fea_list,biases_list):
    print(inputs.shape,biases.shape)

model = HeteGAT_multi(inputs_list=fea_list,nb_classes=nb_classes,nb_nodes=nb_nodes,attn_drop=0.5,
                      ffd_drop=0.0,bias_mat_list=biases_list,hid_units=hid_units,n_heads=n_heads,
                      activation=nn.ELU(),residual=False)
#model.forward(fea_list)
model.to(device)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(params=model.parameters(),lr=lr,betas=(0.9,0.99),weight_decay=0.0)

train_my_labels = torch.from_numpy(train_my_labels).long().to(device)
val_my_labels = torch.from_numpy(val_my_labels).long().to(device)
test_my_labels = torch.from_numpy(test_my_labels).long().to(device)

train_mask = np.where(train_mask == 1)[0]
val_mask = np.where(val_mask == 1)[0]
test_mask = np.where(test_mask == 1)[0]
train_mask = torch.from_numpy(train_mask).to(device)
val_mask = torch.from_numpy(val_mask).to(device)
test_mask = torch.from_numpy(test_mask).to(device)

def main():
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    print("训练节点个数：",len(train_my_labels))
    print("验证节点个数：",len(val_my_labels))
    print("测试节点个数：",len(test_my_labels))
    for epoch in range(1,nb_epochs):
        train_loss,train_acc = train()
        val_loss,val_acc = test("val",val_mask,val_my_labels)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        print("epoch:{:03d}, loss:{:.4f}, TrainAcc:{:.4F}, ValLoss:{:.4f}, ValAcc:{:.4f}".format(epoch,train_loss,train_acc,val_loss,val_acc))
    test_loss,test_acc = test("test",test_mask,test_my_labels)
    print("TestAcc:{:.4f}".format(test_acc))
    return train_loss_history,train_acc_history,val_loss_history,val_acc_history

def train():
    model.train()
    correct = 0
    outputs = model(fea_list)
    train_mask_outputs = torch.index_select(outputs,0,train_mask)
    _, preds = torch.max(train_mask_outputs.data,1)
    #print(preds)
    #print(train_my_labels)
    loss = criterion(train_mask_outputs,train_my_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    correct += torch.sum(preds == train_my_labels).to(torch.float32)
    acc = correct / len(train_my_labels)
    #val_loss,val_acc = test("val",val_mask,val_my_labels,epoch)
    #test_acc = test("test",test_mask,test_my_labels,epoch)
    #test_acc_history.append(test_acc)
    #print("epoch:{:03d}, loss:{:.4f}, TrainAcc:{:.4F}, ValLoss:{:.4f}, ValAcc:{:.4f}".format(epoch,loss,acc,val_loss,val_acc))
    return loss,acc

def test(mode,mask,label):
    model.eval()
    with torch.no_grad():
        correct = 0.0
        outputs = model(fea_list)
        mask_outputs = torch.index_select(outputs,0,mask)
        _, preds = torch.max(mask_outputs,1)
        loss = criterion(mask_outputs,label)
        correct += torch.sum(preds == label).to(torch.float32)
        if mode == "val":
            acc = correct / len(label)
        elif mode == "test":
            acc = correct / len(label)
        else:
            print("请输入合法的mode: val/test")
            return
        #print("[{}]>>>>>  [epoch]:{:03d}, [loss]:{:.4f}, [acc]:{:.4F}".format(mode,epoch,loss,acc))
    return loss,acc



main()
