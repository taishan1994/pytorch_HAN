# pytorch_HAN
一位热心git友指出之前问题是过拟合了，修改以下utils.py相关地方得到以下结果：

![1.image](https://github.com/taishan1994/pytorch_HAN/blob/master/images/1.png)
![2.image](https://github.com/taishan1994/pytorch_HAN/blob/master/images/2.png)

不过，别人都是在train训练，在test测试，咋回事呢？

****

Paper address:<br>
<a href="https://github.com/Jhy1993/Representation-Learning-on-Heterogeneous-Graph">https://github.com/Jhy1993/Representation-Learning-on-Heterogeneous-Graph</a><br>
Heterogeneous Graph Attention Network (HAN) with pytorch. If you want to pursue the performance in the original paper, 
this may not be suitable for you, because there is still a problem: training loss decreases, but verification loss increases.<br>

If you just want to figure out the basic principles of HAN and how to change tensorflow code to pytorch code, then this is for you.
I implemented it according to the original tensorflow code structure.<br>

If you want to pursue higher performance, please refer to:<br>
Official tensorflow implementation:<a href="https://github.com/Jhy1993/HAN">https://github.com/Jhy1993/HAN</a><br>
DGL implementation:<a href="https://github.com/dmlc/dgl/tree/master/examples/pytorch/han">https://github.com/dmlc/dgl/tree/master/examples/pytorch/han</a><br>

# The result
Address of ACM data set：
Preprocessed ACM can be found in: <a href="https://pan.baidu.com/s/1V2iOikRqHPtVvaANdkzROw">https://pan.baidu.com/s/1V2iOikRqHPtVvaANdkzROw</a> 提取码：50k2<br>

You can use the following command to run:
```python
python main.py
```

Training result:
```python
600 300 2125
y_train:(3025, 3), y_val:(3025, 3), y_test:(3025, 3), train_idx:(1, 600), val_idx:(1, 300), test_idx:(1, 2125)
2
model: pre_trained/acm/acm_allMP_multi_fea_.ckpt
fea_list[0].shape torch.Size([1, 1870, 3025])
biases_list[0].shape: torch.Size([1, 3025, 3025])
3
2
torch.Size([1, 1870, 3025]) torch.Size([1, 3025, 3025])
torch.Size([1, 1870, 3025]) torch.Size([1, 3025, 3025])
训练节点个数： 600
验证节点个数： 300
测试节点个数： 2125
epoch:001, loss:1.1004, TrainAcc:0.3517, ValLoss:1.1022, ValAcc:0.4000
epoch:002, loss:1.0762, TrainAcc:0.4250, ValLoss:1.1980, ValAcc:0.0533
epoch:003, loss:1.0007, TrainAcc:0.6300, ValLoss:1.4572, ValAcc:0.0533
epoch:004, loss:0.8876, TrainAcc:0.6583, ValLoss:2.0040, ValAcc:0.0500
epoch:005, loss:0.8145, TrainAcc:0.6350, ValLoss:2.7091, ValAcc:0.0500
epoch:006, loss:0.7897, TrainAcc:0.6267, ValLoss:3.2186, ValAcc:0.0500
epoch:007, loss:0.7804, TrainAcc:0.6150, ValLoss:3.4550, ValAcc:0.0500
epoch:008, loss:0.7527, TrainAcc:0.6150, ValLoss:3.5096, ValAcc:0.0500
epoch:009, loss:0.7404, TrainAcc:0.6117, ValLoss:3.5125, ValAcc:0.0600
epoch:010, loss:0.7329, TrainAcc:0.6633, ValLoss:3.5349, ValAcc:0.0400
epoch:011, loss:0.7169, TrainAcc:0.6983, ValLoss:3.5743, ValAcc:0.0133
epoch:012, loss:0.6934, TrainAcc:0.6917, ValLoss:3.6612, ValAcc:0.0033
epoch:013, loss:0.6711, TrainAcc:0.6750, ValLoss:3.7738, ValAcc:0.0033
epoch:014, loss:0.6645, TrainAcc:0.6733, ValLoss:3.9418, ValAcc:0.0200
epoch:015, loss:0.6652, TrainAcc:0.6833, ValLoss:4.0934, ValAcc:0.0300
epoch:016, loss:0.6515, TrainAcc:0.6883, ValLoss:4.2498, ValAcc:0.0300
epoch:017, loss:0.6238, TrainAcc:0.7050, ValLoss:4.4304, ValAcc:0.0300
epoch:018, loss:0.6082, TrainAcc:0.7317, ValLoss:4.5820, ValAcc:0.0333
epoch:019, loss:0.6030, TrainAcc:0.7517, ValLoss:4.7110, ValAcc:0.0367
epoch:020, loss:0.5933, TrainAcc:0.7850, ValLoss:4.8053, ValAcc:0.0400
epoch:021, loss:0.5824, TrainAcc:0.8267, ValLoss:4.8781, ValAcc:0.0333
epoch:022, loss:0.5655, TrainAcc:0.8017, ValLoss:4.9006, ValAcc:0.0267
epoch:023, loss:0.5333, TrainAcc:0.8083, ValLoss:4.9148, ValAcc:0.0167
epoch:024, loss:0.5175, TrainAcc:0.8050, ValLoss:4.8788, ValAcc:0.0100
epoch:025, loss:0.4994, TrainAcc:0.8117, ValLoss:4.7670, ValAcc:0.0033
epoch:026, loss:0.4888, TrainAcc:0.8333, ValLoss:4.5965, ValAcc:0.0033
```
This is where the problem lies.<br>
If you know how to solve this problem, please don't hesitate to tell me.





