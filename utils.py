import numpy as np
import scipy.io as sio

def sample_mask(idx, l):
    """Create mask."""
    #l:[3025]
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def adj_to_bias(adj, sizes, nhood=1):
    # adj -> [1,3025,3025]
    # sizes -> [3025]
    nb_graphs = adj.shape[0] # 1
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        # 对角元素上变为1
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)

def load_data_dblp(path='data/ACM3025.mat'):
    data = sio.loadmat(path)
    #truelabels：[3025,3] truefeatures:[3025,1870]
    truelabels, truefeatures = data['label'], data['feature'].astype(float)
    N = truefeatures.shape[0] #3025
    # 两种关系的邻接矩阵 [2,3025,3025]
    rownetworks = [data['PAP'] - np.eye(N), data['PLP'] - np.eye(N)]  # , data['PTP'] - np.eye(N)]

    y = truelabels
    #[1,600]
    train_idx = data['train_idx']
    #[1,300]
    val_idx = data['val_idx']
    #[1,2125]
    test_idx = data['test_idx']
    
    my_labels = np.where(y == 1)[1]
    train_my_labels_mask = sample_mask(train_idx,my_labels.shape[0])
    val_my_labels_mask = sample_mask(val_idx,my_labels.shape[0])
    test_my_labels_mask = sample_mask(test_idx,my_labels.shape[0])
    train_my_labels = my_labels[train_my_labels_mask]
    val_my_labels = my_labels[val_my_labels_mask]
    test_my_labels = my_labels[test_my_labels_mask]
    print(len(train_my_labels),len(val_my_labels),len(test_my_labels))
    my_data = {
        'train_my_labels':train_my_labels,
        'val_my_labels':val_my_labels,
        'test_my_labels':test_my_labels,
        'my_labels':my_labels
    }

    train_mask = sample_mask(train_idx, y.shape[0])
    val_mask = sample_mask(val_idx, y.shape[0])
    test_mask = sample_mask(test_idx, y.shape[0])

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    y_train[train_mask, :] = y[train_mask, :]
    y_val[val_mask, :] = y[val_mask, :]
    y_test[test_mask, :] = y[test_mask, :]

    # return selected_idx, selected_idx_2
    print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
                                                                                          y_val.shape,
                                                                                          y_test.shape,
                                                                                          train_idx.shape,
                                                                                          val_idx.shape,
                                                                                          test_idx.shape))
    truefeatures_list = [truefeatures, truefeatures, truefeatures]
    """
    rownetworks: ['PAP','PLP']
    truefeatures_list: [truefeatures, truefeatures, truefeatures]
    """
    return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask, my_data