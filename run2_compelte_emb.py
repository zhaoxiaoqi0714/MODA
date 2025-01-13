import sys
import os
import torch
import pyhocon
import random
import time

from datetime import datetime
from PRAD.model.utils import *
from PRAD.model.models import *
from PRAD.tools.load_data import *
from PRAD.tools.tools import *

torch.autograd.set_detect_anomaly(True)
date = time.strftime('%Y-%m-%d', time.localtime())

import yaml
import os

from PRAD.tools.tools import k_neighbor_search

## parameter
file_path = os.path.abspath(os.path.join(os.path.dirname('__file__'),os.path.pardir))
def par_yaml():
    filepath = file_path+'/运行代码/run2config.yaml'     # 文件路径,这里需要将a.yaml文件与本程序文件放在同级目录下
    with open(filepath, 'r') as f:     # 用with读取文件更好
        configs = yaml.load(f, Loader=yaml.FullLoader) # 按字典格式读取并返回
    return configs
par = par_yaml()

dataSet = par['dataSet']
save_path = file_path + '/示例数据/Run2_output/'
pro_path = save_path + 'process/'
agg_func = par['agg_func']
epochs = par['epochs'] # 30
b_sz = par['b_sz']
seed = par['seed']
cuda = par['cuda']
learn_method = par['learn_method']
unsup_loss = par['unsup_loss']
name = par['name']
num_layers = par['num_layers']
hidden_emb_size = par['hidden_emb_size']
gcn = par['gcn']
loss_mean_inital = par['loss_mean_inital']
weight = par['weight']

if torch.cuda.is_available():
	if not cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	else:
		device_id = torch.cuda.current_device()
		print('using device', device_id, torch.cuda.get_device_name(device_id))

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if cuda else "cpu")
print('DEVICE:', device)

if __name__ == '__main__':
    start_time = datetime.now()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # load data
    ds = dataSet
    dataCenter = DataCenter()
    dataCenter.load_data(ds,weight=weight, file_path=file_path)
    features = torch.FloatTensor(getattr(dataCenter, ds + '_feats')).to(device)
    labels = (getattr(dataCenter, ds + '_labels'))
    num_labels = len(set(labels))
    classification = Classification(hidden_emb_size, num_labels)
    classification.to(device)

    # model
    graphSage = GraphSage(2, features.size(1), features.size(1), features, getattr(dataCenter, ds + '_adj_lists'), device, gcn=gcn,
                          agg_func=agg_func)
    graphSage.to(device)
    unsupervised_loss = UnsupervisedLoss(getattr(dataCenter, ds + '_adj_lists'), getattr(dataCenter, ds + '_train'),device)

    if learn_method == 'sup':
        print('GraphSage with Supervised Learning')
    elif learn_method == 'plus_unsup':
        print('GraphSage with Supervised Learning plus Net Unsupervised Learning')
    else:
        print('GraphSage with Net Unsupervised Learning')

    for epoch in range(epochs):
        print('----------------------EPOCH %d-----------------------' % epoch)
        # train
        graphSage, loss_result, embed_result = apply_model(dataCenter, ds, graphSage, unsupervised_loss, b_sz, unsup_loss, device, learn_method)
        # valid
        valid_result, val_result = valid_model(dataCenter, ds, graphSage, device, learn_method)
        # test
        if valid_result <= loss_mean_inital:
            loss_mean_inital = valid_result
            test_nodes = getattr(dataCenter, ds + '_test')
            embs = graphSage(test_nodes)
        # save
        loss_final, valid_final = info_save_result(dataCenter=dataCenter, ds=ds, loss_result=loss_result, valid_result = valid_result,
                                                   save_path=pro_path, epoch=epoch)

    # save
    embed_final = emb_save_result(dataCenter=dataCenter, ds=ds, embed_result=embs, save_path=save_path, features=features, epoch=epoch,dataSet=dataSet)
    torch.save(graphSage, save_path+'model/'+date+' GraphSage.pth')
    torch.save(graphSage.state_dict(), save_path + 'model/' + date + ' GraphSage_params.pth')
    finish_time = datetime.now()
    print('-----------The Finished! The cost time:{} h-----------'.format((finish_time-start_time).seconds/3600))