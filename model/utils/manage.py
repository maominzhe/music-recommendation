from collections import namedtuple, OrderedDict
from itertools import product
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
import time
import pandas as pd 
import json


class RunBuilder():
    """Management of hyper-parameters, the pre-set hyper-parameters can 
    be automatically combined in the training process training"""
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())

        runs = []

        for ele in product(*params.values()):
            runs.append(Run(*ele))
        
        return runs
    
class RunManager():
    """Runtime data management class"""
    def __init__(self):
        #train set
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_correct_num = 0
        self.epoch_start_time = None

        #test set
        self.test_epoch_count = 0
        self.test_epoch_loss = 0
        self.test_epoch_correct_num = 0

        #hyper params, iter num 
        self.run_params = None
        self.run_count = 0
        self.run_data =[]
        self.run_start_time = None

        self.Network = None
        self.loader = None

        #tensorboard
        self.tb = None

    def begin_run(self, run, network, loader, test_loader):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1
        self.network = network
        self.loader = loader
        self.test_loader = test_loader
        self.tb = SummaryWriter(comment=f'-{run}')

        signal, sr, address = next(iter(self.loader))

        self.tb.add_graph(
            self.network,
            signal.to(run.device)
        )

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0
        self.test_epoch_count = 0

        
    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_correct_num = 0
        self.test_epoch_count += 1
        self.test_epoch_loss = 0
        self.test_epoch_correct_num = 0


    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        # 训练集损失值
        loss = self.epoch_loss / len(self.loader.dataset)
        # 测试集准确率
        accuracy = self.epoch_correct_num / len(self.loader.dataset)
        print(f'正确率：{self.epoch_correct_num} / {len(self.loader.dataset)}')
        
        # 测试集 test
        # print(f"{self.test_epoch_correct_num}+{len(self.test_loader.dataset)}")
        test_loss = self.test_epoch_loss / len(self.test_loader.dataset)
        test_accuracy = self.test_epoch_correct_num / len(self.test_loader.dataset)
        
        # 加入损失函数图像
        self.tb.add_scalars('Loss', {"train_loss": loss, 
                                    "test_loss": test_loss}, self.epoch_count)
        # 加入准确度函数图像
        self.tb.add_scalars('Accuracy', {"train_accuracy": accuracy, 
                                        "test_accuracy": test_accuracy}, self.epoch_count)
        
        # self.tb.add_scalar('Test_Loss', test_loss, self.epoch_count)
        
        #self.tb.add_scalar('Test_Accuracy', test_accuracy, self.epoch_count)
        
        for name, param in self.network.named_parameters():
            # 神经网络每一层的值
            self.tb.add_histogram(name, param, self.epoch_count)
            # 每一层值所对应的梯度
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()

        results['run'] = self.run_count
        results['epoch'] = self.epoch_count
        results['loss'] = loss
        results['accuracy'] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration

        for k, v in self.run_params._asdict().items():
            results[k] = v

        self.run_data.append(results)

        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        clear_output(wait = True)
        display(df)
    # 作业核心数
    def get_num_workers(self,num_workers):
        self.epoch_num_workers = num_workers

    # 记录每一epoch的损失 训练集    
    def track_loss(self,loss,batch):
        self.epoch_loss += loss.item()*batch[0].shape[0]
    
    # 测试集
    def test_loss(self,test_loss, test_batch):
         self.test_epoch_loss += test_loss.item()*test_batch[0].shape[0]
    
    # 记录每一epoch上测试正确的数量 测试集
    def test_num_correct(self, test_preds, test_labels):
        
        self.test_epoch_correct_num += self.get_correct_num(test_preds, test_labels)
        
    # 训练集
    def track_num_correct(self, preds, labels):
        self.epoch_correct_num += self.get_correct_num(preds, labels)
    
    def get_correct_num(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
    
    # 训练数据保存CSV文件中
    def save(self, fileName):
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'{fileName}.csv')
        
        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)


class TestRunManager():
    """Runtime data management class"""
    def __init__(self):
        #train set
        # self.epoch_count = 0
        # self.epoch_loss = 0
        # self.epoch_correct_num = 0
        # self.epoch_start_time = None

        #test set
        self.test_epoch_count = 0
        self.test_epoch_loss = 0
        self.test_epoch_correct_num = 0

        #hyper params, iter num 
        self.run_params = None
        self.run_count = 0
        self.run_data =[]
        self.run_start_time = None

        self.Network = None
        self.test_loader = None

        #tensorboard
        self.tb = None

    def begin_run(self, run, network, test_loader):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1
        self.network = network
        #self.loader = loader
        self.test_loader = test_loader
        self.tb = SummaryWriter(comment=f'-{run}')

        signal, sr, address = next(iter(self.test_loader))

        self.tb.add_graph(
            self.network,
            signal.to(run.device)
        )

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0
        self.test_epoch_count = 0

        
    def begin_epoch(self):
        self.epoch_start_time = time.time()
        # self.epoch_count += 1
        # self.epoch_loss = 0
        # self.epoch_correct_num = 0
        self.test_epoch_count += 1
        self.test_epoch_loss = 0
        self.test_epoch_correct_num = 0


    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        # 训练集损失值
        #loss = self.epoch_loss / len(self.loader.dataset)
        # 测试集准确率
        #accuracy = self.epoch_correct_num / len(self.loader.dataset)
        #print(f'正确率：{self.epoch_correct_num} / {len(self.loader.dataset)}')
        
        # 测试集 test
        # print(f"{self.test_epoch_correct_num}+{len(self.test_loader.dataset)}")
        test_loss = self.test_epoch_loss / len(self.test_loader.dataset)
        test_accuracy = self.test_epoch_correct_num / len(self.test_loader.dataset)
        
        # 加入损失函数图像
        # self.tb.add_scalars('Loss', {"train_loss": loss, 
        #                             "test_loss": test_loss}, self.epoch_count)
        # # 加入准确度函数图像
        # self.tb.add_scalars('Accuracy', {"train_accuracy": accuracy, 
        #                                 "test_accuracy": test_accuracy}, self.epoch_count)
        
        # self.tb.add_scalar('Test_Loss', test_loss, self.epoch_count)
        
        #self.tb.add_scalar('Test_Accuracy', test_accuracy, self.epoch_count)
        
        for name, param in self.network.named_parameters():
            # 神经网络每一层的值
            self.tb.add_histogram(name, param, self.epoch_count)
            # 每一层值所对应的梯度
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()

        results['run'] = self.run_count
        results['epoch'] = self.epoch_count
        results['loss'] = test_loss
        results['accuracy'] = test_accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration

        for k, v in self.run_params._asdict().items():
            results[k] = v

        self.run_data.append(results)

        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        clear_output(wait = True)
        display(df)
    # 作业核心数
    def get_num_workers(self,num_workers):
        self.epoch_num_workers = num_workers

    # 记录每一epoch的损失 训练集    
    def track_loss(self,loss,batch):
        self.epoch_loss += loss.item()*batch[0].shape[0]
    
    # 测试集
    def test_loss(self,test_loss, test_batch):
         self.test_epoch_loss += test_loss.item()*test_batch[0].shape[0]
    
    # 记录每一epoch上测试正确的数量 测试集
    def test_num_correct(self, test_preds, test_labels):
        
        self.test_epoch_correct_num += self.get_correct_num(test_preds, test_labels)
        
    # 训练集
    def track_num_correct(self, preds, labels):
        self.epoch_correct_num += self.get_correct_num(preds, labels)
    
    def get_correct_num(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
    
    # 训练数据保存CSV文件中
    def save(self, fileName):
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'{fileName}.csv')
        
        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)