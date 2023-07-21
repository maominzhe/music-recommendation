from collections import namedtuple, OrderedDict
from itertools import product
import time
from torch.utils.tensorboard import SummaryWriter


class RunBuilder():
    """Management of hyper-parameters, the pre-set hyper-parameters can 
    be automatically combined in the training process training"""
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())

        runs = []

        for ele in product(*params.value()):
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