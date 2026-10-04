import os

import torch

from models import Autoformer, BiGRU, BP, CNN_GRU, DLinear, FEDformer, GRU, Informer
from models import LSTM, MDDS_Mixer, PatchTST, TCN, TimesNet, iTransformer


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "MDDS-Mixer": MDDS_Mixer,
            "PatchTST": PatchTST,
            "DLinear": DLinear,
            "FEDformer": FEDformer,
            "Autoformer": Autoformer,
            "Informer": Informer,
            "TimesNet": TimesNet,
            "iTransformer": iTransformer,
            "CNN_GRU": CNN_GRU,
            "TCN": TCN,
            "GRU": GRU,
            "LSTM": LSTM,
            "BP": BP,
            "BiGRU": BiGRU,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == "cuda" and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _get_data(self, flag):
        raise NotImplementedError

    def vali(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
