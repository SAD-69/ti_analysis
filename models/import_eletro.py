from db.postgres import Postgres
from models import os


class Eletro(Postgres):
    def __init__(self, engine = ..., fgb_folder: str = 'fgb'):
        super().__init__(engine)
        self.fgb_folder_path = fgb_folder
    
    @property
    def fgb_folder(self):
        if not os.path.exists(self.fgb_folder_path):
            os.mkdir(self.fgb_folder)
        return self.fgb_folder_path
    
    @property
    def lista_gdb(self):
        return os.listdir(self.fgb_folder)
    
    @property
    def layer_list(self):
        return ['SUB', 'SSDMT', 'SSDAT', 'SSDBT']