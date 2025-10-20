import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from models.alignn import ALIGNN_PyG
from models.cgcnn import CGCNN_PyG
from models.crabnet import CrabNet
import pytorch_lightning as L
from datamodules.gnn_datamodule import GNNDataModule
from datamodules.crabnet_datamodule import CrabNetDataModule
from utils.utils import count_parameters, RobustL2Loss, QuantileLoss, RobustL1Loss, StudentTLoss, IntervalScoreLoss
from utils.crabnet_utils import Scaler, DummyScaler, Lamb, Lookahead
from torch.nn import HuberLoss, CrossEntropyLoss, MSELoss, L1Loss, BCEWithLogitsLoss
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os


class GNNModel(L.LightningModule):
    def __init__(self, **config):
        super().__init__()
        self.save_hyperparameters()
        L.seed_everything(config['data']['random_seed'])
        self.batch_size = config['data']['batch_size']
        self.learning_rate = config['optim']['learning_rate']
        self.momentum=config['optim']['momentum']
        self.decay = config['optim']['weight_decay']
    
        # Defining the model
        self.model_name = config['model']['name']   
        
        if(self.model_name == 'alignn'):
            self.model=ALIGNN_PyG(**config['model'])
            print(f'Model name: {self.model_name}\n')
            print(f'Model size: {count_parameters(self.model)} parameters\n')
        elif(self.model_name == 'cgcnn'):
            config['data']['lmdb_exist'] = True
            data = GNNDataModule(**config['data'])
            dataset = data.train_dataset
            g = dataset.__getitem__(0)
            orig_atom_fea_len = g.x.shape[-1]
            config['model']['orig_atom_fea_len'] = orig_atom_fea_len

            if hasattr(g, "additional_compound_features"):
                add_feat_len = g.additional_compound_features.shape[-1]
                config['model']['additional_compound_features'] = True
                config['model']['add_feat_len'] = add_feat_len
            else:
                config['model']['additional_compound_features'] = False
                config['model']['add_feat_len'] = None
            
            self.model=CGCNN_PyG(**config['model'])
            print(f'Model name: {self.model_name}\n')
            print(f'Model size: {count_parameters(self.model)} parameters\n')
        
        print(self.model)
        self.save_hyperparameters(config)
        # Defining the loss function 
        self.classification = config['model']['classification']
        self.robust_regression = config['model']['robust_regression']
        self.quantile_regression = config['model']['quantile_regression']
        self.loss_name = config['loss']['name'] 


        if self.classification:
            self.weights = config['optim']['weights']
            self.num_classes = config['model']['num_classes']
            print(f'Using CrossEntropyLoss loss for classification task, {self.num_classes} classes')
            if self.weights:
                df=pd.read_csv(os.path.join(config['data']['root_dir'],config['data']['id_prop_csv']))
                labels=df['1'].values
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
                class_weights = torch.tensor(class_weights, dtype=torch.float)
                self.criterion = CrossEntropyLoss(weight=class_weights)
            else:
                self.criterion = CrossEntropyLoss() # BCEWithLogitsLoss takes in logits with size (batch_size,num_classes)
            
        elif self.robust_regression:
            if self.loss_name == 'RobustL1Loss':
                print('Using RobustL1Loss for regression task')
                self.criterion = RobustL1Loss
            elif self.loss_name == 'RobustL2Loss':
                print('Using RobustL2Loss for regression task')
                self.criterion = RobustL2Loss
            elif self.loss_name == 'StudentTLoss':
                print('Using StudentTLoss for regression task')
                self.student_nu = config['loss']['student_nu']
                self.criterion = lambda output, logstd, target: StudentTLoss(output, logstd, target, nu=self.student_nu)

        elif self.quantile_regression:
            quantile = config['loss']['quantile']
            if self.loss_name == 'QuantileLoss':
                print('Using QuantileLoss with q={quantile} for regression task')
                self.criterion = lambda output, target: QuantileLoss(output, target, quantile=quantile)
            elif self.loss_name == 'IntervalScoreLoss':
                print('Using IntervalScoreLoss with q={quantile} for regression task')
                self.criterion = lambda y_low, y_high, target: IntervalScoreLoss(y_low, y_high, target, quantile=quantile)
            
        else:
            if self.loss_name == 'HuberLoss':
                print('Using HuberLoss for regression task')
                self.criterion = HuberLoss()
            elif self.loss_name == 'MSELoss':
                print('Using MSELoss for regression task')
                self.criterion = MSELoss()
            elif self.loss_name == 'L1Loss':
                print('Using L1Loss for regression task')
                self.criterion = L1Loss()


    def forward(self, graphs):
        if(self.model_name == 'alignn'):
            g,lg = graphs
            return self.model(g,lg)
        elif(self.model_name == 'cgcnn'):
            g = graphs
            return self.model(g)
    
    def configure_optimizers(self):
        if(self.model_name == 'alignn'):
            optimizer = optim.AdamW(self.model.parameters(), self.learning_rate,
                                  weight_decay=self.decay)
            return [optimizer]
        elif(self.model_name == 'cgcnn'):
            optimizer = optim.AdamW(self.model.parameters(), self.learning_rate,
                              weight_decay=self.decay)
            # lr_scheduler=StepLR(optimizer,
            #                     step_size=1,
            #                     gamma=0.5)
            
            # return [optimizer], [lr_scheduler]
            return [optimizer]
    
    def training_step(self, batch, batch_idx):
        output=self(batch)
        if(self.model_name == 'alignn'):
            g, _ = batch
            target = g.y
        elif(self.model_name == 'cgcnn'):
            target = batch.y
            
        if self.robust_regression:
            prediction, log_std = output.chunk(2, dim=-1)
            prediction=prediction.squeeze()
            log_std=log_std.squeeze()
            loss = self.criterion(prediction, log_std, target)
        elif self.classification:
            loss = self.criterion(output, target.long())
        elif self.quantile_regression and self.loss_name == 'IntervalScoreLoss':
            y_low, y_high = output.chunk(2, dim=-1)
            loss = self.criterion(y_low, y_high, target)
        else:
            output = output.view(-1)
            loss = self.criterion(output, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        
        if self.classification:
            prediction = torch.argmax(output, dim=1)
            # probs = F.softmax(output, dim=1)[:, 1]
            acc = accuracy_score(target.cpu().numpy(), prediction.detach().cpu().numpy())
            # auc = roc_auc_score(target.cpu().numpy(), probs.detach().cpu().numpy(),multi_class='ovr',average='macro')
            f1 = f1_score(target.cpu().numpy(), prediction.detach().cpu().numpy(), average='macro')
            mcc = matthews_corrcoef(target.cpu().numpy(), prediction.detach().cpu().numpy())
            self.log("train_acc", float(acc), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            # self.log("train_auc", float(auc), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("train_f1", float(f1), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("train_mcc", float(mcc), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        elif self.robust_regression:
            mse = mean_squared_error(target.cpu(),prediction.data.cpu())
            mae = mean_absolute_error(target.cpu(),prediction.data.cpu())
            self.log("train_mse", float(mse), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("train_mae", float(mae), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        elif self.quantile_regression and self.loss_name == 'IntervalScoreLoss':
            mse = mean_squared_error(target.cpu(),0.5*y_high.data.cpu()+0.5*y_low.data.cpu())
            mae = mean_absolute_error(target.cpu(),0.5*y_high.data.cpu()+0.5*y_low.data.cpu())
            self.log("train_mse", float(mse), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("train_mae", float(mae), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        else:
            mse = mean_squared_error(target.cpu(),output.data.cpu())
            mae = mean_absolute_error(target.cpu(),output.data.cpu())
            self.log("train_mse", float(mse), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("train_mae", float(mae), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        output=self(batch)
        if(self.model_name == 'alignn'):
            g, _ = batch
            target = g.y
        elif(self.model_name == 'cgcnn'):
            target = batch.y

        if self.robust_regression:
            prediction, log_std = output.chunk(2, dim=-1)
            prediction=prediction.squeeze()
            log_std=log_std.squeeze()
            loss = self.criterion(prediction, log_std, target)
        elif self.classification:
            loss = self.criterion(output, target.long())
        elif self.quantile_regression and self.loss_name == 'IntervalScoreLoss':
            y_low, y_high = output.chunk(2, dim=-1)
            loss = self.criterion(y_low, y_high, target)
        else:
            output = output.view(-1)
            loss = self.criterion(output, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
 
        if self.classification:
            prediction = torch.argmax(output, dim=1)
            # probs = F.softmax(output, dim=1)[:, 1]
            acc = accuracy_score(target.cpu().numpy(), prediction.detach().cpu().numpy())
            # auc = roc_auc_score(target.cpu().numpy(), probs.detach().cpu().numpy(),multi_class='ovr',average='macro')
            f1 = f1_score(target.cpu().numpy(), prediction.detach().cpu().numpy(),average='macro')
            mcc = matthews_corrcoef(target.cpu().numpy(), prediction.detach().cpu().numpy())
            self.log("val_acc", float(acc), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            # self.log("val_auc", float(auc), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val_f1", float(f1), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val_mcc", float(mcc), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        elif self.robust_regression:
            mse = mean_squared_error(target.cpu(),prediction.data.cpu())
            mae = mean_absolute_error(target.cpu(),prediction.data.cpu())
            self.log("val_mse", float(mse), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("val_mae", float(mae), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        elif self.quantile_regression and self.loss_name == 'IntervalScoreLoss':
            mse = mean_squared_error(target.cpu(),0.5*y_high.data.cpu()+0.5*y_low.data.cpu())
            mae = mean_absolute_error(target.cpu(),0.5*y_high.data.cpu()+0.5*y_low.data.cpu())
            self.log("val_mse", float(mse), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("val_mae", float(mae), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        else:
            mse = mean_squared_error(target.cpu(),output.data.cpu())
            mae = mean_absolute_error(target.cpu(),output.data.cpu())
            self.log("val_mse", float(mse), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("val_mae", float(mae), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss
    
    def test_step(self, batch, batch_idx):
        output=self(batch)
        if(self.model_name == 'alignn'):
            g, _ = batch
            target = g.y
        elif(self.model_name == 'cgcnn'):
            target = batch.y
        
        if self.robust_regression:
            prediction, log_std = output.chunk(2, dim=-1)
            prediction=prediction.squeeze()
            log_std=log_std.squeeze()
            loss = self.criterion(prediction, log_std, target)
        elif self.classification:
            loss = self.criterion(output, target.long())
        elif self.quantile_regression and self.loss_name == 'IntervalScoreLoss':
            y_low, y_high = output.chunk(2, dim=-1)
            loss = self.criterion(y_low, y_high, target)
        else:
            output = output.view(-1)
            loss = self.criterion(output, target)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        
        if self.classification:
            prediction = torch.argmax(output, dim=1)
            # probs = F.softmax(output, dim=1)[:, 1]
            acc = accuracy_score(target.cpu().numpy(), prediction.detach().cpu().numpy())
            # auc = roc_auc_score(target.cpu().numpy(), probs.detach().cpu().numpy(),multi_class='ovr',average='macro')
            f1 = f1_score(target.cpu().numpy(), prediction.detach().cpu().numpy(),average='macro')
            mcc = matthews_corrcoef(target.cpu().numpy(), prediction.detach().cpu().numpy())
            self.log("test_acc", float(acc), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            # self.log("test_auc", float(auc), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("test_f1", float(f1), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("test_mcc", float(mcc), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        elif self.robust_regression:
            mse = mean_squared_error(target.cpu(),prediction.data.cpu())
            mae = mean_absolute_error(target.cpu(),prediction.data.cpu())
            self.log("test_mse", float(mse), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("test_mae", float(mae), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        elif self.quantile_regression and self.loss_name == 'IntervalScoreLoss':
            mse = mean_squared_error(target.cpu(),0.5*y_high.data.cpu()+0.5*y_low.data.cpu())
            mae = mean_absolute_error(target.cpu(),0.5*y_high.data.cpu()+0.5*y_low.data.cpu())
            self.log("val_mse", float(mse), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("val_mae", float(mae), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        else:    
            mse = mean_squared_error(target.cpu(),output.data.cpu())
            mae = mean_absolute_error(target.cpu(),output.data.cpu())
            self.log("test_mse", float(mse), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("test_mae", float(mae), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output=self(batch)
        if(self.model_name == 'alignn'):
            g, _ = batch
            target = g.y
            idx=g.sample_id
        elif(self.model_name == 'cgcnn'):
            target = batch.y
            idx=batch.sample_id
        
        if self.classification:
            prediction = torch.argmax(output, dim=1)
            probs = F.softmax(output, dim=1)
            return prediction.detach().cpu(), probs.detach().cpu(), target, idx
        elif self.robust_regression:
            prediction, log_std = output.chunk(2, dim=-1) 
            std = torch.exp(log_std) 
            return prediction, std, target, idx
        elif self.quantile_regression and self.loss_name == 'IntervalScoreLoss':
            y_low, y_high = output.chunk(2, dim=-1)
            return y_low, y_high, target, idx
        else:
            return output.data.cpu(), target, idx



class CrabNetLightning(L.LightningModule):
    def __init__(self, **config):
        super().__init__()
        self.save_hyperparameters()
        L.seed_everything(config['data']['random_seed'])
        self.batch_size = config['data']['batch_size']

        self.base_lr = config['optim']['base_lr']
        self.max_lr=config['optim']['max_lr']
        self.weights = config['optim']['weights']
        self.schedule = config['optim']['schedule']
        self.swa = config['optim']['swa']
        self.swa_lr = config['optim']['swa_lr']
        self.swa_start = config['optim']['swa_start']
        
        self.model_name = config['model']['name']
        self.model = CrabNet(out_dims=config['model']['out_dims'],
                             d_model=config['model']['d_model'],
                             N=config['model']['NN'],
                             heads=config['model']['heads'],
                             embedding_file=config["data"]["atomic_features"]["atom_feature_strategy"]["atom_feature_file"])
        print('\nModel architecture: out_dims, d_model, N, heads')
        print(f'{self.model.out_dims}, {self.model.d_model}, '
                  f'{self.model.N}, {self.model.heads}')
        print(f'Model size: {count_parameters(self.model)} parameters\n')

        ### here we define some important parameters
        self.fudge = config['model']['fudge']
        self.classification = config['model']['classification']
        self.robust_regression = config['model']['robust_regression']
        self.quantile_regression = config['model']['quantile_regression']
        self.loss_name = config['loss']['name'] 
        
        ### here we also need to initialise scaler based on training data
        y=pd.read_csv(os.path.join(config['data']['root_dir'],config['data']['train_path']))
        self.step_size = len(y)

        if self.classification:
            # At the moment only binary classification
            self.weights = config['optim']['weights']
            self.num_classes = config['model']['num_classes']
            # if(np.sum(y)>0):
            #     self.weight=torch.tensor(((len(y)-np.sum(y))/np.sum(y))).cuda()
            # print("Using BCE loss for classification task")
            # self.criterion = BCEWithLogitsLoss
            print(f'Using CrossEntropyLoss loss for classification task, {self.num_classes} classes')
            if self.weights:
                df=pd.read_csv(os.path.join(config['data']['root_dir'],config['data']['id_prop_csv']))
                labels=df['1'].values
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
                class_weights = torch.tensor(class_weights, dtype=torch.float)
                self.criterion = CrossEntropyLoss(weight=class_weights)
            else:
                self.criterion = CrossEntropyLoss() # BCEWithLogitsLoss takes in logits with size (batch_size,num_classes)    
        elif self.robust_regression:
            # self.scaler = Scaler(y)
            if self.loss_name == 'RobustL1Loss':
                print('Using RobustL1Loss for regression task')
                self.criterion = RobustL1Loss
            elif self.loss_name == 'RobustL2Loss':
                print('Using RobustL2Loss for regression task')
                self.criterion = RobustL2Loss
            elif self.loss_name == 'StudentTLoss':
                print('Using StudentTLoss for regression task')
                self.student_nu = config['loss']['student_nu']
                self.criterion = lambda output, logstd, target: StudentTLoss(output, logstd, target, nu=self.student_nu)
        elif self.quantile_regression:
            quantile = config['loss']['quantile']
            if self.loss_name == 'QuantileLoss':
                print('Using QuantileLoss with q={quantile} for regression task')
                self.criterion = lambda output, target: QuantileLoss(output, target, quantile=quantile)
            elif self.loss_name == 'IntervalScoreLoss':
                print('Using IntervalScoreLoss with q={quantile} for regression task')
                self.criterion = lambda y_low, y_high, target: IntervalScoreLoss(y_low, y_high, target, quantile=quantile)
        else:
            if self.loss_name == 'HuberLoss':
                print('Using HuberLoss for regression task')
                self.criterion = HuberLoss()
            elif self.loss_name == 'MSELoss':
                print('Using MSELoss for regression task')
                self.criterion = MSELoss()
            elif self.loss_name == 'L1Loss':
                print('Using L1Loss for regression task')
                self.criterion = L1Loss()


    def forward(self, src, frac):
        out=self.model(src, frac)
        return out

    def configure_optimizers(self):
        base_optim = Lamb(params=self.model.parameters(),lr=0.001)
        optimizer = Lookahead(base_optimizer=base_optim)
        lr_scheduler = CyclicLR(optimizer,
                                base_lr=self.base_lr,
                                max_lr=self.max_lr,
                                cycle_momentum=False,
                                step_size_up=self.step_size)
        # lr_scheduler=StepLR(optimizer,
        #                     step_size=3,
        #                     gamma=0.5)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        X, y, formula = batch
        # y = self.scaler.scale(y)
        src, frac = X.squeeze(-1).chunk(2, dim=1)
        frac = frac * (1 + (torch.randn_like(frac))*self.fudge)
        frac = torch.clamp(frac, 0, 1)
        frac[src == 0] = 0
        frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
        output = self(src, frac)

        if self.robust_regression:
            prediction, log_std = output.chunk(2, dim=-1)
            loss = self.criterion(prediction.view(-1), log_std.view(-1), y.view(-1))
        elif self.classification:
            loss = self.criterion(output, y.long())
        elif self.quantile_regression and self.loss_name == 'IntervalScoreLoss':
            y_low, y_high = output.chunk(2, dim=-1)
            loss = self.criterion(y_low, y_high, y.view(-1))
        else:
            output = output.view(-1)
            loss = self.criterion(output, y.view(-1))
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        
        if self.classification:
            prediction = torch.argmax(output, dim=1)
            # probs = F.softmax(output, dim=1)[:, 1]
            acc = accuracy_score(y.cpu().numpy(), prediction.detach().cpu().numpy())
            # auc = roc_auc_score(target.cpu().numpy(), probs.detach().cpu().numpy(),multi_class='ovr',average='macro')
            f1 = f1_score(y.cpu().numpy(), prediction.detach().cpu().numpy(), average='macro')
            mcc = matthews_corrcoef(y.cpu().numpy(), prediction.detach().cpu().numpy())
            self.log("train_acc", float(acc), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            # self.log("train_auc", float(auc), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("train_f1", float(f1), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("train_mcc", float(mcc), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        elif self.robust_regression:
            # log_std = torch.exp(log_std) * self.scaler.std
            # prediction = self.scaler.unscale(prediction)
            mse = mean_squared_error(y.cpu(),prediction.data.cpu())
            mae = mean_absolute_error(y.cpu(),prediction.data.cpu())
            self.log("train_mse", float(mse), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("train_mae", float(mae), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        elif self.quantile_regression and self.loss_name == 'IntervalScoreLoss':
            mse = mean_squared_error(y.cpu(),0.5*y_high.data.cpu()+0.5*y_low.data.cpu())
            mae = mean_absolute_error(y.cpu(),0.5*y_high.data.cpu()+0.5*y_low.data.cpu())
            self.log("train_mse", float(mse), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("train_mae", float(mae), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        else:
            mse = mean_squared_error(y.cpu(),output.data.cpu())
            mae = mean_absolute_error(y.cpu(),output.data.cpu())
            self.log("train_mse", float(mse), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("train_mae", float(mae), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
    
        return loss

    def validation_step(self, batch, batch_idx):
        X, y, formula = batch
        # y = self.scaler.scale(y)
        src, frac = X.squeeze(-1).chunk(2, dim=1)
        frac = frac * (1 + (torch.randn_like(frac))*self.fudge)
        frac = torch.clamp(frac, 0, 1)
        frac[src == 0] = 0
        frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
        output = self(src, frac)

        if self.robust_regression:
            prediction, log_std = output.chunk(2, dim=-1)
            loss = self.criterion(prediction.view(-1), log_std.view(-1), y.view(-1))
        elif self.classification:
            loss = self.criterion(output, y.long())
        elif self.quantile_regression and self.loss_name == 'IntervalScoreLoss':
            y_low, y_high = output.chunk(2, dim=-1)
            loss = self.criterion(y_low, y_high, y.view(-1))
        else:
            output = output.view(-1)
            loss = self.criterion(output, y.view(-1))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        
        if self.classification:
            prediction = torch.argmax(output, dim=1)
            # probs = F.softmax(output, dim=1)[:, 1]
            acc = accuracy_score(y.cpu().numpy(), prediction.detach().cpu().numpy())
            # auc = roc_auc_score(target.cpu().numpy(), probs.detach().cpu().numpy(),multi_class='ovr',average='macro')
            f1 = f1_score(y.cpu().numpy(), prediction.detach().cpu().numpy(), average='macro')
            mcc = matthews_corrcoef(y.cpu().numpy(), prediction.detach().cpu().numpy())
            self.log("val_acc", float(acc), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            # self.log("train_auc", float(auc), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val_f1", float(f1), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("val_mcc", float(mcc), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        elif self.robust_regression:
            # log_std = torch.exp(log_std) * self.scaler.std
            # prediction = self.scaler.unscale(prediction)
            mse = mean_squared_error(y.cpu(),prediction.data.cpu())
            mae = mean_absolute_error(y.cpu(),prediction.data.cpu())
            self.log("val_mse", float(mse), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("val_mae", float(mae), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        elif self.quantile_regression and self.loss_name == 'IntervalScoreLoss':
            mse = mean_squared_error(y.cpu(),0.5*y_high.data.cpu()+0.5*y_low.data.cpu())
            mae = mean_absolute_error(y.cpu(),0.5*y_high.data.cpu()+0.5*y_low.data.cpu())
            self.log("val_mse", float(mse), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("val_mae", float(mae), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        else:
            mse = mean_squared_error(y.cpu(),output.data.cpu())
            mae = mean_absolute_error(y.cpu(),output.data.cpu())
            self.log("val_mse", float(mse), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("val_mae", float(mae), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss
     
    def test_step(self, batch, batch_idx):
        X, y, formula = batch
        # y = self.scaler.scale(y)
        src, frac = X.squeeze(-1).chunk(2, dim=1)
        frac = frac * (1 + (torch.randn_like(frac))*self.fudge)
        frac = torch.clamp(frac, 0, 1)
        frac[src == 0] = 0
        frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
        output = self(src, frac)

        if self.robust_regression:
            prediction, log_std = output.chunk(2, dim=-1)
            loss = self.criterion(prediction.view(-1), log_std.view(-1), y.view(-1))
        elif self.classification:
            loss = self.criterion(output, y.long())
        elif self.quantile_regression and self.loss_name == 'IntervalScoreLoss':
            y_low, y_high = output.chunk(2, dim=-1)
            loss = self.criterion(y_low, y_high, y.view(-1))
        else:
            output = output.view(-1)
            loss = self.criterion(output, y.view(-1))
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        
        if self.classification:
            prediction = torch.argmax(output, dim=1)
            # probs = F.softmax(output, dim=1)[:, 1]
            acc = accuracy_score(y.cpu().numpy(), prediction.detach().cpu().numpy())
            # auc = roc_auc_score(target.cpu().numpy(), probs.detach().cpu().numpy(),multi_class='ovr',average='macro')
            f1 = f1_score(y.cpu().numpy(), prediction.detach().cpu().numpy(), average='macro')
            mcc = matthews_corrcoef(y.cpu().numpy(), prediction.detach().cpu().numpy())
            self.log("test_acc", float(acc), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            # self.log("train_auc", float(auc), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("test_f1", float(f1), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("test_mcc", float(mcc), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        elif self.robust_regression:
            # log_std = torch.exp(log_std) * self.scaler.std
            # prediction = self.scaler.unscale(prediction)
            mse = mean_squared_error(y.cpu(),prediction.data.cpu())
            mae = mean_absolute_error(y.cpu(),prediction.data.cpu())
            self.log("test_mse", float(mse), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("test_mae", float(mae), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        elif self.quantile_regression and self.loss_name == 'IntervalScoreLoss':
            mse = mean_squared_error(y.cpu(),0.5*y_high.data.cpu()+0.5*y_low.data.cpu())
            mae = mean_absolute_error(y.cpu(),0.5*y_high.data.cpu()+0.5*y_low.data.cpu())
            self.log("test_mse", float(mse), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("test_mae", float(mae), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        else:
            mse = mean_squared_error(y.cpu(),output.data.cpu())
            mae = mean_absolute_error(y.cpu(),output.data.cpu())
            self.log("test_mse", float(mse), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size)
            self.log("test_mae", float(mae), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return 
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X, y, formula = batch
        # y = self.scaler.scale(y)
        src, frac = X.squeeze(-1).chunk(2, dim=1)
        frac = frac * (1 + (torch.randn_like(frac))*self.fudge)
        frac = torch.clamp(frac, 0, 1)
        frac[src == 0] = 0
        frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
        output = self(src, frac)
        
        if self.classification:
            prediction = torch.argmax(output, dim=1)
            probs = F.softmax(output, dim=1)
            return prediction.detach().cpu(), probs.detach().cpu(), y, formula
        elif self.robust_regression:
            prediction, log_std = output.chunk(2, dim=-1) 
            std = torch.exp(log_std) 
            return prediction, std, y, formula
        elif self.quantile_regression and self.loss_name == 'IntervalScoreLoss':
            y_low, y_high = output.chunk(2, dim=-1)
            return y_low, y_high, y, formula
        else:
            return output.data.cpu(), y,formula

        