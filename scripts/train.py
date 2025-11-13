import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, Callback
from pytorch_lightning.loggers import MLFlowLogger
import mlflow.pytorch
import argparse
from pytorch_lightning.callbacks import Callback as PLCallback

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from datamodules.gnn_datamodule import GNNDataModule
from datamodules.crabnet_datamodule import CrabNetDataModule
from utils.utils import load_yaml_config
from models.modelmodule import GNNModel, CrabNetLightning


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config_file",
                        default="alignn.yaml",
                        help="Provide the experiment configuration file")


    args = parser.parse_args(sys.argv[1:])

    path_config = Path(__file__).resolve().parent.parent / "configs" / args.config_file
    config = load_yaml_config(path_config)
    
    path_mlf_logger = Path(__file__).resolve().parent.parent / 'mlruns'
    mlf_logger = MLFlowLogger(experiment_name=config['train']['experiment_name'], \
                              run_name=config['train']['run_name'], \
                              tags={"version": config['train']['version']}, \
                              save_dir=path_mlf_logger, \
                              log_model=False)
    
    if(config['model']['name']=='cgcnn' or config['model']['name']=='alignn'):
        data = GNNDataModule(**config['data'])
        config['data']['lmdb_exist']=True
        print('-----------------------------------------')
        print(f'atomic features: {config["data"]["atomic_features"]}')
        print('-----------------------------------------')
        print(f'compound features: {config["data"]["compound_features"]}')
        print('-----------------------------------------')
        model = GNNModel(**config)
    elif(config['model']['name']=='crabnet'):
        data = CrabNetDataModule(**config['data'])
        print('-----------------------------------------')
        print(f'atomic features: {config["data"]["atomic_features"]}')
        print('-----------------------------------------')
        model = CrabNetLightning(**config)
    
    
    if config['model']['classification']:
        checkpoint_callback = ModelCheckpoint(monitor='val_mcc', \
                                                mode="max", \
                                                save_top_k=config['train']['number_of_checkpoints'], \
                                                dirpath=f"trained_models/{config['model']['name']}/", \
                                                filename='{epoch:02d}_{val_mcc:.2f}')
    else:
        checkpoint_callback = ModelCheckpoint(monitor='val_mae', \
                                                mode="min", \
                                                save_top_k=config['train']['number_of_checkpoints'], \
                                                dirpath=f"trained_models/{config['model']['name']}/", \
                                                filename='{epoch:02d}_{val_mae:.3f}')

                        
            
    if config['optim']['swa']:
        swa = StochasticWeightAveraging(swa_lrs=config['optim']['swa_lr'], annealing_epochs=5, annealing_strategy='linear', swa_epoch_start=config['optim']['swa_start'])
        early_stopping_cb = EarlyStopping(
                                            monitor='val_mae',
                                            patience=config['train']['patience'],
                                        )

        callbacks=[early_stopping_cb,checkpoint_callback, swa]
        
        assert all(isinstance(cb, PLCallback) for cb in callbacks), [type(cb) for cb in callbacks]
        
        trainer = Trainer(
                            max_epochs=config['train']['epochs'],
                            accelerator=config['train']['accelerator'],
                            devices=config['train']['devices'],
                            logger=mlf_logger,
                            callbacks=callbacks
                        )
    else:
        trainer = Trainer(max_epochs=config['train']['epochs'], \
                                accelerator=config['train']['accelerator'],  \
                                devices=config['train']['devices'], \
                                logger=mlf_logger, \
                                callbacks=[EarlyStopping(monitor='val_loss', patience=config['train']['patience']), 
                                        checkpoint_callback])



    trainer.fit(model, datamodule=data)
    if config['optim']['swa']:
        swa_path = Path(__file__).resolve().parent.parent / 'trained_models'/ config['model']['name']/'swa_model.ckpt'
        trainer.save_checkpoint(swa_path)
    

    if(config['model']['name'] == 'cgcnn'):
        with mlflow.start_run(run_id=mlf_logger.run_id):
            if checkpoint_callback.best_model_path:
                mlflow.pytorch.log_model(
                    pytorch_model=trainer.model,
                    artifact_path="best_cgcnn_model", 
                    registered_model_name="cgcnn_model"
                )
    elif(config['model']['name'] == 'alignn'):
        with mlflow.start_run(run_id=mlf_logger.run_id):
            if checkpoint_callback.best_model_path:
                mlflow.pytorch.log_model(
                    pytorch_model=trainer.model,
                    artifact_path="best_alignn_model", 
                    registered_model_name="alignn_model"
                )
    elif(config['model']['name'] == 'crabnet'):
        with mlflow.start_run(run_id=mlf_logger.run_id):
            if checkpoint_callback.best_model_path:
                mlflow.pytorch.log_model(
                    pytorch_model=trainer.model,
                    artifact_path="best_crabnet_model", 
                    registered_model_name="crabnet_model"
                )