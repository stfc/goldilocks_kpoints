import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, Callback
from lightning.pytorch.loggers import MLFlowLogger
import mlflow.pytorch
import argparse

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from datamodules.datamodule import GNNDataModule
from utils.utils import load_yaml_config
from models.modelmodule import GNNModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config_file",
                        default="cgcnn.yaml",
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
    data = GNNDataModule(**config['data'])
    config['data']['lmdb_exist']=True
    model = GNNModel(**config)
    
    if(config['model']['name'] == 'cgcnn'):
        if config['model']['classification']:
            checkpoint_callback = ModelCheckpoint(monitor='val_mcc', \
                                                mode="max", \
                                                save_top_k=config['train']['number_of_checkpoints'], \
                                                dirpath='trained_models/cgcnn/', \
                                                filename='cgcnn_{epoch:02d}_{val_mcc:.2f}')
        else:
            checkpoint_callback = ModelCheckpoint(monitor='val_mae', \
                                                mode="min", \
                                                save_top_k=config['train']['number_of_checkpoints'], \
                                                dirpath='trained_models/cgcnn/', \
                                                filename='cgcnn_{epoch:02d}_{val_mae:.2f}')
            
    elif(config['model']['name'] == 'alignn'):
        if config['model']['classification']:
            checkpoint_callback = ModelCheckpoint(monitor='val_mcc', \
                                                mode="max", \
                                                save_top_k=config['train']['number_of_checkpoints'], \
                                                dirpath='trained_models/alignn/', \
                                                filename='alignn_{epoch:02d}_{val_mcc:.2f}')
        else:
            checkpoint_callback = ModelCheckpoint(monitor='val_mae', \
                                                mode="min", \
                                                save_top_k=config['train']['number_of_checkpoints'], \
                                                dirpath='trained_models/alignn/', \
                                                filename='alignn_{epoch:02d}_{val_mae:.2f}')
    
    if config['optim']['swa']:
        swa = StochasticWeightAveraging(swa_lrs=config['optim']['swa_lr'], swa_epoch_start=config['optim']['swa_start'])
    
        trainer = Trainer(max_epochs=config['train']['epochs'], \
                                accelerator=config['train']['accelerator'],  \
                                devices=config['train']['devices'], \
                                logger=mlf_logger, \
                                callbacks=[EarlyStopping(monitor='val_loss', patience=config['train']['patience']), 
                                        checkpoint_callback, swa])
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