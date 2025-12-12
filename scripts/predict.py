from pytorch_lightning import Trainer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from scipy.stats import spearmanr, kendalltau
import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
import torch
import os
from sklearn.preprocessing import RobustScaler

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from datamodules.gnn_datamodule import GNNDataModule
from datamodules.crabnet_datamodule import CrabNetDataModule
from utils.utils import load_yaml_config, concordance_index
from models.modelmodule import GNNModel
from models.ensembles import Ensembles
from models.modelmodule import CrabNetLightning

def crabnet(**config):
    print(config['model']['name'])
    data =  CrabNetDataModule(**config['data'])
    model = CrabNetLightning(**config)
    list_of_checkpoints = os.listdir(args.checkpoint_path)
    list_of_checkpoints = [f for f in list_of_checkpoints if f != ".DS_Store"]

    df=pd.DataFrame()
    for i,checkpoint in enumerate(list_of_checkpoints):
        trainer = Trainer(max_epochs=5,accelerator='cpu', devices=1)
        pred = trainer.predict(model, ckpt_path=os.path.join(args.checkpoint_path,checkpoint), datamodule=data)
    
        truth=[]
        prediction=[]
        pred_low=[]
        pred_high=[]
        if config['model']['robust_regression']:
            logstd=[]
        elif config['model']['classification']:
            probs=[]
        test_ids=[]

        for idx in range(len(pred)):
            if config['model']['robust_regression']:
                prediction.append(pred[idx][0])
                truth.append(pred[idx][2])
                logstd.append(pred[idx][1])
                test_ids+=pred[idx][3]
            elif config['model']['classification']:
                prediction.append(pred[idx][0])
                truth.append(pred[idx][2])
                probs.append(pred[idx][1])
                test_ids+=pred[idx][3]
            elif config['model']['quantile_regression'] and config['loss']['name'] == 'IntervalScoreLoss':
                pred_low.append(pred[idx][0])
                pred_high.append(pred[idx][1])
                truth.append(pred[idx][2])
                test_ids+=pred[idx][3]
            else:
                prediction.append(pred[idx][0])
                truth.append(pred[idx][1])
                test_ids+=pred[idx][2]

        if config['model']['robust_regression']:        
            truth=torch.cat(truth,dim=0).squeeze(-1)
            prediction=torch.cat(prediction,dim=0).squeeze(-1)
            logstd=torch.cat(logstd,dim=0).squeeze(-1)
        elif config['model']['classification']:        
            truth=torch.cat(truth,dim=0).squeeze(-1)
            prediction=torch.cat(prediction,dim=0).squeeze(-1)
            probs=torch.cat(probs,dim=0).squeeze(-1)
        elif config['model']['quantile_regression'] and config['loss']['name'] == 'IntervalScoreLoss':
            truth=torch.cat(truth,dim=0).squeeze(-1)
            pred_low=torch.cat(pred_low,dim=0).squeeze(-1)
            pred_high=torch.cat(pred_high,dim=0).squeeze(-1)
        else:
            truth=torch.cat(truth,dim=0).squeeze(-1)
            prediction=torch.cat(prediction,dim=0).squeeze(-1)
        
        if config['model']['quantile_regression'] and config['loss']['name'] == 'IntervalScoreLoss':
            df['id']=np.array(test_ids)
            df['truth']=truth
            df['pred_low'+str(i)]=pred_low
            df['pred_high'+str(i)]=pred_high
        else:
            df['id']=np.array(test_ids)
            df['truth_scaled']=truth
            df['prediction'+str(i)]=prediction
            if config['model']['robust_regression']:
                df['logstd'+str(i)]=logstd

    num=len(list_of_checkpoints)
    if config['model']['quantile_regression'] and config['loss']['name'] == 'IntervalScoreLoss':
        prediction_list_low=[]
        prediction_list_high=[]
        for i in range(num):
            prediction_list_low.append('pred_low'+str(i))
            prediction_list_high.append('pred_high'+str(i))

        df["avg_pred_low"] = df[prediction_list_low].mean(axis=1)
        df["avg_pred_high"] = df[prediction_list_high].mean(axis=1)
    else:
        prediction_list=[]
        for i in range(num):
            prediction_list.append('prediction'+str(i))

        df["avg_prediction_scaled"] = df[prediction_list].mean(axis=1)
        truth = truth
        prediction = df["avg_prediction_scaled"].values
        
        if config['data']['scale_y']:
            scaler = RobustScaler()
            data = pd.read_csv(os.path.join(config['data']['root_dir'],config['data']['id_prop_csv']),header=None)
            y=np.array(data[1].values).reshape(-1, 1)
            scaler = scaler.fit(y)
            prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(len(prediction))
            df['prediction'] = prediction
            truth = scaler.inverse_transform(truth.reshape(-1, 1)).reshape(len(truth))
            df['truth'] = truth
        else:
            df['prediction'] = prediction
            df['truth'] = truth

    df.to_csv(args.output_name)

    # metrics
    if config['model']['classification']:
        acc = accuracy_score(truth.cpu().numpy(), prediction)
        f1 = f1_score(truth.cpu().numpy(), prediction, average='macro')
        mcc = matthews_corrcoef(truth.cpu().numpy(), prediction)
        cm = confusion_matrix(truth.cpu().numpy(), prediction)
    
        print(f'Model name: {config["model"]["name"]}')
        print(f'Test set acc: {acc}')
        print(f'Test set f1_score: {f1}')
        print(f'Test set mcc: {mcc}')
        print(f'Test set confusion matrix: {cm}')
    elif config['model']['quantile_regression'] and config['loss']['name'] == 'IntervalScoreLoss':
        print('no metrics for IntervalScoreLoss yet')
    else:
        mse = mean_squared_error(truth,prediction)
        mae = mean_absolute_error(truth,prediction)
        mape = mean_absolute_percentage_error(truth,prediction)
        r2 = r2_score(truth,prediction)
        spearman_corr, _ = spearmanr(truth,prediction)
        kendall_corr, _ = kendalltau(truth,prediction)
        c_index = concordance_index(truth,prediction)
        
        print(f'Model name: {config["model"]["name"]}')
        print(f'Test set MAE: {mae}')
        print(f'Test set MAPE: {mape}')
        print(f'Test set MSE: {mse}')
        print(f'Test set R2 score: {r2}')
        print(f'Test set spearman_corr: {spearman_corr}')
        print(f'Test set kendall_corr: {kendall_corr}')
        print(f'Test set C-index: {c_index}')
        

    if config['prediction']['predict_for_validation']:
        pass
    return

def gnn_neural_networks(**config):
    config['data']['lmdb_exist']=True
    try:
        data = GNNDataModule(**config['data'])
    except:
        config['data']['lmdb_exist']=False
        data = GNNDataModule(**config['data'])
    model = GNNModel(**config)

    list_of_checkpoints = os.listdir(args.checkpoint_path)
    list_of_checkpoints = [f for f in list_of_checkpoints if f != ".DS_Store"]
    
    df=pd.DataFrame()
    for i,checkpoint in enumerate(list_of_checkpoints):
        trainer = Trainer(max_epochs=5,accelerator='cpu', devices=1)
        pred = trainer.predict(model, ckpt_path=os.path.join(args.checkpoint_path,checkpoint), datamodule=data)
    
        truth=[]
        prediction=[]
        pred_low=[]
        pred_high=[]
        if config['model']['robust_regression']:
            logstd=[]
        elif config['model']['classification']:
            probs=[]
        test_ids=[]

        for idx in range(len(pred)):
            if config['model']['robust_regression']:
                prediction.append(pred[idx][0])
                truth.append(pred[idx][2])
                logstd.append(pred[idx][1])
                test_ids+=pred[idx][3]
            elif config['model']['classification']:
                prediction.append(pred[idx][0])
                truth.append(pred[idx][2])
                probs.append(pred[idx][1])
                test_ids+=pred[idx][3]
            elif config['model']['quantile_regression'] and config['loss']['name'] == 'IntervalScoreLoss':
                pred_low.append(pred[idx][0])
                pred_high.append(pred[idx][1])
                truth.append(pred[idx][2])
                test_ids+=pred[idx][3]
            else:
                prediction.append(pred[idx][0])
                truth.append(pred[idx][1])
                test_ids+=pred[idx][2]

        if config['model']['robust_regression']:        
            truth=torch.cat(truth,dim=0).squeeze(-1)
            prediction=torch.cat(prediction,dim=0).squeeze(-1)
            logstd=torch.cat(logstd,dim=0).squeeze(-1)
        elif config['model']['classification']:        
            truth=torch.cat(truth,dim=0).squeeze(-1)
            prediction=torch.cat(prediction,dim=0).squeeze(-1)
            probs=torch.cat(probs,dim=0).squeeze(-1)
        elif config['model']['quantile_regression'] and config['loss']['name'] == 'IntervalScoreLoss':
            truth=torch.cat(truth,dim=0).squeeze(-1)
            pred_low=torch.cat(pred_low,dim=0).squeeze(-1)
            pred_high=torch.cat(pred_high,dim=0).squeeze(-1)
        else:
            truth=torch.cat(truth,dim=0).squeeze(-1)
            prediction=torch.cat(prediction,dim=0).squeeze(-1)
        
        if config['model']['quantile_regression'] and config['loss']['name'] == 'IntervalScoreLoss':
            df['id']=np.array(test_ids)
            df['truth']=truth
            df['pred_low'+str(i)]=pred_low
            df['pred_high'+str(i)]=pred_high
        else:
            df['id']=np.array(test_ids)
            df['truth_scaled']=truth
            df['prediction'+str(i)]=prediction
            if config['model']['robust_regression']:
                df['logstd'+str(i)]=logstd

    num=len(list_of_checkpoints)
    if config['model']['quantile_regression'] and config['loss']['name'] == 'IntervalScoreLoss':
        prediction_list_low=[]
        prediction_list_high=[]
        for i in range(num):
            prediction_list_low.append('pred_low'+str(i))
            prediction_list_high.append('pred_high'+str(i))

        df["avg_pred_low"] = df[prediction_list_low].mean(axis=1)
        df["avg_pred_high"] = df[prediction_list_high].mean(axis=1)
    else:
        prediction_list=[]
        for i in range(num):
            prediction_list.append('prediction'+str(i))

        df["avg_prediction_scaled"] = df[prediction_list].mean(axis=1)
        truth = truth
        prediction = df["avg_prediction_scaled"].values
        
        if config['data']['scale_y']:
            scaler = RobustScaler()
            data = pd.read_csv(os.path.join(config['data']['root_dir'],config['data']['id_prop_csv']),header=None)
            y=np.array(data[1].values).reshape(-1, 1)
            scaler = scaler.fit(y)
            prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(len(prediction))
            df['prediction'] = prediction
            truth = scaler.inverse_transform(truth.reshape(-1, 1)).reshape(len(truth))
            df['truth'] = truth
        else:
            df['prediction'] = prediction
            df['truth'] = truth

    df.to_csv(args.output_name)

    # metrics
    if config['model']['classification']:
        acc = accuracy_score(truth.cpu().numpy(), prediction)
        f1 = f1_score(truth.cpu().numpy(), prediction, average='macro')
        mcc = matthews_corrcoef(truth.cpu().numpy(), prediction)
        cm = confusion_matrix(truth.cpu().numpy(), prediction)
    
        print(f'Model name: {config["model"]["name"]}')
        print(f'Test set acc: {acc}')
        print(f'Test set f1_score: {f1}')
        print(f'Test set mcc: {mcc}')
        print(f'Test set confusion matrix: {cm}')
    elif config['model']['quantile_regression'] and config['loss']['name'] == 'IntervalScoreLoss':
        print('no metrics for IntervalScoreLoss yet')
    else:
        mse = mean_squared_error(truth,prediction)
        mae = mean_absolute_error(truth,prediction)
        mape = mean_absolute_percentage_error(truth,prediction)
        r2 = r2_score(truth,prediction)
        spearman_corr, _ = spearmanr(truth,prediction)
        kendall_corr, _ = kendalltau(truth,prediction)
        c_index = concordance_index(truth,prediction)
        
        print(f'Model name: {config["model"]["name"]}')
        print(f'Test set MAE: {mae}')
        print(f'Test set MAPE: {mape}')
        print(f'Test set MSE: {mse}')
        print(f'Test set R2 score: {r2}')
        print(f'Test set spearman_corr: {spearman_corr}')
        print(f'Test set kendall_corr: {kendall_corr}')
        print(f'Test set C-index: {c_index}')
        

    if config['prediction']['predict_for_validation']:
        pass
    return

def ensembles(save_model_path=None, save_model_name=None, **config):
    model = Ensembles(**config)
    model.prep_data()
    predictions = model.train_predict_model(save_model_path=save_model_path, 
                                            save_model_name=save_model_name)
    
    predictions.to_csv(args.output_name)
    if(config['model']['classification']):
        print('** Model evaluation **')
        acc = accuracy_score(predictions['truth'].values, predictions['pred'].values)
        f1 = f1_score(predictions['truth'].values, predictions['pred'].values, average='macro')
        mcc = matthews_corrcoef(predictions['truth'].values, predictions['pred'].values)
        cm = confusion_matrix(predictions['truth'].values, predictions['pred'].values)

        print(f'Model name: {config["model"]["model_name"]}')
        print(f'Test set acc: {acc}')
        print(f'Test set f1_score: {f1}')
        print(f'Test set mcc: {mcc}')
        print(f'Test set confusion matrix: {cm}')

    else:
        print('** Model evaluation **')
        mse = mean_squared_error(predictions['truth'].values, predictions['pred'].values)
        mae = mean_absolute_error(predictions['truth'].values, predictions['pred'].values)
        mape = mean_absolute_percentage_error(predictions['truth'].values, predictions['pred'].values)
        r2 = r2_score(predictions['truth'].values, predictions['pred'].values)
        spearman_corr, _ = spearmanr(predictions['truth'].values, predictions['pred'].values)
        kendall_corr, _ = kendalltau(predictions['truth'].values, predictions['pred'].values)
        c_index = concordance_index(predictions['truth'].values, predictions['pred'].values)
        
        print(f'Model name: {config["model"]["model_name"]}')
        print(f'Test set MAE: {mae}')
        print(f'Test set MAPE: {mape}')
        print(f'Test set MSE: {mse}')
        print(f'Test set R2 score: {r2}')
        print(f'Test set spearman_corr: {spearman_corr}')
        print(f'Test set kendall_corr: {kendall_corr}')
        print(f'Test set C-index: {c_index}')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction script")

    parser.add_argument("--config_file",
                        default="ensembles.yaml",
                        help="Provide the experiment configuration file")
    parser.add_argument("--checkpoint_path",
                        default="trained_models/cgcnn/basic_l1robust_6_oct_paper/",
                        help="Provide the path to model checkpoint")
    parser.add_argument("--output_name",
                        default="output/GB/errors.csv",
                        help="Provide the path to save predictions")
    parser.add_argument("--ensemble_model_save_name",
                        default="GB_CSLM.pkl",
                        help="Provide the path to save predictions")
    parser.add_argument("--ensemble_model_save_path",
                        default="trained_models/GB/",
                        help="Provide the path to save predictions")

    args = parser.parse_args(sys.argv[1:])

    path_config = Path(__file__).resolve().parent.parent / 'configs' / args.config_file
    config = load_yaml_config(path_config)
    
    if(args.config_file == 'ensembles.yaml'):
        ensembles(save_model_path=args.ensemble_model_save_path, 
                  save_model_name=args.ensemble_model_save_name,
                  **config)
    elif(args.config_file == 'crabnet.yaml'):
        crabnet(**config)
    else:
        gnn_neural_networks(**config)
    
    



