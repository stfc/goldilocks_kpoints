# RF, ...  models from sklearn
import numpy as np
import pandas as pd
import os
import pickle

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier 
from sklearn_quantile import RandomForestQuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure

from utils.compound_features_utils import matminer_composition_features, matminer_structure_features
from utils.compound_features_utils import soap_features, jarvis_features, lattice_features, cgcnn_features 
from utils.compound_features_utils import matscibert_features

class Ensembles:
    """Ensemble models for materials property prediction.
    
    Supports Random Forest, Gradient Boosting, and Histogram Gradient Boosting
    models for both classification and regression tasks.
    """
    def __init__(self, **config):
        """Initialize the ensemble model.
        
        Args:
            **config: Configuration dictionary containing:
                - model: Model configuration (model_name, classification, quantile_regression, etc.)
                - data: Data configuration (root_dir, id_prop_csv, train_ratio, etc.)
                - features: Feature configuration (feature_file, composition_features, etc.)
        """
        # defining the model
        self.classification = config['model']['classification']
        self.quantile_regression = config['model']['quantile_regression']
        self.quantile = config['model']['quantile']
        self.model_name = config['model']['model_name']
        self.n_estimators = config['model']['n_estimators']
        self.learning_rate = config['model']['learning_rate']
        self.random_seed = config['data']['random_seed']
        self.qe_input_path = config['data']['qe_input_files']  
        if(self.model_name == 'RF' and self.quantile_regression == False):
            if self.classification:
                self.model = RandomForestClassifier(n_estimators=self.n_estimators,
                                                    random_state =self.random_seed)
            else:
                self.model = RandomForestRegressor(n_estimators=self.n_estimators,
                                                   random_state =self.random_seed)
        elif(self.model_name == 'RF' and self.quantile_regression == True):
            if self.classification:
                self.model = RandomForestClassifier(n_estimators=self.n_estimators,
                                                    random_state =self.random_seed)
            else:
                self.model = RandomForestQuantileRegressor(n_estimators=self.n_estimators, 
                                                           q=[(1-self.quantile)*0.5,0.5,(1+self.quantile)*0.5],
                                                           random_state =self.random_seed)                                 
        elif(self.model_name == 'GB'):
            if self.classification:
                self.model = GradientBoostingClassifier(learning_rate=self.learning_rate,
                                                        random_state =self.random_seed)
            else:
                self.model = GradientBoostingRegressor(learning_rate=self.learning_rate,
                                                       random_state =self.random_seed)                                     
        elif(self.model_name == 'HGB'):
            if self.classification:
                self.model = HistGradientBoostingClassifier(learning_rate=self.learning_rate,
                                                            random_state =self.random_seed)
            else:
                self.model = HistGradientBoostingRegressor(learning_rate=self.learning_rate,
                                                           random_state =self.random_seed)

        print(self.model)                                              
        # defining features
        self.feature_file = config['features']['feature_file']
        self.composition_features = config['features']['composition_features']
        self.structure_features = config['features']['structure_features']
        self.jarvis_features = config['features']['jarvis_features']
        self.soap_features = config['features']['soap_features']
        self.soap_params = config['features']['soap_params']
        self.lattice_features = config['features']['lattice_features']
        self.cgcnn_features = config['features']['cgcnn_features']
        self.matscibert_features = config['features']['matscibert_features']
        # self.is_metal = config['model']['is_metal']
        # self.is_metal_ckpt_path = config['data']['is_metal_ckpt_path']
    
        self.train_ratio = config['data']['train_ratio']
        self.val_ratio = config['data']['val_ratio']
        self.test_ratio = config['data']['test_ratio']
        self.save_features = config['data']['save_features']
        self.path = config['data']['root_dir']
        self.checkpoint_path = config['features']['checkpoint_path']
        self.lmdb_exist = config['features']['lmdb_exist']
        
        self.data = pd.read_csv(os.path.join(config['data']['root_dir'],config['data']['id_prop_csv']),header=None)
        structures=[]
        compositions=[]
        formulas=[]
        for ind in self.data.index:
            struct = Structure.from_file(os.path.join(config['data']['root_dir'],str(ind)+'.cif'))
            comp = Composition(struct.formula)
            structures.append(struct)
            compositions.append(comp)
            formulas.append(struct.formula)
        self.data['structure']=structures
        self.data['composition']=compositions
        self.data['formula']=formulas
        
    def prep_data(self):
        """Prepare feature data for training.
        
        Either loads pre-computed features from a file or computes features
        using various featurization methods (composition, structure, SOAP, etc.).
        """
        if self.feature_file is not None:
            self.features=np.load(os.path.join(self.path,self.feature_file))
            print(f'** Features are loaded from feature file {os.path.join(self.path,self.feature_file)} **')
        else:
            print('** Features are being calculated **')
            
            features_list = []
            if self.composition_features is not None:
                list_of_feat = [k for k, v in self.composition_features.items() if v]
                composition_features = matminer_composition_features(self.data, list_of_feat)
                features_list.append(composition_features)
            if self.structure_features is not None:
                list_of_feat = [k for k, v in self.structure_features.items() if v]
                structure_features = matminer_structure_features(self.data, list_of_feat)
                features_list.append(structure_features)
            if self.soap_features:
                soap_f = soap_features(self.data,self.soap_params)
                features_list.append(soap_f)
            if self.lattice_features:
                lattice_f = lattice_features(self.data)
                features_list.append(lattice_f)
            if self.jarvis_features:
                jarvis_f = jarvis_features(self.data)
                features_list.append(jarvis_f)
            if self.cgcnn_features:
                cgcnn_f = cgcnn_features(self.checkpoint_path, self.path, self.lmdb_exist)
                features_list.append(cgcnn_f)
            if self.matscibert_features:
                matscibert_f = matscibert_features(df=self.data, data_path=self.qe_input_path)
                features_list.append(matscibert_f)

            self.features = np.concatenate(features_list, axis=1)
            if self.save_features:
                np.save(os.path.join(self.path,'features.npy'), self.features)

    def train_predict_model(self, save_model_path=None, save_model_name=None):
        """Train the ensemble model and make predictions on the test set.
        
        Args:
            save_model_path: Optional path to save the trained model.
            save_model_name: Optional name for the saved model file.
        
        Returns:
            DataFrame containing predictions and ground truth values.
        """
        print('** Model training **')
        if not hasattr(self, "features"):
            print('Calculate feature first Ensembles.prep_data()!!')
            return
        train_ind, test_ind = train_test_split(self.data.index.values, 
                                               test_size=self.test_ratio, 
                                               random_state=self.random_seed)
        
        if self.classification:
            ytrain = np.array(self.data.iloc[train_ind][1].values,dtype='int')
            ytest = np.array(self.data.iloc[test_ind][1].values,dtype='int')
        else:
            ytrain = np.array(self.data.iloc[train_ind][1].values,dtype='float')
            ytest = np.array(self.data.iloc[test_ind][1].values,dtype='float')
        Xtrain = self.features[train_ind,:]
        Xtest = self.features[test_ind,:]

        if self.classification:
            self.model.fit(Xtrain,ytrain)
            ypred = self.model.predict(Xtest)
            probs = self.model.predict_proba(Xtest)
            predictions = pd.DataFrame({"ids": np.array(self.data.iloc[test_ind][0], dtype=int),
                                        "truth": ytest,
                                        "pred": ypred,
                                        "probs": probs[:,1] })
        else:
            if not self.quantile_regression:
                self.model.fit(Xtrain,ytrain)
                if(self.model_name == 'RF'):
                    all_preds = np.stack([tree.predict(Xtest) for tree in self.model.estimators_], axis=0)
                    ypred = np.mean(all_preds, axis=0)
                    ypred_std = np.std(all_preds, axis=0)
                    predictions = pd.DataFrame({
                                                    "ids": np.array(self.data.iloc[test_ind][0], dtype=int),
                                                    "truth": ytest,
                                                    "pred": ypred,
                                                    "pred_std": ypred_std
                                                })
                else:
                    ypred = self.model.predict(Xtest)
                    predictions = pd.DataFrame({
                                                    "ids": np.array(self.data.iloc[test_ind][0], dtype=int),
                                                    "truth": ytest,
                                                    "pred": ypred
                                                })
            elif self.quantile_regression:
                if(self.model_name == 'RF'):
                    self.model.fit(Xtrain,ytrain)
                    all_preds = np.stack([tree.predict(Xtest) for tree in self.model.estimators_], axis=1)  # shape: [n_samples, n_trees]
                    ypred = np.percentile(all_preds, 100*self.quantile, axis=1)
                    predictions = pd.DataFrame({
                                                    "ids": np.array(self.data.iloc[test_ind][0], dtype=int),
                                                    "truth": ytest,
                                                    "pred": ypred
                                                })
                elif(self.model_name == 'GB'):
                    self.model = GradientBoostingRegressor(loss="quantile", 
                                                        alpha=self.quantile,
                                                        learning_rate = self.learning_rate,
                                                        random_state =self.random_seed)
                    self.model.fit(Xtrain,ytrain)
                    ypred = self.model.predict(Xtest)
                    predictions = pd.DataFrame({
                                                    "ids": np.array(self.data.iloc[test_ind][0], dtype=int),
                                                    "truth": ytest,
                                                    "pred": ypred
                                                })
                elif(self.model_name == 'HGB'):
                    self.model = HistGradientBoostingRegressor(loss="quantile", 
                                                        quantile=self.quantile,
                                                        learning_rate = self.learning_rate,
                                                        random_state =self.random_seed)
                    self.model.fit(Xtrain,ytrain)
                    ypred = self.model.predict(Xtest)
                    predictions = pd.DataFrame({
                                                    "ids": np.array(self.data.iloc[test_ind][0], dtype=int),
                                                    "truth": ytest,
                                                    "pred": ypred
                                                })

        if isinstance(save_model_path,str):
            if isinstance(save_model_name,str):
                with open(os.path.join(save_model_path, save_model_name),'wb') as file:
                    pickle.dump(self.model, file)
            else:
                with open(os.path.join(save_model_path, self.model_name),'wb') as file:
                    pickle.dump(self.model, file)
 
        return predictions

                



        
        
