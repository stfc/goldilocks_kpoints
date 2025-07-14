# RF, ...  models from sklearn
import numpy as np
import pandas as pd
import os
import pickle

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier 
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
import matminer
from matminer.featurizers.structure.composite import JarvisCFID
from dscribe.descriptors import SOAP
from pymatgen.io.ase import AseAtomsAdaptor
from dscribe.descriptors import SOAP
from matplotlib import pyplot as plt
from matminer.datasets import load_dataset
from matminer.featurizers.base import MultipleFeaturizer

from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure


class Ensembles:
    def __init__(self, **config):
        # defining the model
        self.classification = config['model']['classification']
        self.quantile_regression = config['model']['quantile_regression']
        self.quantile = config['model']['quantile']
        self.model_name = config['model']['model_name']
        self.n_estimators = config['model']['n_estimators']
        self.learning_rate = config['model']['learning_rate']
        self.random_seed = config['data']['random_seed']  
        if(self.model_name == 'RF'):
            if self.classification:
                self.model = RandomForestClassifier(n_estimators=self.n_estimators,
                                                    random_state =self.random_seed)
            else:
                self.model = RandomForestRegressor(n_estimators=self.n_estimators,
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
                                                        
        # defining features
        self.composition_features = config['model']['composition_features']
        self.structure_features = config['model']['structure_features']
        self.jarvis_features = config['model']['jarvis_features']
        self.soap_features = config['model']['soap_features']
        self.soap_params = config['model']['soap_params']
        
        if self.composition_features is not None:
            list_of_feat = [k for k, v in self.composition_features.items() if v]
            list_of_feat_meth=[]
            for feat in list_of_feat:
                if hasattr(matminer.featurizers.composition, feat):
                    if(feat=='ElementProperty'):
                        method = getattr(matminer.featurizers.composition , feat).from_preset('magpie')
                    else:
                        method = getattr(matminer.featurizers.composition , feat)()
                    list_of_feat_meth.append(method)
            
            # Use individual featurizers instead of MultipleFeaturizer to avoid argument passing issues
            self.composition_featurizer = MultipleFeaturizer(list_of_feat_meth)

        if self.structure_features is not None:
            list_of_feat = [k for k, v in self.structure_features.items() if v]
            list_of_feat_meth=[]
            for feat in list_of_feat:
                if(feat=='GlobalSymmetryFeatures'):
                    props=["spacegroup_num", "crystal_system_int", "is_centrosymmetric"]
                    method = getattr(matminer.featurizers.structure, feat)(props)
                elif(feat=='DensityFeatures'):
                    props=["density", "vpa", "packing fraction"]
                    method = getattr(matminer.featurizers.structure, feat)(props)
                list_of_feat_meth.append(method)
            self.structure_featurizer = MultipleFeaturizer(list_of_feat_meth)

        if self.jarvis_features:
            self.jarvis_featurizer = JarvisCFID()

        if self.soap_features:
            self.soap_featurizer = SOAP(species=['X'],  # or whatever elements you're using
                                        r_cut= self.soap_params['r_cut'],
                                        n_max=self.soap_params['n_max'],
                                        l_max=self.soap_params['l_max'],
                                        sigma=self.soap_params['sigma'],
                                        periodic=True,
                                        sparse=False)

         
        self.train_ratio = config['data']['train_ratio']
        self.val_ratio = config['data']['val_ratio']
        self.test_ratio = config['data']['test_ratio']
        self.save_features = config['data']['save_features']
        self.path = config['data']['root_dir']
        
        
        self.data = pd.read_csv(os.path.join(config['data']['root_dir'],config['data']['id_prop_csv']),header=None)
        structures=[]
        compositions=[]
        for ind in self.data.index:
            struct = Structure.from_file(os.path.join(config['data']['root_dir'],str(ind)+'.cif'))
            comp = Composition(struct.formula)
            structures.append(struct)
            compositions.append(comp)
        self.data['structure']=structures
        self.data['composition']=compositions
        
    def prep_data(self):
        print('** Features are calculated **')
        feature_len = 0
        if self.composition_features is not None:
            comp_feat_len = len(self.composition_featurizer.featurize(self.data.iloc[0]['composition']))
            feature_len += comp_feat_len
        else:
            comp_feat_len=0
        if self.structure_features is not None:
            struct_feat_len = len(self.structure_featurizer.featurize(self.data.iloc[0]['structure']))
            feature_len += struct_feat_len
        else:
            struct_feat_len=0
        if self.jarvis_features:
            jarvis_feat_len = len(self.jarvis_featurizer.featurize(self.data.iloc[0]['structure']))
            feature_len += jarvis_feat_len 
        else:
            jarvis_feat_len=0
        if self.soap_features:
            atoms = AseAtomsAdaptor.get_atoms(self.data.iloc[0]['structure'])
            atoms.set_chemical_symbols(["X"] * len(atoms))
            soap=self.soap_featurizer.create(atoms).mean(axis=0)
            soap_feat_len = len(soap)
            feature_len += soap_feat_len
        else:
            soap_feat_len=0
        print(f'feature length is {feature_len}...')
        features=np.zeros((len(self.data),feature_len))
        
        if self.composition_features is not None:
            for i,comp in enumerate(self.data['composition'].values):
                features[i,:comp_feat_len]=self.composition_featurizer.featurize(comp)
        if self.structure_features is not None:
            for i, struct in enumerate(self.data['structure'].values):
                try:
                    features[i, comp_feat_len:comp_feat_len+struct_feat_len] = self.structure_featurizer.featurize(struct)
                except Exception as e:
                    print(f"Warning: Structure featurization failed for index {i}, formula {struct.formula} with error: {e}")
                    features[i, comp_feat_len:comp_feat_len+struct_feat_len] = np.zeros(struct_feat_len)
        if self.jarvis_features:
            for i,struct in enumerate(self.data['structure'].values):
                features[i,comp_feat_len+struct_feat_len:comp_feat_len+struct_feat_len+jarvis_feat_len]=\
                    self.jarvis_featurizer.featurize(struct) 
        if self.soap_features:
            for i,struct in enumerate(self.data['structure'].values):
                atoms = AseAtomsAdaptor.get_atoms(struct)
                atoms.set_chemical_symbols(["X"] * len(atoms))
                soap=self.soap_featurizer.create(atoms).mean(axis=0)
                features[i,comp_feat_len+struct_feat_len+jarvis_feat_len:comp_feat_len+struct_feat_len+jarvis_feat_len+soap_feat_len]=soap
        
        features=np.nan_to_num(features, copy=True, nan=0.0, posinf=None, neginf=None)
        self.features = features
        if self.save_features:
            np.save(os.path.join(self.path,'features.npy'), features)

    def train_predict_model(self, save_model_path=None, save_model_name=None):
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
            predictions = pd.DataFrame({
                                                    "ids": np.array(self.data.iloc[test_ind][0], dtype=int),
                                                    "truth": ytest,
                                                    "pred": ypred,
                                                    "probs": probs
                                                })
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

                



        
        
