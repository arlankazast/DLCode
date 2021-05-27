import gc

import keras
import numpy as np
import random
import tensorflow as tf
from keras import backend as K


def random_seed(seed_value):
    np.random.seed(seed_value) # cpu vars
    random.seed(seed_value) # Python
    tf.set_random_seed(seed_value)


random_seed(42)

from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.ensemble import GradientBoostingRegressor
# from tqdm.keras import TqdmCallback
from models.optimizer import Optimizer
from models import *

class TextClassifier:
    def __init__(self,config,dataset,evaluator,hybrid=False):
        self.config = config
        self.dataset=dataset
        self.hybrid=hybrid
        self.evaluator=evaluator
        self.classification_model=None
        self.model_name=None
    def init_model(self,X_train,X_train_vec):
        classifier=self.get_classification_model(self.model_name)

        self.classification_model=classifier.init(self.dataset.word_to_index_map,
                                                  X_train.shape[1] if self.config.data.format=="text" else X_train.shape[2],
                                                  X_train_vec.shape[2] if self.config.data.format=="text" and self.config.data.text.branching else None
                                                  )
        # plot_model(self.classification_model,
        #            to_file=os.path.join(self.config.summary_dir, model_name, ".png"),
        #            show_shapes=True)
    def train_test(self,model_name):
        self.model_name=model_name
        # callbacks = self.init_callbacks()
        if self.config.train.k_folds:
            return self.k_folds_train_test()

    def simple_train_test(self,callbacks,X_train,y_train,X_test,y_test,X_valid,y_valid,X_train_vec,X_test_vec,X_valid_vec):
        self.init_model(X_train,X_train_vec)
        self.classification_model.compile(optimizer=Optimizer(self.config).get_optimizer(),
                                          loss=self.config.train.loss_type,
                                          metrics=self.config.train.metrics)
        if self.config.problem=="classification" and self.config.type=="multi_class":
            y_train=to_categorical(y_train)
            y_valid=to_categorical(y_valid)
            y_test=to_categorical(y_test)
        train_inputs=list()
        test_inputs=list()
        valid_inputs=list()
        if self.config.data.format == "text" and self.config.data.text.branching:
            train_inputs.append(X_train)
            train_inputs.append(X_train_vec)
            test_inputs.append(X_test)
            test_inputs.append(X_test_vec)
            valid_inputs.append(X_valid)
            valid_inputs.append(X_valid_vec)

        else:
            train_inputs.append(X_train)
            test_inputs.append(X_test)
            valid_inputs.append(X_valid)
        self.classification_model.fit(train_inputs,
                                      y_train,
                                      epochs=self.config.train.epochs,
                                      batch_size=self.config.train.batch_size,
                                      validation_data=(valid_inputs,y_valid),
                                      verbose=self.config.train.verbose,
                                      callbacks=callbacks)
        self.classification_model.load_weights(self.model_name+self.config.train.checkpoint.weight_file_path_suffix,)
        if self.hybrid:
            #extracting features from last layer
            model_feat=Model(inputs=self.classification_model.input, outputs=self.classification_model.get_layer('concatenate_1').output)
            feat_train = model_feat.predict(train_inputs)
            feat_test = model_feat.predict(test_inputs)
            feat_valid = model_feat.predict(valid_inputs)
            reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.0001, max_depth=1, random_state=0,
                                            loss='ls')

            reg=reg.fit(feat_train, y_train)
            y_pred=reg.predict(feat_test)
        else:
            metricS=' '.join([word.capitalize() for word in self.config.train.metrics[0].split(" ")])
            # evaluate the keras model
            _, train_accuracy = self.classification_model.evaluate(train_inputs, y_train)
            print('Best Training '+metricS+': %.2f' % (train_accuracy * 100))
            _, val_accuracy = self.classification_model.evaluate(valid_inputs, y_valid)
            print('Best Validation '+metricS+': %.2f' % (val_accuracy * 100))
            _, test_accuracy = self.classification_model.evaluate(test_inputs, y_test)
            print('Best Testing '+metricS+': %.2f' % (test_accuracy * 100))
            y_pred=self.classification_model.predict(test_inputs)
        keras.backend.clear_session()
        del self.classification_model
        gc.collect()
        # K.clear_session()
        # tf.compat.v1.reset_default_graph()

        # tf.keras.backend.clear_session()
        # del self.classification_model
        # del callbacks
        # K.clear_session()
        # gc.collect()
        # callbacks=None
        # self.classification_model=None

        return self.evaluator.evaluate(y_test,y_pred)

    def k_folds_train_test(self):
        kFold_Results=[]


        for fold in range(self.config.train.n_folds):
            callbacks = self.init_callbacks()

            import glob, os
            for files in glob.glob("*.hdf5"):

                os.remove(files)
            print("----------------------FOLD: "+str(fold+1)+"----------------------------")
            n_fold=self.dataset.folds[str(fold)]
            results=self.simple_train_test(callbacks,
                                           n_fold["X_train"],n_fold["y_train"],
                                           n_fold["X_test"],n_fold["y_test"],
                                           n_fold["X_valid"],n_fold["y_valid"],
                                           n_fold["X_train"],
                                           n_fold["X_test"],
                                           n_fold["X_valid"],
                                           )
            kFold_Results.append(results)
            print("-------------------------------------------------------------------")
        return kFold_Results
    def eval(self,model_name):
        pass
    def predict(self,model_name):
        pass
    def get_classification_model(self,model_name):


        model = globals()[model_name](self.config)
        return model
    def init_callbacks(self):
        callbacks = []
        if self.config.train.early_stopping.enable:
            callbacks.append(EarlyStopping
                             (monitor=self.config.train.early_stopping.monitor,
                              mode=self.config.train.early_stopping.mode,
                              verbose=self.config.train.early_stopping.verbose,
                              patience=self.config.train.early_stopping.patience)
                             )
        if 'checkpoint' in self.config.train:
            callbacks.append(ModelCheckpoint(filepath=self.model_name+self.config.train.checkpoint.weight_file_path_suffix,
                                             verbose=self.config.train.verbose,
                                             save_best_only=self.config.train.checkpoint.save_best_only,
                                             monitor=self.config.train.checkpoint.monitor)
                             )
        # callbacks.append(TqdmCallback(verbose=0))
        # callbacks.append(TqdmCallback(verbose=0))
        return callbacks