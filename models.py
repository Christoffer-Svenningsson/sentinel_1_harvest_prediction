from lightgbm import LGBMRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from enum import Enum, auto

class ModelType(Enum):
    LGBM_Regression = auto()
    FNN_Regression = auto()
    FNN_Large_Regression = auto()

def get_lgbm_regressor(X_train, y_train):
    return LGBMRegressor().fit(X_train, y_train)

def get_fnn_regressor(X_train, y_train):
    n_features = X_train.shape[1]
    
    inputs = keras.Input(shape=(n_features,))
    x = layers.Dense(max(n_features*5, 500), activation="relu")(inputs)
    x = layers.Dense(max(n_features*3, 300), activation="relu")(x)
    x = layers.Dense(max(n_features*2, 200), activation="relu")(x)
    # Max 200, min 50 nodes
    x = layers.Dense(min(max(int(n_features* 1/2), 50), 200), activation="relu")(x)
    # Max 50, min 20 nodes
    x = layers.Dense(min(max(int(n_features* 1/5), 20), 50), activation="relu")(x)
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="MyFNN_regressor")
    
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError(),
    )
    
    callbacks = [    
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    ]
    
    history = model.fit(
        X_train, 
        y_train, 
        batch_size=32, 
        validation_split=0.1, 
        callbacks=callbacks,
        epochs=50, 
        verbose=0,
    )
    
    return model

def get_fnn_large(X_train, y_train):
    n_features = X_train.shape[1]
    
    inputs = keras.Input(shape=(n_features,))
    x = layers.Dense(max(n_features*15, 500), activation="relu")(inputs)
    x = layers.Dense(max(n_features*13, 500), activation="relu")(x)
    x = layers.Dense(max(n_features*11, 500), activation="relu")(x)
    x = layers.Dense(max(n_features*5, 500), activation="relu")(x)
    x = layers.Dense(max(n_features*5, 500), activation="relu")(x)
    x = layers.Dense(max(n_features*5, 500), activation="relu")(x)
    x = layers.Dense(max(n_features*5, 500), activation="relu")(x)
    x = layers.Dense(max(n_features*5, 500), activation="relu")(x)
    x = layers.Dense(max(n_features*5, 500), activation="relu")(x)
    x = layers.Dense(max(n_features*5, 500), activation="relu")(x)
    x = layers.Dense(max(n_features*5, 500), activation="relu")(x)
    x = layers.Dense(max(n_features*5, 500), activation="relu")(x)
    x = layers.Dense(max(n_features*5, 500), activation="relu")(x)
    x = layers.Dense(max(n_features*3, 300), activation="relu")(x)
    x = layers.Dense(max(n_features*2, 200), activation="relu")(x)
    # Max 200, min 50 nodes
    x = layers.Dense(min(max(int(n_features* 1/2), 50), 200), activation="relu")(x)
    # Max 50, min 20 nodes
    x = layers.Dense(min(max(int(n_features* 1/5), 20), 50), activation="relu")(x)
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="MyFNN_regressor")
    
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError(),
    )
    
    callbacks = [    
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=100)
    ]
    
    history = model.fit(
        X_train, 
        y_train, 
        batch_size=64, 
        validation_split=0.1, 
        callbacks=callbacks,
        epochs=1000, 
        verbose=1,
    )
    
    return model
    
def get_model(X_train, y_train, model_type):
    if model_type == ModelType.LGBM_Regression:
        return get_lgbm_regressor(X_train, y_train)
    
    elif model_type == ModelType.FNN_Regression:
        return get_fnn_regressor(X_train, y_train)
    
    elif model_type == ModelType.FNN_Large_Regression:
        return get_fnn_large(X_train, y_train)
    
    else:
        raise ValueError(f"Model type {model_type} not supported!")