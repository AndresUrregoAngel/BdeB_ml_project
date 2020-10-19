import argparse
import pandas as pd
import numpy as np
import json
import os
import pickle
import boto3
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from io import StringIO
from joblib import dump, load
from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

logging.basicConfig(level=logging.INFO)


def _customPreprocessing(X):
    logging.info("starting the custom process for data preprocessing")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    scaler = "standard_scaler_fit"
    localscaler = f"{dir_path}/{scaler}"
    pca = "pca_fit"
    localpca = f"{dir_path}/{pca}"
    
    s3cli = boto3.client("s3")
    logging.info("download config scaler and pca")
    s3cli.download_file("bucket",f"mlproject/configs/{scaler}", localscaler)
    s3cli.download_file("bucket",f"mlproject/configs/{pca}", localpca)
    
    logging.info(f"payload to be scaled and reduced {X}")
    logging.info("load config files for return")
    scaler_processing = load(localscaler)
    pca_processing = load(localpca)
    
    X_sc = scaler_processing.transform(X)
    logging.info(f"data has been scaled as : {X_sc}")
    X_pca = pca_processing.transform(X_sc)
    logging.info(f"input features after pca : {X_pca}")
    
    return X_pca
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    # parser.add_argument('--max_leaf_nodes', type=int, default=-1)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_csv(file, sep=",", engine="python") for file in input_files ]
    fulldata = pd.concat(raw_data)

    # labels are in the first column
    X = fulldata.drop(['target'],axis=1)
    y = fulldata['target']
      
    # Preprocessing scaler
    
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
  
      
    # Split datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Here we support a single hyperparameter, 'max_leaf_nodes'. Note that you can add as many
    # as your training my require in the ArgumentParser above.
    # max_leaf_nodes = args.max_leaf_nodes
    logging.info(f"the current shape for the features dataset is {X_train.shape}")
    # Now use scikit-learn's decision tree classifier to train the model.
    clf = DecisionTreeClassifier(max_depth=3)
#     X_normalized = _customPreprocessing(X_train)
    clf = clf.fit(X_train, y_train)

    # Print the coefficients of the trained classifier, and save the coefficients
    dump(clf, os.path.join(args.model_dir, "model.joblib"))
    

    
def input_fn(input_data, content_type):
    """Parse input data payload
    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    logging.info(f"the payload content is {input_data}")
    items = [ float(item)  for item in input_data.split(',')]  
    input_pr = np.array(items).reshape(1,-1)
    X_normalized = _customPreprocessing(input_pr)
    logging.info(f"the payload content normalized is {X_normalized}")
    
    return X_normalized
    

     
def predict_fn(input_data, model):
    """Preprocess input data
    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().
    The output is returned in the following order:
        rest of features either one hot encoded or standardized
    """
  
    prediction = model.predict(input_data)
    pred_prob = model.predict_proba(input_data)
   
    return pred_prob
    
def output_fn(prediction, accept):
    """Format prediction output
    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), accept, mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), accept, mimetype=accept)
    else:
        raise Exception("{} accept type is not supported by this script.".format(accept))
    
    
    
def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    clf = load(os.path.join(model_dir, "model.joblib"))
    return clf
