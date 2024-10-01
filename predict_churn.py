import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(filepath):
    """
    Loads churn data into a DataFrame from a string filepath.
    """
    df = pd.read_csv(filepath)
    return df


def make_predictions(df, threshold=0.5):
    """
  Uses the pycaret best model to make predictions on data in the df dataframe.
    
    """
    model = load_model('K_Neighbors_Classifier')
    predictions = predict_model(model, data=df)
    print(predictions)
    predictions['Churn_prediction'] = (predictions['prediction_label'] >= threshold)
    predictions['Churn_prediction'].replace({True: 'Churn', False: 'No churn'}, inplace=True)
    drop_cols = predictions.columns.tolist()
    drop_cols.remove('Churn_prediction')
    return predictions.drop(drop_cols, axis=1)


if __name__ == "__main__":
  df = load_data(r'C:\Users\thelo\OneDrive\Documentos\School\MSDS_600\Week_5_Assignment\new_churn_data.csv')
  predictions = make_predictions(df)
  print('predictions:')
  print(predictions)
  