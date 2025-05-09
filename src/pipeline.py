from src.utils.files import create_result_folder
from src.models import ensamble
from src.models import NN_pytorch
import pandas as pd

# from src.train_test.train import train
# from src.train_test.test import test

# def check_device(device):
#     """
#     Checks if the specified device is 'cuda' and sets it to CUDA if available, otherwise defaults to CPU.

#     Args:
#     device (str): Device name, typically 'cuda' or 'cpu'.

#     Returns:
#     torch.device: Torch device object representing the selected device.
#     """
#     if device == 'cuda':
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     return device


def load_model(model_name):
    """
    Load a machine learning model based on the specified model name in model_params.

    Args:
    model_params (dict): Parameters containing the 'model_name' key to determine which model to load.

    Returns:
    object: Loaded machine learning model instance corresponding to the specified model_name.
    """
    model = None
    
    if model_name == 'ensamble':
        model = ensamble.EnhancedSeasonalColorModel()

    elif model_name == 'nn_pytorch':
        model = NN_pytorch.NNSeasonalColorModel()

    return model


def run_model(model_params, data_params):
    """
    Run the machine learning model training and evaluation pipeline.

    Args:
    model_params (dict): Parameters related to the machine learning model configuration.
    data_params (dict): Parameters related to the dataset configuration.

    Returns:
    dict: A dictionary containing paths to saved results ('save_path' and 'save_cv_path').
    """
    save_path = create_result_folder(model_params, data_params)
    
    model = load_model(model_params['model_name'])
    model.train_model(data_params['data_train_path'], data_params['data_val_path'], model_params=model_params, save_name=model_params['configs_file_name'])
    model.eval_model(data_params['data_val_path'], save_path)
    
    return save_path



def test_model(model_name, model_path, test_dataset_path):
    model = load_model(model_name)
    model.load_params_model(model_path)

    df = pd.read_csv(test_dataset_path)
    counter_correct_pred = 0
    for i, img in df.iterrows():
        result = model.predict_season(img.to_frame().T)
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Real Season: {img['season']}")
            print(f"Predicted Season: {result['predicted_season']}")
            print("\nConfidence Scores:")
            for season, prob in result['confidence_scores'].items():
                print(f"{season}: {prob:.2%}")

            if result['predicted_season'] == img['season']:
                counter_correct_pred += 1

            print('\n------------------------- \n')
    
    print(f"\n\nCantidad de predicciones correctas: {counter_correct_pred}")