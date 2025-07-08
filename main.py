import argparse

from src.utils.files import read_configs
from src.pipeline import run_model, test_model

# Como correrlo:
# python main.py --configs configs/mod_resNeXt_weighted_avg_superDataset_balanced.py --path_test data/split_dataset/SuperDataset/test_SuperDataset.csv --path_model runs/model_mod_resNeXt_weighted_avg_superDataset_balanced.pt 



'''
el argumento de --topk k permite tomar una prediccion como correcta si la etiqueta verdadera 
esta entre las k etiquetas con mayor probabilidad predicha por el modelo.

el argumento --seasons_only permite que el test se evalue sobre las 4 seasons en vez de las 12 categorias
'''

def main(args):
    check_args = read_configs(args)
    
    if check_args == -1:
        print("Some parameters were missing.")
        return -1
    
    elif check_args[0] == 0:
        _, model, data = check_args
        run_model(model, data)

    else:
        _, model, model_path, test_dataset_path = check_args
        seasons_only = getattr(args, "seasons_only", False)
        topk = getattr(args, "topk", None)
        test_model(model, model_path, test_dataset_path, seasons_only=seasons_only, topk=topk)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, help="Should add path of the configs file", required=False)
    parser.add_argument("--path_test", type=str, help="Should add path to test dataset", required=False)
    parser.add_argument("--path_model", type=str, help="Should add path to saved model", required=False)
    parser.add_argument("--seasons_only", action="store_true", help="If set, test only on seasons (not subcategories)")
    parser.add_argument("--topk", type=int, help="If set, mark prediction as true if the correct label is in the top k predicted probabilities")
    args = parser.parse_args()
    main(args)