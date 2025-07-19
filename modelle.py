import random
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore, WinClip
import torch
import gc
import pandas as pd
import os

torch.set_float32_matmul_precision("medium")

def main():
    random.seed(42) 
    runs = 10           

    model_classes = {
        "PaDiM" : Padim,
        "PatchCore" : Patchcore,
        "WinCLIP" : WinClip,
    }

    categories = ["bottle", "cable","capsule","carpet","grid","hazelnut","metal_nut","pill","screw","toothbrush","tile","transistor","wood","zipper","leather"]
    metrics = ["image_AUROC","image_F1Score","pixel_AUROC","pixel_F1Score"]

   
    column_index = pd.MultiIndex.from_product([categories, metrics])
    row_index = pd.Index(model_classes.keys(), name = "method")
    excel_path = ""                                                     #your path to excel file 
    
    if os.path.exists(excel_path):                                      #checks if excel already exists
        result_df = pd.read_excel(excel_path, header=[0,1], index_col=0)    
    else:
        result_df = pd.DataFrame(index = row_index, columns = column_index)    


    for model_name in model_classes.items():       #loop for every model, category, run
        print(f"Current model: {model_name}")
        for category in categories:                             #checks if some values are already calculated
            allready_done = all(
                pd.notnull(result_df.loc[model_name, (category, metric)]) for metric in metrics)
            if allready_done:
                print(f"Skipping {model_name}, {category}. Training already done!")
                continue

            print(f"Current category: {category}")

            all_results = []                            #saves all results
            for run in range(runs):
                print(f"Run {run+1} of {runs}")

                datamodule = MVTecAD(                       #MVTecAD as dataset 
                    root=".\datasets\MVTecAD",
                    category=category,
                    train_batch_size=1,
                    eval_batch_size=1,
                    val_split_mode="from_test",
                    num_workers=1
                )

                datamodule.setup()

                if model_name == "PaDiM":                       #selects model

                    model = Padim(
                        backbone="resnet50",
                        layers=["layer1", "layer2", "layer3"],
                        pre_trained=True,
                        n_features=100
                    )

                elif model_name == "PatchCore":

                    model = Patchcore(
                        backbone="resnet50",
                        layers=["layer2", "layer3"],
                        pre_trained=True,
                        coreset_sampling_ratio=0.01,
                        num_neighbors=3,
                    )

                elif model_name == "WinCLIP":

                    model = WinClip(
                      k_shots = 50,
                    )
                

                engine = Engine(
                    accelerator="gpu",
                )

                engine.fit(                         #trains model
                    model = model,
                    datamodule = datamodule,
                )
                print("Training done!")

                engine.predict(                    #uses model for prediction
                    model = model,
                    datamodule = datamodule,
                )
                print("Prediction done!")

                test_result = engine.test(          #tests model accuracy
                    model = model,
                    datamodule = datamodule,
                )
                print(f"Test done for {model_name}, {category}, run {run+1}")

                for result in test_result:                                                          #saves values in a dataframe
                    result.update({"run": run+1, "method": model_name, "category": category})
                    all_results.append(result)

                del model                       #clean for more vram 
                del engine
                torch.cuda.empty_cache()
                gc.collect

            all_results_df = pd.DataFrame(all_results)                                          #calculates mean and std of all 10 runs 
            avg_df = all_results_df.groupby(["method"])[metrics].mean().reset_index()
            std_df = all_results_df.groupby(["method"])[metrics].std().reset_index()
            avg_std_df = pd.merge(avg_df, std_df, on =["method"], suffixes = ("", "_std"))

            for i, row in avg_std_df.iterrows():                                                #rounds the values by three decimal places
                method = row["method"]
                for metric in metrics:
                    mean_result = round(row[metric], 3)
                    std_result = round(row[f"{metric}_std"], 3)
                    result_df.loc[(method), (category, metric)] = f"{mean_result} ± {std_result}"

            result_df.to_excel("")                                                                          #your path to excel
            print("Data load to your excel file!")                                                          #loads the mean and std of all models, categories, metrics to your excel file

if __name__ == "__main__":
    main()

