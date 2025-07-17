import random
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore, WinClip
import torch
import gc

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

   
    
    for model_name, model_class in model_classes.items():
        print(f"Current model: {model_name}")
        for category in categories:
            print(f"Current category: {category}")


            results = []
            for run in range(runs):
                print(f"Run {run+1} of {runs}")

                datamodule = MVTecAD(                       #CHOOSES MVTEC AS DATASET
                    root=".\datasets\MVTecAD",
                    category=category,
                    train_batch_size=1,
                    eval_batch_size=1,
                    val_split_mode="from_test",
                    num_workers=1
                )

                datamodule.setup()

                if model_name == "PaDiM":                       #CHOOSES MODEL

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

                engine.fit(                         #TRAINS MODEL
                    model = model,
                    datamodule = datamodule,
                )
                print("Training done!")

                engine.predict(                    #USES MODEL
                    model = model,
                    datamodule = datamodule,
                )
                print("Prediction done!")

                test_result = engine.test(          #TESTS MODEL
                    model = model,
                    datamodule = datamodule,
                )
                print(f"Test done for {model_name}, {category}, run {run+1}")

                for result in test_result:
                    result.update({"run": run+1, "method": model_name, "category": category})
                    results.append(result)

                del model                       #CLEAN FOR MORE VRAM
                del engine
                torch.cuda.empty_cache()
                gc.collect

if __name__ == "__main__":
    main()

