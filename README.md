# Multiresolution Knowledge Distillation for Anomaly Detection.

```

1- Clone this repo:
``` bash
git clone https://github.com/BaoSon21/AnomalyDetection.git
cd AnomalyDetection
Install requirements.txt
```
### 2- Datsets:
This repository performs Anomaly Detection in the following datasets: MVTecAD,
Furthermore, Anomaly Localization have been performed on MVTecAD dataset.

You have to download [MVTecAD](https://www.mvtec.com/company/research/datasets/mvtec-ad/), and unpack them into the `Dataset` folder.

##### For Localization test you should remove the `good` folder in `{mvtec_class_name}/test/` folder.

### 3- Train the Model:
Start the training using the following command. The checkpoints will be saved in the folder `outputs/{experiment_name}/{dataset_name}/checkpoints`.

Train parameters such as experiment_name, dataset_name, normal_class, batch_size and etc. can be specified in `configs/config.yaml`.
``` bash
python train.py --config configs/config.yaml
```

### 4- Test the Trained Model:
Test parameters can also be specified in `configs/config.yaml`.
``` bash
python test.py --config configs/config.yaml
```
