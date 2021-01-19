This repository is a heavily modified version of the YOLO v5 object detector, the original of which can be found [here](https://github.com/ultralytics/yolov5/).

# Running the training

The training can be executed from the command line (see [run_training.sh](/run_training.sh) for an example) or directly from the code (see example in [main.py](/main.py)).

## Running from command-line

To run the training from command line, execute the `train.py` script, and set the following command-line arguments:

* `--data` - should point to a JSON file with the training dataset metadata in the HHI Json Metadata Format (see [this](https://vigitlab.fe.hhi.de/kovalenko/dataset_annotation) repository for more details)
* `--cfg` - defines the model architecture (how deep/large the model will me), should be one of the "yaml" files found in the "[models](/models)" directory
* `--val-size` - a float value between 0 and 1, defining the percentage of the dataset that will be used for validation (default is 0.05/5%)
* `--epochs` - number of training epochs (default is 300)
* `--batch-size` - the batch size for training. Use the largest `--batch-size` the GPU allows (default is 16)
* `--multi-scale` - will vary the sizes of images +/- 50% in each epoch for more robust training (default is False)
* `--classes` - override with a whitespace-separated list of classes, that should be taken from the dataset for training (defaults to 'rectangle', which will take ALL the bounding-box-type classes from the dataset)

Example of training:

```bash
$ python --data dataset/metadata.json --val-size 0.05 --classes gec_object bad_gec_object "screw hole" hand --cfg models/yolov5l.yaml --multi-scale --epochs 100 --batch-size 32 
```

## Running from the code

See the [main.py](/main.py) script for an example. Set the values for each parameter in the `TrainingArguments` class.
Example:

```python
from train import main as start_training
    
# set up the training parameters
class TrainingArguments:                
    data = "e:\\Datasets\\small_dataset\\metadata.json" 
    cfg = "models\\yolov5l.yaml"
    
    multi_scale = True      
    val_size = 0.05         
    epochs = 100           
    batch_size = 32          
    classes = ["gec_object", "bad_gec_object", "screw hole", "hand"]                                

# start training
start_training(TrainingArguments())
```
