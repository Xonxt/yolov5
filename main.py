from train import main

if __name__ == "__main__":
    
    # set up the training parameters
    class TrainingArguments:
        
        # path to the HHI Json Format Dataset
        data = "e:\\Datasets\\small_dataset\\metadata.json"
        
        # which YOLO config to use
        cfg = "models\\yolov5l.yaml"
        
        multi_scale = True      # vary image size +/- 50% during training
        val_size = 0.05         # % of the dataset to use for validation
        epochs = 1              # number of epochs
        batch_size = 2          # batch size
        classes = ['rectangle'] # list the names of the classes to use during training or leave as 'rectangle' for ALL classes
                                # e.g. ['gec_object', 'bad_gec_object', 'screw_hole', 'screw_head', 'hand']
    
    # start training
    main(TrainingArguments())