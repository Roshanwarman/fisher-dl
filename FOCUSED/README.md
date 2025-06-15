****train.py****
This is the file that does everything: loads in the data, builds & trains the model, saves the results

You will need to download the file 'dataset.pkl' (~1.34 Gb) from the OneDrive in folder "Model stuff/". Keep it in the same folder as train.py. This is the entire dataset of all 333 patients as a pickle file.

Then run ****python train.py**** and it will run the model on the validation set and generate the confusion matrices.

I also uploaded the best_3dcnn.pth model weights from the best results so far in case you wanted to see specifically this model version.