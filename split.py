import splitfolders
input_folder = "data"
output =  "random"
splitfolders.ratio(input_folder,output,seed=42,ratio=(0.6,0.2,0.2))