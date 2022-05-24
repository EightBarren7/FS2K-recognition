import torch

train_json_path = './FS2K/anno_train.json'
test_json_path = './FS2K/anno_test.json'
data_dir = './FS2K/sketch/'
batch_size = 8
num_epochs = 100
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

