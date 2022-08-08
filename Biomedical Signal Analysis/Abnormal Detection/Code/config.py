# Name of dataset
# name = 'brainCT'
name = 'brainCT_cls'

# Base directory for data formats
data_base = '../../../../dataset/'
data_dir = data_base + name

# Base directory for augmented data formats
resize_base = data_base + '/resized/'
split_base = data_base + '/split/'
sam_base = data_base + '/sample/'

# Directory for data formats
resize_dir = resize_base + name
split_dir = split_base + name
sam_dir = sam_base + name

# Train augmentation
rotate_mode = 'strict'

# Validation split
split = 'fix' # [ratio/fix]
val_ratio = 0.2
val_num = 100

# Sampling process
sam_num = 1000
sam_val_num = 100

# Databases for each formats
data_dir = data_base + name
aug_dir = split_base + name
test_dir = split_base + name + '/val'

# model option
batch_size = 64
num_epochs = 100
lr_decay_epoch = 20
feature_size = 500

# Global meanstd
mean = [0.42352142932368259, 0.46167925008138017, 0.49023161345837163]
std = [0.22595048333178538, 0.22503028985594206, 0.23220585942785971]

# model directory
model_dir = './path'
