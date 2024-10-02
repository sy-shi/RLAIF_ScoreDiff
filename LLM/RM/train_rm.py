import torch, csv,os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
#from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
import numpy as np
import argparse, os
import torch.nn.functional as F
import pickle
from enum import Enum
#from ray.rllib.models.torch.misc import SlimConv2d

#AGENT = 0   # {0,1}
LARGE_NETWORK = False
ORI_NETWORK = True
USE_GPT4 = False
USE_STARLING = False
USE_TRUE_DATA = True

EVALUATE = True
VALIDATION = not EVALUATE

INIT_LR =  [0.001, 0.01][LARGE_NETWORK]
BATCH_SIZE = 16#[32, 768][LARGE_NETWORK]

class Ranking(Enum):
    GREATER = 0
    LESSER = 1
    EQUAL = 2

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    default="train",
    choices=["train", "validate"],
    help="train or validate"
)

parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="the path of the model we need to validate"
)

# Define the Dataset
class RewardDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.obs_data = [[], []]
        self.eq_num = 0

        if not USE_TRUE_DATA:        
            # Open the CSV file for reading
            with open(FILES[d], 'r') as file:
                reader = csv.reader(file)
                # Read all rows into the data list
                self.obs_data[d] = list(reader)[:size]
        else:
            training_data = pickle.load(open(GT_FILE, "rb"))

            for n in range(len(training_data["input"])):
                if training_data["target"][n] == Ranking.GREATER:
                    for d in range(2):
                        self.obs_data[d].append(training_data["input"][n][1-d]['image'])
                elif training_data["target"][n] == Ranking.LESSER or training_data["target"][n] == Ranking.EQUAL:
                    for d in range(2):
                        self.obs_data[d].append(training_data["input"][n][d]['image'])
                else:
                    self.eq_num += 1
                if len(self.obs_data[0]) > size:
                    break
            #print(len(self.obs_data[0]), self.eq_num)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = [None,None]
        for d in range(2):
            if not USE_TRUE_DATA:
                data[d] = np.array([float(ob) for ob in self.obs_data[d][idx]]).reshape(9,9,3).astype(np.float32)
            else:
                #print(np.array(self.obs_data).shape)
                data[d] = np.array(self.obs_data[d][idx]).astype(np.float32)
        return torch.tensor(data[0]), torch.tensor(data[1])
    
class ValidDataset(RewardDataset):
    def __init__(self, size):
        self.size = size
        self.obs_data = [[],[]]
        # if EVALUATE:
        #     files = [os.getcwd()+'/data/validating_states_high' + str(AGENT) + '-v2-clean.csv', os.getcwd()+'/data/validating_states_low' + str(AGENT) + '-v2-clean.csv']
        # elif VALIDATION:
            #files = FILES

        if not USE_TRUE_DATA:
            files = FILES
            for d in range(2):
                # Open the CSV file for reading
                with open(files[d], 'r') as file:
                    reader = csv.reader(file)
                    # Read all rows into the data list
                    self.obs_data[d] = list(reader)[-size:]
        else:
            training_data = pickle.load(open(GT_FILE, "rb"))
            for n in range(len(training_data["input"])):
                idx = [n, - (n + 1)][EVALUATE]
                if training_data["target"][idx] == Ranking.GREATER:
                    for d in range(2):
                        self.obs_data[d].append(training_data["input"][idx][1-d]['image'])
                elif training_data["target"][idx] == Ranking.LESSER:
                    for d in range(2):
                        self.obs_data[d].append(training_data["input"][idx][d]['image'])
                if len(self.obs_data[0]) > size:
                    break
                

# Define the Model
class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()

        if ORI_NETWORK:
            self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
            self.conv4 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
            self.pad = nn.ConstantPad2d((2, 2, 1, 1), 0)

            #self.fc1 = nn.Linear(64*2*2, 256)
            #self.fc2 = 2*torch.sigmoid(nn.Linear(256, 1)) -1 
            self.fc0 = nn.Linear(128*4*9, 256)
            self.fc1 = nn.Linear(256, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 16)
            self.fc5 = nn.Linear(16, 1)

        # 2fc
        # by stage accuracy [[0.69551777 0.89204545        nan]
        # [0.62132353 0.9978022         nan]]
        # vf accuracy [0.764 0.792]

        # TEST with clip
        # by stage accuracy [[0.55023184 0.78125           nan]
        # [0.45772059 0.90549451        nan]]
        # vf accuracy [0.631 0.661]
            
        # 1fc with relu
        # by stage accuracy [[0.50540958 0.69034091        nan]
        # [0.43566176 0.76703297        nan]]
        # vf accuracy [0.571  0.629]
            
        # 2fc with relu
        # by stage accuracy [[0.35394127 0.79261364        nan]
        # [0.40073529 0.82197802        nan]]
        # vf accuracy [0.508 0.592]

        # 1conv pool 1conv 1fc
        # by stage accuracy [[0.41731066 0.76420455        nan]
        # [0.43933824 0.83296703        nan]]
        # vf accuracy [0.539 0.618]

        # 3conv 1fc: bad

        elif LARGE_NETWORK:
            # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
            # self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            # self.fc1 = nn.Linear(32 * 9 * 9, 128)
            # self.fc2 = nn.Linear(128, 1)

            # by stage accuracy [[0.41731066 0.76420455        nan]
            # [0.50367647 0.77142857        nan]]
            # vf accuracy [0.539 0.625]

            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            # Last pooling might be adjusted based on whether you apply it after the last conv layer
            self.fc1 = nn.Linear(128 * 1 * 1, 256)  # Adjusted to the correct number of input features
            self.fc2 = nn.Linear(256, 1) 
            
            # by stage accuracy [[0.65378671 0.91761364        nan]
            # [0.73713235 0.91428571        nan]]
            # vf accuracy [0.746 0.817]

            # TEST with clip
            # by stage accuracy [[0.51004637 0.78409091        nan]
            #  [0.51286765 0.77142857        nan]]
            # vf accuracy [0.606 0.63 ]

            # Train with clip
            # by stage accuracy [[0.5007728  0.82102273        nan]
            #  [0.51286765 0.82857143        nan]]
            # vf accuracy [0.613 0.656]
        else:
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(8 * 4 * 4, 1)
            #self.fc2 = nn.Linear(64, 1)

            # 2fc
            # by stage accuracy [[0.38948995 0.69034091        nan]
            #  [0.45772059 0.74065934        nan]]
            # vf accuracy [0.495 0.586]

            # 1fc
            # by stage accuracy [[0.63369397 0.90340909        nan]
            # [0.70955882 0.91868132        nan]]
            # vf accuracy [0.728 0.804]

            # TEST with CLip
            # by stage accuracy [[0.48840804 0.83806818        nan]
            #  [0.62132353 0.85714286        nan]]
            # vf accuracy [0.611 0.728]

    def forward(self, x):
        if ORI_NETWORK:
            #print(x.shape)
            x = torch.relu(self.conv1(self.pad(x)))
            # #print(x.shape)
            x = self.pool(x)
            # #print(x.shape)
            x = torch.relu(self.conv2(self.pad(x)))
            
            x = torch.relu(self.conv3(self.pad(x)))
            x = self.pool(x)
            x = torch.relu(self.conv4(self.pad(x)))
            #print(x.shape)
            #x = self.conv3(x)

            #print(x.shape)
            x = x.reshape(-1, 128*4*9)
            #x = x.reshape(-1, 9*9*3)
            x = torch.relu(self.fc0(x))
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = torch.relu(self.fc5(x))
            #x = torch.tanh(torch.relu(self.fc3(x)))
            #x = 2*torch.sigmoid(torch.relu(self.fc3(x))) - 1
            #x = torch.clamp(torch.relu(self.fc3(x)), min=-1, max=1)

        elif LARGE_NETWORK:
            # x = torch.relu(self.conv1(x))
            # x = torch.relu(self.conv2(x))
            # x = x.reshape(-1,32 * 9 * 9)
            # x = torch.relu(self.fc1(x))
            # x = self.fc2(x)

            x = F.leaky_relu(self.bn1(self.pool(self.conv1(x))))
            x = F.leaky_relu(self.bn2(self.pool(self.conv2(x))))
            x = F.leaky_relu(self.bn3(self.pool(self.conv3(x))))  # Consider if pooling is applicable here
            x = x.reshape(-1, 128*1*1)  # Flatten the tensor for the fully connected layer
            x = F.leaky_relu(self.fc1(x))
            x = self.fc2(x)

        else:
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = x.reshape(-1, 8 * 4 * 4)  # Adjust for the output size after pooling
            x = self.fc1(x)
            #x = self.fc2(x)

        return x

# Custom Loss Function
def custom_loss(r_a, r_b):
    return -nn.functional.logsigmoid(r_a - r_b) #+ 0.001 * 0.5 * (r_a**2 + r_b**2)
    #return 

# Training Function
def train(model, device, data_loader, optimizer, writer, epochs=10):
    model.train()
    global_step = 0

    valid_data = ValidDataset(size=VALI_DATA_SIZE)
    valid_loader = DataLoader(valid_data, batch_size=VALI_DATA_SIZE, shuffle=False)
    
    for epoch in range(epochs):
        total_loss = 0

        for a, b in data_loader:
            a, b = a.to(device), b.to(device)
            optimizer.zero_grad()
            r_a = model(a.permute(0, 3, 1, 2))
            r_b = model(b.permute(0, 3, 1, 2))
            loss = custom_loss(r_a, r_b)
            loss.mean().backward()
            optimizer.step()
            total_loss += loss.sum().item()

            # Log loss to TensorBoard
            if writer != None:
                writer.add_scalar('Training Loss', round(loss.mean().item(),2), global_step)
            global_step += 1

        scheduler.step()
        # Optionally print the current learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader.dataset)}. Current Learning Rate: {current_lr}")
        validate(model, valid_loader, device)
        print('-------------------------------------------')

    torch.save(model.state_dict(), os.getcwd() + '/LLM/RM/models/reward_model_pol' + '_corrected.pth')


# Define the lambda function for learning rate adjustment
def lr_lambda(epoch):
    if epoch < 20: #180
        return 0.00001 / INIT_LR #0.0001# Keep the initial learning rate 000133# relu-1fc ori #00004 relu-2fc ori
    elif LARGE_NETWORK:#100
        if 20 <= epoch < 250:
            return 0.0008 / INIT_LR
        else:
            return 0.0001 / INIT_LR  
    elif not LARGE_NETWORK:
        if 20 <= epoch < 230:
            return 0.00004 / INIT_LR # 00004 relu-1fc ori 00001 relu-2fc ori #ELSE 0.001 LARGE-SPECIAL 0.0006
        else:
            return 0.000002 / INIT_LR  #0005# 00002 2fc ori 00001 relu-2fc ori
        # if 100 <= epoch < 200: #!
        #     return 0.0008 / INIT_LR 
        # else:
        #     return 0.00000001 / INIT_LR  #!

def validate(model, valid_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    training_ranking_correct_rate = 0
    # No gradient updates needed for validation, so wrap in no_grad to save memory
    with torch.no_grad():
        for a, b in valid_loader:
            a, b = a.to(device), b.to(device) 

            r_a = model(a.permute(0, 3, 1, 2))
            r_b = model(b.permute(0, 3, 1, 2))

            loss = custom_loss(r_a, r_b)
            total_loss += loss.sum().item()

            training_ranking_correct_rate += len(np.where(r_a.cpu().numpy() > r_b.cpu().numpy())[0])

    # Calculate average loss over all samples
    avg_loss = total_loss / len(valid_loader.dataset)
    training_ranking_correct_rate /= len(valid_loader.dataset)
    #if training_ranking_correct_rate == 0.0:
    # for r in range(len(r_a)):
    #     print(r_a[r], r_b[r])
    
    print(f'Validation Loss: {avg_loss}')
    print(f'Validation Ranking Correctness: {training_ranking_correct_rate}')

# Main
if __name__ == "__main__":
    args = parser.parse_args()
    global AGENT, DATASET_SIZE, VALI_DATA_SIZE, FILES
    #DATASET_SIZE = [[277, 299], [8365, 8295]][LARGE_NETWORK][AGENT]
    #DATASET_SIZE = [3033, 3034][AGENT]
    #DATASET_SIZE = [1541, 1541][AGENT]
    DATASET_SIZE = 3210

    VALI_DATA_SIZE = 600#int(DATASET_SIZE / 5)
    DATASET_SIZE = DATASET_SIZE - VALI_DATA_SIZE

    if USE_TRUE_DATA:
        GT_FILE = "ranking_data100k.pkl"
    elif USE_STARLING:
        FILES = [os.getcwd()+'/data/data/starling_states_high' + '.csv', os.getcwd()+'/data/data/starling_states_low' + '.csv'] #48.5% 55.3%
        # by stage accuracy [[0.49149923 0.47443182        nan]
        #  [0.51654412 0.5978022         nan]]
        # vf accuracy [0.485 0.553]
    elif USE_GPT4:
        FILES = [os.getcwd()+'/data/data/old-gpt4_states_high' + '.csv', os.getcwd()+'/data/data/old-gpt4_states_low' + '.csv'] 
        # by stage accuracy [[0.51004637 0.41477273        nan]
        # [0.48713235 0.57362637        nan]]
        # vf accuracy [0.476 0.526]

        # clip
        # by stage accuracy [[0.51622875 0.53409091        nan]
        # [0.47610294 0.56703297        nan]]
        # vf accuracy [0.522 0.517]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    model = RewardModel().to(device)

    if args.mode == 'train':
        optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
        dataset = RewardDataset(size=DATASET_SIZE)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Initialize SummaryWriter for TensorBoard
        writer = SummaryWriter()
        # Initialize the learning rate scheduler
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        scheduler = LambdaLR(optimizer, lr_lambda)

        train(model, device, data_loader, optimizer, writer, epochs=50)

        # Close the SummaryWriter to flush the information to disk
        writer.close()

    else:
        model.load_state_dict(torch.load(args.model_path, map_location=device))

        valid_data = ValidDataset(size=VALI_DATA_SIZE)
        valid_loader = DataLoader(valid_data, batch_size=VALI_DATA_SIZE, shuffle=False)

        validate(model, valid_loader, device)