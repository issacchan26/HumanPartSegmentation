import numpy as np
import torch
from torch_geometric.loader import DataLoader
from model import BodySeg
from dataset import HumanSeg
import pandas as pd

test_dataset_path = '/path to/test_data'  #  where to put the 'raw' folder that contains test .ply / .txt files
model_path = '/path to/checkpoints/best.pt'  # the path of trained model
ply_path = '/path to/ply_data/input/'  # where to put all original .ply files
output_ply_path = '/path to/ply_data/output/'  # where to save the .ply with color
body_part = 4  # 4/6/14/28 body parts
acc_threshold = 0.76  # Apply when validation with label
with_label = False


def test(loader):
    pred_list = []
    model.eval()
    correct_nodes = total_nodes = flag = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.pos, data.batch)
        pred_df = get_pred_df(data.pos, out.argmax(dim=1))
        pred_list.append(pred_df)

        if data.y is not None:
            flag = 1
            correct_nodes += out.argmax(dim=1).eq(data.y.squeeze(1)).sum().item()
            total_nodes += data.num_nodes
    
    if flag == 1:
        eval_acc = correct_nodes / total_nodes
        print(f'Test Acc: {eval_acc:.4f}')
        return eval_acc, pred_list
    else:
        return pred_list

def get_pred_df(pos, pred):
    df = pd.DataFrame(pos.cpu().numpy(), columns=['x','y','z'])
    df['pred'] = pred.cpu().numpy()
    df['r'] = ''
    df['g'] = ''
    df['b'] = ''

    return df

color_array = [
        [0, 0, 0],        # rest of body
        [255, 255, 51],   # head
        [102, 102, 0],    # neck
        [255, 0, 0],      # right_shoulder
        [255, 153, 153],  # left_shoulder
        [255, 128, 0],    # right_upper_arm
        [255, 204, 153],  # left_upper_arm
        [204, 0, 102],    # right_elbow
        [255, 102, 178],  # left_elbow
        [128, 255, 0],    # right_fore_arm
        [178, 255, 102],  # left_fore_arm
        [204, 0, 204],    # right_wrist
        [255, 102, 255],  # left_wrist
        [0, 204, 0],      # right_hand
        [102, 255, 102],  # left_hand
        [255, 255, 255],  # main_body
        [102, 0, 204],    # right_hip
        [204, 153, 255],  # left_hip
        [0, 204, 102],    # right_thigh
        [102, 255, 178],  # left_thigh
        [0, 0, 204],      # right_knee
        [102, 102, 255],  # left_knee
        [0, 204, 204],    # right_leg
        [153, 255, 255],  # left_leg
        [0, 102, 204],    # right_ankle
        [153, 204, 255],  # left_ankle
        [244, 244, 244],  # right_foot
        [128, 128, 128]   # left_foot
        ]

def df_to_color_ply(file_index, df):

    filename_list = sorted(test_dataset.raw_file_names)
    test_filename = filename_list[file_index].split('.')[0]
    ply_file = ply_path + test_filename + '.ply'
    output_file_path = output_ply_path + test_filename + '.ply'
    ply_df = pd.read_csv(ply_file, sep=' ', names=['x', 'y', 'z', 'r', 'g', 'b'])
    ply_xyz = pd.read_csv(ply_file, sep=' ', names=['x', 'y', 'z', 'r', 'g', 'b'], skiprows=10, nrows=10000)

    ply_xyz['xyz'] = ply_xyz['x'].map('{:,.6f}'.format).astype(str) + ply_xyz['y'].map('{:,.6f}'.format).astype(str) + ply_xyz['z'].map('{:,.6f}'.format).astype(str)
    ply_xyz.set_index('xyz', inplace=True)

    df['y'] = df['y'] * -1
    df = df.round(6)
    df['xyz'] = df['x'].map('{:,.6f}'.format) + df['y'].map('{:,.6f}'.format) + df['z'].map('{:,.6f}'.format)
    df.set_index('xyz', inplace=True)
    df = df.reindex(ply_xyz.index)
    df = df.reset_index()
    df = df.drop('xyz', axis=1)
    
    for j in range(0, len(color_array)):
        df.loc[df['pred'] == j, ['r', 'g', 'b']] = color_array[j]
    df = df.drop(['pred'], axis=1)
    df[['r', 'g', 'b']] = df[['r', 'g', 'b']].astype('Int64')
    df.index += 10
    ply_df.update(df)
    ply_df.loc[6.5] = ['property', 'uchar', 'red', np.nan, np.nan, np.nan]
    ply_df.loc[6.6] = ['property', 'uchar', 'green', np.nan, np.nan, np.nan]
    ply_df.loc[6.7] = ['property', 'uchar', 'blue', np.nan, np.nan, np.nan]
    ply_df = ply_df.sort_index().reset_index(drop=True)
    ply_df.dropna()
    ply_df.to_csv(output_file_path,  sep=' ', index=False, header=False, lineterminator='\n')

def ply_to_color_ply(file_index, df):

    filename_list = sorted(test_dataset.raw_file_names)
    test_filename = filename_list[file_index].split('.')[0]
    ply_file = ply_path + test_filename + '.ply'
    output_file_path = output_ply_path + test_filename + '.ply'
    ply_df = pd.read_csv(ply_file, sep=' ', names=['x', 'y', 'z', 'r', 'g', 'b'])
    
    df['y'] = df['y'] * -1
    for j in range(0, len(color_array)):
        df.loc[df['pred'] == j, ['r', 'g', 'b']] = color_array[j]
    df = df.drop(['pred'], axis=1)
    df[['r', 'g', 'b']] = df[['r', 'g', 'b']].astype('Int64')
    df.index += 10
    ply_df.update(df)
    ply_df.loc[6.5] = ['property', 'uchar', 'red', np.nan, np.nan, np.nan]
    ply_df.loc[6.6] = ['property', 'uchar', 'green', np.nan, np.nan, np.nan]
    ply_df.loc[6.7] = ['property', 'uchar', 'blue', np.nan, np.nan, np.nan]
    ply_df = ply_df.sort_index().reset_index(drop=True)
    ply_df.dropna()
    ply_df.to_csv(output_file_path,  sep=' ', index=False, header=False, lineterminator='\n')


def predict(with_label: bool = False):

    if with_label == True:
        eval_acc = 0
        while eval_acc < acc_threshold:
            eval_acc, pred_list = test(test_loader)
        for i, pred in enumerate(pred_list):
            df_to_color_ply(i, pred)
    else:
        pred_list = test(test_loader)
        for i, pred in enumerate(pred_list):
            ply_to_color_ply(i, pred)
    
    return

if __name__ == '__main__':

    test_dataset = dataset = HumanSeg(test_dataset_path, include_normals=False, body_part=body_part, with_label=with_label)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BodySeg(3, test_dataset.num_classes, dim_model=[32, 64, 128, 256, 512], k=16).to(device)
    model.load_state_dict(torch.load(model_path))
    predict(with_label=with_label) 
        

