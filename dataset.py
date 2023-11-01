# Reference to pyg shapenet

import os
from typing import Callable, List, Optional, Union
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset


class HumanSeg(InMemoryDataset):
    
    category_ids = {
        'Human': '02691156',
    }

    seg_classes = {
        'Human': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
    }

    def __init__(
        self,
        root: str,
        categories: Optional[Union[str, List[str]]] = None,
        include_normals: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        body_part: int = None,
        with_label: bool = True,
    ):
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)
        self.categories = categories
        self.body_part = body_part
        self.with_label = with_label
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data.x = self.data.x if include_normals else None
        self.y_mask = torch.zeros((len(self.seg_classes.keys()), len(list(self.seg_classes.values())[0])), dtype=torch.bool)
        for i, labels in enumerate(self.seg_classes.values()):
            self.y_mask[i, labels] = 1
            
        self.raw_filename_list = []

    @property
    def num_classes(self) -> int:
        return self.y_mask.size(-1)

    @property
    def raw_file_names(self):
        self.raw_filename_list = os.listdir(self.raw_dir)
        return self.raw_filename_list

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def simply_label(self, df, body_part):
        if body_part == 14:
            df['label'] = df['label'].replace([
                0,   # rest of body
                1,   # head
                2,   # neck
                3,   # right_shoulder
                4,   # left_shoulder
                5,   # right_upper_arm
                6,   # left_upper_arm
                7,   # right_elbow
                8,   # left_elbow
                9,   # right_fore_arm
                10,  # left_fore_arm
                11,  # right_wrist
                12,  # left_wrist
                13,  # right_hand
                14,  # left_hand
                15,  # main_body
                16,  # right_hip
                17,  # left_hip
                18,  # right_thigh
                19,  # left_thigh
                20,  # right_knee
                21,  # left_knee
                22,  # right_leg
                23,  # left_leg
                24,  # right_ankle
                25,  # left_ankle
                26,  # right_foot
                27   # left_foot
                ],[
                0,   # rest of body
                1,   # head
                1,   # neck
                2,   # right_shoulder
                2,   # left_shoulder
                3,   # right_upper_arm
                3,   # left_upper_arm
                4,   # right_elbow
                4,   # left_elbow
                5,   # right_fore_arm
                5,  # left_fore_arm
                6,  # right_wrist
                6,  # left_wrist
                7,  # right_hand
                7,  # left_hand
                0,  # main_body
                8,  # right_hip
                8,  # left_hip
                9,  # right_thigh
                9,  # left_thigh
                10,  # right_knee
                10,  # left_knee
                11,  # right_leg
                11,  # left_leg
                12,  # right_ankle
                12,  # left_ankle
                13,  # right_foot
                13   # left_foot
                ])
            
        elif body_part == 6:
            # {0: main body, 1: head, 2: right arm, 3: left arm, 4: right leg, 5: left leg}
            df['label'] = df['label'].replace([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27], [0,1,1,2,3,2,3,2,3,2,3,2,3,2,3,0,4,5,4,5,4,5,4,5,4,5,4,5])

        elif body_part == 4:
            # {0: main body, 1: head, 2: arm, 3: leg}
            df['label'] = df['label'].replace([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27], [0,1,1,2,2,2,2,2,2,2,2,2,2,2,2,0,3,3,3,3,3,3,3,3,3,3,3,3])
        
        else:
            df['label'] = df['label']
            
        return df

    def process(self):
        data_list = []
        for raw_file_path in sorted(self.raw_paths):          
            if self.with_label == True:
                df = pd.read_csv(raw_file_path, skiprows=10, names=['x','y','z','label','type'], sep=' ')
                pos = df[['x','y','z']]
                pos = torch.from_numpy(pos.values).to(torch.float)
                if self.body_part is not None:
                    df = self.simply_label(df, self.body_part)

                y = df[['label']]
                y = torch.from_numpy(y.values).type(torch.LongTensor)
                data = Data(pos=pos, y=y)

            else:
                df = pd.read_csv(raw_file_path, skiprows=10, nrows=10000, names=['x','y','z'], sep=' ')
                df['y'] = df['y'] * -1
                pos = df[['x','y','z']]
                pos = torch.from_numpy(pos.values).to(torch.float)
                data = Data(pos=pos)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'categories={self.categories})')

