from typing import Callable, Optional

import pickle
from ffrecord import FileReader
from hfai.datasets.base import (
    BaseDataset,
    get_data_dir,
)


"""
Expected file organization:

    [data_dir]
        train.ffr
            meta.pkl
            PART_00000.ffr
            PART_00001.ffr
            ...
        val.ffr
            meta.pkl
            PART_00000.ffr
            PART_00001.ffr
            ...
"""

class AlphafoldData(BaseDataset):
    def __init__(
        self,
        transform: Optional[Callable] =None,
        check_data: bool = True,
    ) -> None:
        super(AlphafoldData, self).__init__()

        data_dir = get_data_dir()
        ffr_file = data_dir / "Alphafold" / "train" / "ffrdata"
        self.reader = FileReader(ffr_file, check_data=check_data)
        self.transform = transform

    def __len__(self):
        return self.reader.n

    def __getitem__(self, indices):
        data = self.reader.read(indices)
        samples = []
        for bytes_ in data:
            mmcifdata = pickle.loads(bytes_)
            if self.transform:
                mmcifdata = self.transform(*mmcifdata)
            samples.append(mmcifdata)
        return samples
    
