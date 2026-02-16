import torch
from modules.encodec import EncodecModel

model = EncodecModel()
state_dict = torch.load("work_dir/encodec/checkpoint_0/pytorch_model.bin")
success = model.load_state_dict(state_dict)
print(success)