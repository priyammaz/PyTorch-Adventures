import torch
from accelerate import Accelerator

accelerator = Accelerator()
rand = torch.randn(2,).to(accelerator.device)



# ### GATHER TEST ###
# print(accelerator.device, rand)
# # Put gathered tensors on the same device as the input
# gathered_rands = [torch.zeros(2,).to(accelerator.device) for _ in range(accelerator.num_processes)]
# torch.distributed.all_gather(gathered_rands, rand)

# if accelerator.is_main_process:
#     print(gathered_rands)

torch.cuda.set_device(accelerator.device)


### Copy all buffers from main GPU to others 
accelerator.print("BEFORE")
print(accelerator.device, rand)
accelerator.wait_for_everyone()

accelerator.print("AFTER")
torch.distributed.broadcast(rand.data, src=0)
print(accelerator.device, rand)

accelerator.end_training()