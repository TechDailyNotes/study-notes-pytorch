# PyTorch Distributed Pipeline

0. Trigger Function
   - `mp.spawn()`
1. Init Process
   1. Init System Params
      1. `MASTER_ADDR`
      2. `MASTER_PORT`
   2. `init_process_group()`
   3. `torch.cuda.set_device()`
2. Data Wrapper
   - `torch.utils.data.distributed.DistributedSampler`
3. Model Wrapper
   - `torch.nn.parallel.DistributedDataParallel`
4. Destroy Process
