import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import json
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %10 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()
    
    # Save performance metrics to JSON file
    Procedure.save_metrics()
    with open(metrics_filename, 'w') as f:
        json.dump(Procedure.EPOCH_METRICS, f, indent=2)
    print(f"\nPerformance metrics saved to {metrics_filename}")
    
    # Print summary statistics
    if Procedure.EPOCH_METRICS['epoch_times']:
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Mode: {'QUANTIZED (8-bit)' if world.config.get('quantization', False) else 'ORIGINAL (Float32)'}")
        print(f"Average Epoch Time: {np.mean(Procedure.EPOCH_METRICS['epoch_times']):.3f}s")
        print(f"Average Sample Time: {np.mean(Procedure.EPOCH_METRICS['sample_times']):.3f}s")
        print(f"Average Batch Time: {np.mean(Procedure.EPOCH_METRICS['batch_times']):.4f}s")
        print(f"Average Memory Usage: {np.mean(Procedure.EPOCH_METRICS['memory_usage']):.1f} MB")
        print(f"Average CPU Usage: {np.mean(Procedure.EPOCH_METRICS['cpu_usage']):.1f}%")
        print("="*50)
