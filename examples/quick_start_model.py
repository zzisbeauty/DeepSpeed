import deepspeed

# cmd_args = {
#     'model_name_or_path':'/home/fzm/Desktop/DeepSpeed/models-files/facebook/opt-350m',
#     'train_batch_size':2,
#     'gradient_accumulation_steps':8,
#     'deepspeed':True,
#     'zero_stage':0,
#     'deepspeed_config':',',
#     'output_dir':'/home/fzm/Desktop/DeepSpeed/output'
# }

# import json

# with open('/home/fzm/Desktop/DeepSpeed/models-files/facebook/opt-350m/config.json','r',encoding='utf-8') as f:
#     params = json.load(f)

# model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
#                                                      model='/home/fzm/Desktop/DeepSpeed/models-files/facebook/opt-350m',
#                                                      model_parameters=params)

import torch


import torch.utils.data as tud
class MyDataset(tud.Dataset):
    def __init__(self, image_shape:tuple, num_category:int, sample_count:int=1000):
        super().__init__()
        self.image_shape = image_shape
        self.num_category = num_category
        self.sample_count = sample_count
    
    def __len__(self):
        return self.sample_count
    
    def __getitem__(self, index:int):
        xx = torch.randn(self.image_shape, dtype=torch.float32)
        yy = torch.randint(low=0, high=self.num_category, size=[1])
        return xx, yy
        return super().__getitem__(index)

import torch.nn as nn
class ConvBlock(nn.Module):
    def __init__(self, num_channels: int, layer_scale_init: float = 1e-6):
        super().__init__()
        self.residual = nn.Sequential(
            nn.GroupNorm(1, num_channels),
            nn.GELU(),
            nn.Conv2d(num_channels,num_channels,kernel_size=3,stride=1,padding=1),
            nn.GroupNorm(1, num_channels),
            nn.GELU(),
            nn.Conv2d(num_channels,num_channels,kernel_size=3,stride=1,padding=1)
        )
        self.layer_scale = nn.Parameter(torch.Tensor(layer_scale_init))


    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        h = input + self.layer_scale * self.residual(inputs)
        return h




class MyClassifier(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, ch_multi: int = 32):
        super().__init__()

        self.stage1 = nn.Sequential(
            # downscale
            nn.Sequential(
                nn.Conv2d(in_channels, ch_multi, kernel_size=2, stride=2, padding=0),
                nn.GroupNorm(1, ch_multi),  # LayerNorm
            ),
            ConvBlock(ch_multi),
            ConvBlock(ch_multi),
        )

        self.stage2 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(ch_multi, 2 * ch_multi, kernel_size=2, stride=2, padding=0),
                nn.GroupNorm(1, 2 * ch_multi),  # LayerNorm
            ),
            ConvBlock(2 * ch_multi),
            ConvBlock(2 * ch_multi),
        )

        self.stage3 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(2 * ch_multi, 4 * ch_multi, kernel_size=2, stride=2, padding=0),
                nn.GroupNorm(1, 4 * ch_multi),  # LayerNorm
            ),
            ConvBlock(4 * ch_multi),
            ConvBlock(4 * ch_multi),
        )

        self.stage4 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4 * ch_multi, 8 * ch_multi, kernel_size=2, stride=2, padding=0),
                nn.GroupNorm(1, 8 * ch_multi),  # LayerNorm
            ),
            ConvBlock(8 * ch_multi),
            ConvBlock(8 * ch_multi),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(8 * ch_multi, out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.stage1(inputs)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        h = self.classifier(h)
        return h


import torch.nn.functional as tnf
import logging
import numpy as np

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='deepspeed training script.')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def train():
    args = parse_arguments()

    # init distributed
    deepspeed.init_distributed()

    # init model
    model = MyClassifier(3, 100, ch_multi=128)

    # init dataset
    ds = MyDataset((3, 512, 512), 100, sample_count=int(1e6))

    # init engine
    engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=ds,
        # config=deepspeed_config,
    )

    # load checkpoint
    engine.load_checkpoint("./data/checkpoints/MyClassifier/")

    # train
    import time
    last_time = time.time()
    loss_list = []
    echo_interval = 10

    engine.train()
    for step, (xx, yy) in enumerate(training_dataloader):
        step += 1
        xx = xx.to(device=engine.device, dtype=torch.float16)
        yy = yy.to(device=engine.device, dtype=torch.long).reshape(-1)

        outputs = engine(xx)
        loss = tnf.cross_entropy(outputs, yy)
        engine.backward(loss)
        engine.step()
        loss_list.append(loss.detach().cpu().numpy())

        if step % echo_interval == 0:
            loss_avg = np.mean(loss_list[-echo_interval:])
            used_time = time.time() - last_time
            time_p_step = used_time / echo_interval
            if args.local_rank == 0:
                logging.info(
                    "[Train Step] Step:{:10d}  Loss:{:8.4f} | Time/Batch: {:6.4f}s",
                    step, loss_avg, time_p_step,
                )
            last_time = time.time()
    # save checkpoint
    engine.save_checkpoint("/home/fzm/Desktop/DeepSpeed/output/quick_start_cls")


def main():
    train()

if __name__ == '__main__':
    main()




# deepspeed.init_distributed()
# model = MyClassifier(3,100, ch_multi=512)
# datasets = MyDataset((3,512,512),100,sample_count=100)
# engine, optimizer, traing_dataloader, lr_scheduler = deepspeed.initialize(
#     args='',
#     model=model,
#     model_parameters=model.parameters(),
#     training_data=datasets,
#     config=''
# )