# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import argparse
import json
import os
import torch
import wandb
import math
import time

#=====START: ADDED FOR DISTRIBUTED======
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
#=====END:   ADDED FOR DISTRIBUTED======

from torch.utils.data import DataLoader
from shutil import copyfile
from glow import WaveGlow, WaveGlowLoss
from mel2samp import Mel2Samp

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration

def load_pretrained(pretrained_path, model):
    assert os.path.isfile(pretrained_path)
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    model_for_loading = pretrained_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded pretrained weights '{}'" .format(
          pretrained_path))
    return model

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    model_for_saving = WaveGlow(**waveglow_config).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_clip_grad_norm(grad_norm, grad_clip_thresh):
    clip_coef = grad_clip_thresh / (grad_norm + 1e-6)
    if clip_coef < grad_clip_thresh:
        clipped_grad_norm = grad_norm * clip_coef
    else:
        clipped_grad_norm = grad_norm

    return clipped_grad_norm

def train(num_gpus, rank, group_name, prj_name, run_name,
          output_directory, epochs, learning_rate,
          sigma, iters_per_checkpoint, batch_size, seed, fp16_run,
          grad_clip_thresh,
          checkpoint_path, pretrained_path,
          with_tensorboard, with_wandb):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
    #=====END:   ADDED FOR DISTRIBUTED======

    criterion = WaveGlowLoss(sigma)
    model = WaveGlow(**waveglow_config).cuda()

    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)
    #=====END:   ADDED FOR DISTRIBUTED======

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model,
                                                      optimizer)
        iteration += 1  # next iteration is iteration + 1

    if pretrained_path != "":
        model = load_pretrained(pretrained_path, model)

    trainset = Mel2Samp(**data_config)
    # =====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        train_sampler = DistributedSampler(trainset)
        shuffle_at_dataloader = False
    else:
        train_sampler = None
        shuffle_at_dataloader = True
    # =====END:   ADDED FOR DISTRIBUTED======
    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle_at_dataloader,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)

    # Get shared output_directory ready
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    if with_tensorboard and rank == 0:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(os.path.join(output_directory, 'logs'))

    model.train()
    epoch_offset = max(0, int(iteration / len(train_loader)))
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            iter_start = time.perf_counter()

            float_epoch = float(iteration) / len(train_loader)

            model.zero_grad()

            mel, audio = batch
            mel = torch.autograd.Variable(mel.cuda())
            audio = torch.autograd.Variable(audio.cuda())
            outputs = model((mel, audio))

            loss, etc = criterion(outputs)
            (z_L2_normalized, neg_log_s_total, neg_log_det_W_total) = etc
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()

            if fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            is_overflow = False
            if fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
                if not is_overflow:
                    clipped_grad_norm = get_clip_grad_norm(grad_norm, grad_clip_thresh)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_thresh)
                clipped_grad_norm = get_clip_grad_norm(grad_norm, grad_clip_thresh)

            optimizer.step()
            iter_duration = time.perf_counter() - iter_start

            print("{}:\t{:.9f}".format(iteration, reduced_loss))
            if with_tensorboard and rank == 0:
                logger.add_scalar('training_loss', reduced_loss, i + len(train_loader) * epoch)

            if with_wandb and rank == 0:
                wandb.log({
                    'iteration': iteration,
                    'epoch': float_epoch,
                    'iter_duration': iter_duration,
                    'training_loss': reduced_loss,
                    'training_loss/z_L2_normalized': z_L2_normalized,
                    'training_loss/neg_log_s_total': neg_log_s_total,
                    'training_loss/neg_log_det_W_total': neg_log_det_W_total,
                }, step=iteration)
                if not is_overflow:
                    wandb.log({
                        'grad_norm': grad_norm,
                        'clipped_grad_norm': clipped_grad_norm,
                    }, step=iteration)

            if (iteration % iters_per_checkpoint == 0):
                if rank == 0:
                    checkpoint_path = "{}/{}/{}/waveglow_{}".format(
                        output_directory,
                        prj_name, run_name,
                        iteration)
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prj_name', type=str,
                        help='give a project name for this running')
    parser.add_argument('--run_name', type=str,
                        help='give a distinct name for this running')
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    parser.add_argument('--visible_gpus', type=str, default="0",
                        required=False, help='CUDA visible GPUs')
    parser.add_argument('--resume', type=str, default="",
                        required=False, help='wandb running ID')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global dist_config
    dist_config = config["dist_config"]
    global waveglow_config
    waveglow_config = config["waveglow_config"]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    if args.rank == 0 and train_config["with_wandb"]:
        if args.resume == "":
            wandb.init(name=args.run_name, project=args.prj_name, resume=args.resume)
        else:
            wandb.init(project=args.prj_name, resume=args.resume)

    output_directory = train_config["output_directory"]
    create_dir('{}'.format(output_directory))
    create_dir('{}/{}'.format(output_directory, args.prj_name))
    create_dir('{}/{}/{}'.format(output_directory, args.prj_name, args.run_name))
    new_config_path = '{}/{}/{}/{}'.format(
        output_directory, args.prj_name, args.run_name, os.path.split(args.config)[1])
    if args.config != new_config_path:
        copyfile(args.config, new_config_path)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(num_gpus, args.rank, args.group_name,
        args.prj_name, args.run_name, **train_config)
