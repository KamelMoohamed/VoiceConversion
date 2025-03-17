import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from dataloaders import (
    DistributedBucketSampler,
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from losses import discriminator_loss, generator_loss
from models import MultiPeriodDiscriminator, SynthesizerTrn
from utils.constants import (
    DATA_FILTER_LENGTH,
    DATA_TRAINING_FILES,
    DATA_VALIDATION_FILES,
    MODEL_USE_SPECTRAL_NORM,
    TRAIN_BATCH_SIZE,
    TRAIN_BETAS,
    TRAIN_EPOCHS,
    TRAIN_EPS,
    TRAIN_EVAL_INTERVAL,
    TRAIN_FP16_RUN,
    TRAIN_LEARNING_RATE,
    TRAIN_LR_DECAY,
    TRAIN_PORT,
    TRAIN_SEGMENT_SIZE,
)

torch.backends.cudnn.benchmark = True


def main():
    assert torch.cuda.is_available(), "CPU training is not allowed."
    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = TRAIN_PORT

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus,))


def run(rank, n_gpus):
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(1234)
    torch.cuda.set_device(rank)

    train_dataset = TextAudioSpeakerLoader(DATA_TRAINING_FILES)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        TRAIN_BATCH_SIZE,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        pin_memory=True,
        batch_sampler=train_sampler,
        collate_fn=TextAudioSpeakerCollate(),
    )

    eval_loader = None
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(DATA_VALIDATION_FILES)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=8,
            batch_size=TRAIN_BATCH_SIZE,
            pin_memory=False,
            collate_fn=TextAudioSpeakerCollate(),
        )

    net_g = SynthesizerTrn(
        DATA_FILTER_LENGTH // 2 + 1, TRAIN_SEGMENT_SIZE // DATA_FILTER_LENGTH
    ).cuda(rank)
    net_d = MultiPeriodDiscriminator(MODEL_USE_SPECTRAL_NORM).cuda(rank)

    optim_g = torch.optim.AdamW(
        net_g.parameters(), TRAIN_LEARNING_RATE, betas=TRAIN_BETAS, eps=TRAIN_EPS
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(), TRAIN_LEARNING_RATE, betas=TRAIN_BETAS, eps=TRAIN_EPS
    )

    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])

    scaler = GradScaler(enabled=TRAIN_FP16_RUN)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=TRAIN_LR_DECAY)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=TRAIN_LR_DECAY)

    for epoch in range(1, TRAIN_EPOCHS + 1):
        train(rank, net_g, net_d, optim_g, optim_d, scaler, train_loader)
        if rank == 0 and epoch % TRAIN_EVAL_INTERVAL == 0:
            evaluate(net_g, eval_loader)
        scheduler_g.step()
        scheduler_d.step()


def train(rank, net_g, net_d, optim_g, optim_d, scaler, train_loader):
    net_g.train()
    net_d.train()

    for batch_idx, (c, spec, y, filenames) in enumerate(train_loader):
        spec, y, c = (
            spec.cuda(rank, non_blocking=True),
            y.cuda(rank, non_blocking=True),
            c.cuda(rank, non_blocking=True),
        )

        with autocast(enabled=TRAIN_FP16_RUN):
            y_hat, ids_slice, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                c, spec, filenames
            )
            loss_gen, _ = generator_loss(net_d(y, y_hat)[1])
            loss_disc, _, _ = discriminator_loss(*net_d(y, y_hat.detach()))

        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.step(optim_d)

        optim_g.zero_grad()
        scaler.scale(loss_gen).backward()
        scaler.step(optim_g)
        scaler.update()


def evaluate(generator, eval_loader):
    generator.eval()
    with torch.no_grad():
        for c, spec, y, filenames in eval_loader:
            c, spec = c.cuda(0, non_blocking=True), spec.cuda(0, non_blocking=True)
    generator.train()


if __name__ == "__main__":
    main()
