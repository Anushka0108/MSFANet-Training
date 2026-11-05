import os
import torch
import torchreid

from msfanet import build_msfanet
from msfa_engine import MSFAEngine


def main():
    save_dir = 'log/msfanet_market1501'
    os.makedirs(save_dir, exist_ok=True)

    # 1. DataManager
    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources='market1501',
        targets='market1501',
        height=256,
        width=128,
        batch_size_train=128,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop', 'random_erase']
    )

    # 2. Build MSFANet
    model = torchreid.models.build_model(
        name='msfanet',
        num_classes=datamanager.num_train_pids,
        loss='triplet'
    ).cuda()

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.065,
        weight_decay=5e-4,
        amsgrad=True
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[40, 70, 90],
        gamma=0.1
    )

    # 4. Engine with integrated visualizer
    engine = MSFAEngine(
        datamanager=datamanager,
        model=model,
        optimizer=optimizer,
        margin=0.3,
        weight_t=0.5,
        weight_x=1.0,
        scheduler=scheduler,
        use_gpu=True,
        vis_samples=5
    )

    # 5. Run full 100 epochs
    engine.run(
        save_dir=save_dir,
        max_epoch=100,
        eval_freq=10,            # Triggers rank evaluation & map visualization every 10 epochs
        ranks=[1, 5, 10, 20],    # Computes Rank-1, 5, 10, 20 and mAP
        print_freq=20
    )


if __name__ == '__main__':
    main()