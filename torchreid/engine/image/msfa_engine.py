import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from torchreid.engine import ImageTripletEngine
from torchreid.losses import CrossEntropyLoss, TripletLoss


def _normalize_map(heatmap):
    """Normalizes 2D float array to [0, 255] uint8."""
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-12)
    return np.uint8(255 * heatmap)


def _overlay_heatmap(img_tensor, map_2d):
    """Overlays a 2D map onto the normalized original RGB image."""
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    img_uint8 = np.uint8(255 * img)
    h, w, _ = img_uint8.shape

    map_resized = cv2.resize(map_2d, (w, h))
    colormap = cv2.applyColorMap(map_resized, cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(0.4 * img_uint8 + 0.6 * colormap)
    return img_uint8, colormap, overlay


class MSFAEngine(ImageTripletEngine):
    def __init__(self, datamanager, model, optimizer, margin=0.3,
                 weight_t=0.5, weight_x=1.0, scheduler=None, use_gpu=True,
                 vis_samples=5):
        super(MSFAEngine, self).__init__(
            datamanager, model, optimizer, margin=margin,
            weight_t=weight_t, weight_x=weight_x,
            scheduler=scheduler, use_gpu=use_gpu
        )
        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=False
        )
        self.weight_t = weight_t   # 0.5 per paper
        self.weight_x = weight_x   # 1.0 per paper
        self.vis_samples = vis_samples

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)
        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        outputs, features = self.model(imgs)
        loss_t = self.compute_loss(self.criterion_t, features, pids)
        loss_x = self.compute_loss(self.criterion_x, outputs, pids)
        loss = self.weight_x * loss_x + self.weight_t * loss_t

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'loss_x': loss_x.item(),
            'loss_t': loss_t.item()
        }

    def test(self, epoch, ranks=[1, 5, 10, 20], **kwargs):
        """
        Runs standard evaluation (Rank-1, 5, 10, 20, mAP) and extracts 
        activation & attention maps every time the engine evaluates.
        """
        # 1. Run standard gallery retrieval evaluation
        rank1 = super(MSFAEngine, self).test(epoch=epoch, ranks=ranks, **kwargs)

        # 2. Generate and save maps
        self.visualize_maps(epoch=epoch)
        return rank1

    def visualize_maps(self, epoch):
        save_dir = os.path.join(self.save_dir, 'vis_maps')
        os.makedirs(save_dir, exist_ok=True)
        self.model.eval()

        # Fetch a test batch from the query dataloader
        target_name = self.datamanager.targets[0]
        query_loader = self.datamanager.test_loader[target_name]['query']

        with torch.no_grad():
            for batch in query_loader:
                imgs = batch['img']
                break

            imgs_cuda = imgs.cuda() if self.use_gpu else imgs

            # 1. Activation Map: Sum across channels of final feature map
            featmaps = self.model(imgs_cuda, return_featuremaps=True)
            actmaps = torch.sum(featmaps, dim=1).cpu().numpy()

            # 2. Attention Map: Feature-gate weighting from stage 3 MSFA block
            target_msfa = self.model.stage3[-1]
            f = target_msfa.prep(featmaps if featmaps.size(1) == 512 else self.model.stage2(self.model.stage1(self.model.stem(imgs_cuda))))
            g1 = target_msfa.ag1(f)
            g2 = target_msfa.ag2(f)
            g3 = target_msfa.ag3(f)
            fused_attention = (g1 + g2 + g3) * f
            attnmaps = torch.mean(fused_attention, dim=1).cpu().numpy()

        for i in range(min(self.vis_samples, imgs.size(0))):
            norm_act = _normalize_map(actmaps[i])
            norm_attn = _normalize_map(attnmaps[i])

            orig, act_color, act_overlay = _overlay_heatmap(imgs[i], norm_act)
            _, attn_color, attn_overlay = _overlay_heatmap(imgs[i], norm_attn)

            # Horizontal strip: [Original | Act Heatmap | Act Overlay | Attn Heatmap | Attn Overlay]
            row = np.hstack([orig, act_color, act_overlay, attn_color, attn_overlay])

            out_path = os.path.join(save_dir, f'epoch_{epoch:03d}_sample_{i+1}.png')
            cv2.imwrite(out_path, cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

        print(f"--> [Epoch {epoch:03d}] Visual maps saved to: {save_dir}/")