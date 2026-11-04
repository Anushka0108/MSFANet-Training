import csv
import os
from pathlib import Path

import torch
import torch.nn.functional as F

import torchreid
from torchreid import metrics
from torchreid.engine import Engine

try:
    from .losses import batch_hard_triplet_loss
except ImportError:
    from losses import batch_hard_triplet_loss


class MSFANetEngine(Engine):
    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        weight_softmax=1.0,
        weight_triplet=0.5,
        triplet_margin=0.3,
        use_gpu=True,
        output_dir="log/msfanet"
    ):
        super().__init__(datamanager, use_gpu)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.weight_softmax = weight_softmax
        self.weight_triplet = weight_triplet
        self.triplet_margin = triplet_margin
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric_history = []
        self.batch_history_path = self.output_dir / "training_batches.csv"
        self.metric_csv_path = self.output_dir / "epoch_metrics.csv"

        self.register_model("model", model, optimizer, scheduler)

        if not self.batch_history_path.exists():
            with self.batch_history_path.open("w", newline="") as handle:
                csv.writer(handle).writerow([
                    "epoch", "batch", "total_loss", "softmax_loss",
                    "trihard_loss", "accuracy", "learning_rate"
                ])

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)
        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        logits, features = self.model(imgs)
        loss_softmax = F.cross_entropy(logits, pids)
        loss_triplet = batch_hard_triplet_loss(
            features, pids, margin=self.triplet_margin
        )
        total_loss = (
            self.weight_softmax * loss_softmax
            + self.weight_triplet * loss_triplet
        )

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()

        accuracy = metrics.accuracy(logits, pids)[0].item()
        lr = self.optimizer.param_groups[0]["lr"]

        with self.batch_history_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([
                self.epoch + 1,
                self.batch_idx + 1,
                total_loss.item(),
                loss_softmax.item(),
                loss_triplet.item(),
                accuracy,
                lr
            ])

        return {
            "loss": total_loss.item(),
            "softmax": loss_softmax.item(),
            "trihard": loss_triplet.item(),
            "acc": accuracy
        }

    @torch.no_grad()
    def _evaluate(
        self,
        dataset_name="",
        query_loader=None,
        gallery_loader=None,
        dist_metric="euclidean",
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir="",
        use_metric_cuhk03=False,
        ranks=(1, 5, 10, 20),
        rerank=False
    ):
        self.set_model_mode("eval")

        def extract(loader):
            all_features, all_pids, all_camids = [], [], []
            for data in loader:
                imgs, pids, camids = self.parse_data_for_eval(data)
                if self.use_gpu:
                    imgs = imgs.cuda()
                features = self.extract_features(imgs).cpu()
                all_features.append(features)
                all_pids.extend(pids.tolist())
                all_camids.extend(camids.tolist())
            return (
                torch.cat(all_features, dim=0),
                all_pids,
                all_camids
            )

        query_features, query_pids, query_camids = extract(query_loader)
        gallery_features, gallery_pids, gallery_camids = extract(gallery_loader)

        if normalize_feature:
            query_features = F.normalize(query_features, p=2, dim=1)
            gallery_features = F.normalize(gallery_features, p=2, dim=1)

        distance_matrix = metrics.compute_distance_matrix(
            query_features, gallery_features, metric=dist_metric
        )
        cmc, mean_ap = metrics.evaluate_rank(
            distance_matrix.numpy(),
            query_pids,
            gallery_pids,
            query_camids,
            gallery_camids,
            max_rank=max(20, max(ranks)),
            use_metric_cuhk03=use_metric_cuhk03
        )

        row = {
            "epoch": self.epoch + 1,
            "dataset": dataset_name,
            "rank1": float(cmc[0]),
            "rank5": float(cmc[4]),
            "rank10": float(cmc[9]),
            "rank20": float(cmc[19]),
            "mAP": float(mean_ap)
        }
        self.metric_history.append(row)
        self._save_metric_history()

        if self.writer is not None:
            for key in ("rank1", "rank5", "rank10", "rank20", "mAP"):
                self.writer.add_scalar(
                    f"Test/{dataset_name}/{key}", row[key], self.epoch
                )

        print(
            f"MSFANet | Epoch {row['epoch']} | "
            f"Rank-1 {row['rank1']:.4f} | Rank-5 {row['rank5']:.4f} | "
            f"Rank-10 {row['rank10']:.4f} | Rank-20 {row['rank20']:.4f} | "
            f"mAP {row['mAP']:.4f}"
        )
        return row["rank1"], row["mAP"]

    def _save_metric_history(self):
        if not self.metric_history:
            return
        with self.metric_csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.metric_history[0]))
            writer.writeheader()
            writer.writerows(self.metric_history)

        try:
            from openpyxl import Workbook
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Epoch Metrics"
            headers = list(self.metric_history[0])
            sheet.append(headers)
            for row in self.metric_history:
                sheet.append([row.get(header) for header in headers])
            workbook.save(self.output_dir / "msfanet_results.xlsx")
        except ImportError:
            pass
