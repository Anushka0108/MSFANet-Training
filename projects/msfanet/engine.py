import csv
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
        self.loss_history = []

        self.metric_csv_path = self.output_dir / "epoch_metrics.csv"
        self.loss_csv_path = self.output_dir / "epoch_losses.csv"
        self.excel_path = self.output_dir / "msfanet_results.xlsx"

        self.loss_sums = {}
        self.loss_batch_count = 0
        self.loss_epoch = None

        self.register_model("model", model, optimizer, scheduler)

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        logits, features = self.model(imgs)

        loss_softmax = F.cross_entropy(logits, pids)

        loss_triplet = batch_hard_triplet_loss(
            features,
            pids,
            margin=self.triplet_margin
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

        batch_losses = {
            "loss": total_loss.item(),
            "softmax": loss_softmax.item(),
            "trihard": loss_triplet.item(),
            "acc": accuracy,
            "lr": lr
        }

        current_epoch = self.epoch + 1

        if self.loss_epoch != current_epoch:
            self.loss_epoch = current_epoch
            self.loss_sums = {
                key: 0.0 for key in batch_losses
            }
            self.loss_batch_count = 0

        for key, value in batch_losses.items():
            self.loss_sums[key] += value

        self.loss_batch_count += 1

        if self.batch_idx + 1 == len(self.train_loader):
            self.save_epoch_losses(current_epoch)

        return batch_losses

    def save_epoch_losses(self, epoch):
        if self.loss_batch_count == 0:
            return

        row = {
            "epoch": epoch,
            "total_loss": self.loss_sums["loss"] / self.loss_batch_count,
            "softmax_loss": self.loss_sums["softmax"] / self.loss_batch_count,
            "trihard_loss": self.loss_sums["trihard"] / self.loss_batch_count,
            "accuracy": self.loss_sums["acc"] / self.loss_batch_count,
            "learning_rate": self.loss_sums["lr"] / self.loss_batch_count
        }

        self.loss_history.append(row)

        with self.loss_csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(row.keys())
            )
            writer.writeheader()
            writer.writerows(self.loss_history)

        self._save_excel_sheet("Epoch Losses", self.loss_history)

        print(
            f"MSFANet | Epoch {epoch} Loss | "
            f"Total: {row['total_loss']:.4f} | "
            f"Softmax: {row['softmax_loss']:.4f} | "
            f"Triplet: {row['trihard_loss']:.4f} | "
            f"Accuracy: {row['accuracy']:.2f}%"
        )

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
            all_features = []
            all_pids = []
            all_camids = []

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
            query_features,
            gallery_features,
            metric=dist_metric
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

        rank1 = float(cmc[0])
        rank5 = float(cmc[4])
        rank10 = float(cmc[9])
        rank20 = float(cmc[19])
        mean_ap = float(mean_ap)

        row = {
            "epoch": self.epoch + 1,
            "dataset": dataset_name,
            "rank1": rank1 * 100,
            "rank5": rank5 * 100,
            "rank10": rank10 * 100,
            "rank20": rank20 * 100,
            "mAP": mean_ap * 100
        }

        self.metric_history.append(row)
        self._save_metric_history()

        if self.writer is not None:
            for key in ("rank1", "rank5", "rank10", "rank20", "mAP"):
                self.writer.add_scalar(
                    f"Test/{dataset_name}/{key}",
                    row[key],
                    self.epoch
                )

        print(
            f"MSFANet | Epoch {row['epoch']} | "
            f"Rank-1 {row['rank1']:.2f}% | "
            f"Rank-5 {row['rank5']:.2f}% | "
            f"Rank-10 {row['rank10']:.2f}% | "
            f"Rank-20 {row['rank20']:.2f}% | "
            f"mAP {row['mAP']:.2f}%"
        )

        return rank1, mean_ap

    def save_epoch_visualizations(self, epoch):
        if epoch % 10 != 0:
            return

        query_loader = self.test_loader["market1501"]["query"]
        sample_batch = next(iter(query_loader))

        from visualization import save_feature_visualizations

        save_feature_visualizations(
            self.model,
            sample_batch["img"],
            self.output_dir,
            epoch=epoch,
            max_images=4
        )

    def _save_metric_history(self):
        if not self.metric_history:
            return

        with self.metric_csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(self.metric_history[0].keys())
            )
            writer.writeheader()
            writer.writerows(self.metric_history)

        self._save_excel_sheet("Epoch Metrics", self.metric_history)

    def _save_excel_sheet(self, sheet_name, history):
        from openpyxl import Workbook, load_workbook

        if self.excel_path.exists():
            workbook = load_workbook(self.excel_path)
        else:
            workbook = Workbook()

        if sheet_name in workbook.sheetnames:
            del workbook[sheet_name]

        if workbook.sheetnames == ["Sheet"] and not history:
            sheet = workbook["Sheet"]
        elif workbook.sheetnames == ["Sheet"] and sheet_name != "Sheet":
            del workbook["Sheet"]
            sheet = workbook.create_sheet(sheet_name)
        else:
            sheet = workbook.create_sheet(sheet_name)

        sheet.title = sheet_name

        if history:
            headers = list(history[0].keys())
            sheet.append(headers)

            for row in history:
                sheet.append([row.get(header) for header in headers])

        workbook.save(self.excel_path)