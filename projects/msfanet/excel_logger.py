
import os
import pandas as pd


class ExcelLogger:

    def __init__(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        self.file_path = os.path.join(
            output_dir,
            "msfanet_training_results.xlsx"
        )

        self.training_records = []
        self.evaluation_records = []

    def add_training(
        self,
        epoch,
        classification_loss,
        triplet_loss,
        total_loss,
        accuracy,
        learning_rate
    ):
        self.training_records.append({
            "Epoch": epoch,
            "Classification Loss": classification_loss,
            "Triplet Loss": triplet_loss,
            "Total Loss": total_loss,
            "Training Accuracy (%)": accuracy,
            "Learning Rate": learning_rate
        })

    def add_evaluation(
        self,
        epoch,
        rank1,
        rank5,
        rank10,
        rank20,
        mean_ap
    ):
        self.evaluation_records.append({
            "Epoch": epoch,
            "Rank-1 (%)": rank1,
            "Rank-5 (%)": rank5,
            "Rank-10 (%)": rank10,
            "Rank-20 (%)": rank20,
            "mAP (%)": mean_ap
        })

    def save(self, config, complexity):
        with pd.ExcelWriter(
            self.file_path,
            engine="openpyxl"
        ) as writer:

            pd.DataFrame(
                self.training_records
            ).to_excel(
                writer,
                sheet_name="Training",
                index=False
            )

            pd.DataFrame(
                self.evaluation_records
            ).to_excel(
                writer,
                sheet_name="Evaluation",
                index=False
            )

            pd.DataFrame([
                {"Parameter": key, "Value": value}
                for key, value in config.items()
            ]).to_excel(
                writer,
                sheet_name="Configuration",
                index=False
            )

            pd.DataFrame([
                {"Metric": key, "Value": value}
                for key, value in complexity.items()
            ]).to_excel(
                writer,
                sheet_name="Complexity",
                index=False
            )

        print("Excel workbook saved:", self.file_path)
