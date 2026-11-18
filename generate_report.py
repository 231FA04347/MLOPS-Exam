"""Generate a compact PDF report from training artifacts."""

from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

from src.data_pipeline import build_feature_store
from src.train import train_models


def main() -> None:
    Path("reports").mkdir(exist_ok=True)
    data = build_feature_store()
    metrics = train_models()
    summary = data.groupby("dataset_version")["energy_consumption_kwh"].agg(["count", "mean"])
    doc = SimpleDocTemplate("reports/kiln_energy_report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("Ceramic Kiln Energy MLOps Report", styles["Title"])]
    story.append(Paragraph(
        "The application generates two synthetic kiln versions, reuses engineered thermal and moisture features, "
        "compares Linear Regression with Random Forest, registers the best MLflow model, and serves predictions through FastAPI.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Dataset version comparison", styles["Heading2"]))
    story.append(Table([["Version", "Records", "Mean energy (kWh)"]] +
                       [[index, int(row["count"]), f'{row["mean"]:.2f}'] for index, row in summary.iterrows()],
                       style=TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)])))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Model performance", styles["Heading2"]))
    story.append(Table([list(metrics.columns)] +
                       [[row["model"], f'{row["mae"]:.3f}', f'{row["rmse"]:.3f}', f'{row["r2"]:.3f}']
                        for _, row in metrics.iterrows()],
                       style=TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)])))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Architecture: CSV generator -> feature store -> sklearn preprocessing/model pipeline -> MLflow registry -> FastAPI. "
        "GitHub Actions runs pytest for validation and non-negative prediction behavior. "
        "The registered model URI is saved in artifacts/registered_model_uri.txt.",
        styles["BodyText"],
    ))
    doc.build(story)


if __name__ == "__main__":
    main()
