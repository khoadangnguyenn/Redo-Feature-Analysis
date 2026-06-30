#!/usr/bin/env python3
"""Build the redone feature-analysis appendix PDF."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(".")
ANALYSIS_DIR = ROOT / "output" / "feature_analysis"
OUT_PDF = ROOT / "output" / "pdf" / "feature_analysis_redone.pdf"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def fnum(value: str | float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    iw, ih = ImageReader(str(path)).getSize()
    scale = min(max_width / iw, max_height / ih)
    return Image(str(path), width=iw * scale, height=ih * scale)


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def make_table(data: list[list[str]], widths: list[float], font_size: int = 8) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawString(0.75 * inch, 0.45 * inch, "Redone feature analysis")
    canvas.drawRightString(7.75 * inch, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    probe = read_csv(ANALYSIS_DIR / "ridge_probe_results.csv")
    align = read_csv(ANALYSIS_DIR / "depth_alignment_results.csv")
    index = read_csv(ANALYSIS_DIR / "depth_index_results.csv")
    moran = read_csv(ANALYSIS_DIR / "umap_moran_results.csv")
    results = json.loads((ANALYSIS_DIR / "feature_analysis_results.json").read_text())
    split = results["split"]

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#111827"),
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#111827"),
        spaceBefore=6,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=6,
    )
    note = ParagraphStyle(
        "Note",
        parent=body,
        backColor=colors.HexColor("#EFF6FF"),
        borderColor=colors.HexColor("#60A5FA"),
        borderWidth=0.6,
        borderPadding=7,
        leading=12,
    )

    story = []
    story.append(para("Redone Feature Analysis for D-LAPA Depth Injection", title))
    story.append(
        para(
            "Inputs: local paper <i>Depth Information Injection in VLA with Latent Action "
            "Pretraining via Synthetic Depth Data</i>, the prior feature-analysis appendix, "
            "and the provided <font name='Courier'>libero10_features</font> cache.",
            body,
        )
    )
    story.append(
        para(
            f"Protocol: {split['total_samples']:,} LIBERO-10 frame pairs, video-disjoint "
            f"80/20 split with seed 42 ({split['train_samples']:,} train, "
            f"{split['test_samples']:,} held out). Features are z-scored using train "
            "statistics only. The probe is Ridge regression with alpha=1000 and the LSQR "
            "solver, evaluated on xyz translation magnitude.",
            body,
        )
    )
    story.append(
        para(
            "<b>Cache compatibility note.</b> The provided "
            "<font name='Courier'>z_rgb_feature_input</font> behaves like the paper's "
            "pre-VQ LAPA-LAQ reference (R2=0.454), not the earlier appendix's finetuned-LAPA "
            "RGB reference (R2=0.619). The old figure should therefore be relabeled or rerun "
            "with a finetuned-LAPA feature cache if that baseline is required.",
            note,
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(scaled_image(ANALYSIS_DIR / "ridge_r2_probe_bars.png", 7.0 * inch, 2.5 * inch))
    story.append(Spacer(1, 0.08 * inch))

    baseline = float(next(row for row in probe if row["feature"] == "RGB")["ridge_r2"])
    table_data = [["Representation", "Dim", "R2", "Delta", "Spearman"]]
    for row in probe:
        feature = row["feature"]
        if feature == "GT depth":
            continue
        delta = float(row["ridge_r2"]) - baseline
        table_data.append(
            [
                feature,
                str(int(float(row["dim"]))),
                fnum(row["ridge_r2"], 4),
                f"{delta:+.4f}",
                fnum(row["spearman_rho"], 4),
            ]
        )
    story.append(make_table(table_data, [2.35 * inch, 0.55 * inch, 0.65 * inch, 0.65 * inch, 0.8 * inch]))

    story.append(PageBreak())
    story.append(para("UMAP Smoothness Snapshot", title))
    story.append(
        para(
            "UMAP is used only as a qualitative visualization on 3,000 sampled held-out pairs. "
            "Moran's I measures whether nearby projected points share similar action magnitude. "
            "The quantitative conclusions should be read from the probe table.",
            body,
        )
    )
    story.append(scaled_image(ANALYSIS_DIR / "umap_moran_feature_snapshot.png", 7.2 * inch, 4.0 * inch))
    story.append(Spacer(1, 0.1 * inch))
    moran_table = [["Representation", "Moran's I"]]
    for row in moran:
        moran_table.append([row["feature"], fnum(row["morans_i"], 4)])
    story.append(make_table(moran_table, [3.4 * inch, 1.0 * inch], font_size=8))
    story.append(
        para(
            "Interpretation: Model 2 and Model 6.1 give the strongest linear magnitude "
            "decodability (R2=0.520), while Model 4 gives the smoothest UMAP magnitude "
            "neighborhoods (Moran's I=0.472). These are complementary diagnostics, not "
            "identical claims.",
            body,
        )
    )

    story.append(PageBreak())
    story.append(para("Diagnostics and Replacement Text", title))
    story.append(para("Depth-feature faithfulness to the Stage-1 depth teacher:", h2))
    align_table = [["Feature", "MSE to GT", "Cosine to GT"]]
    for row in align:
        align_table.append([row["feature"], fnum(row["mse_to_gt"], 5), fnum(row["mean_cosine_to_gt"], 4)])
    story.append(make_table(align_table, [2.1 * inch, 1.1 * inch, 1.1 * inch], font_size=8))
    story.append(Spacer(1, 0.12 * inch))
    story.append(para("Depth-index diagnostics where predicted indices/confidence are available:", h2))
    index_table = [["Feature", "Token acc.", "Seq. acc.", "Mean conf."]]
    for row in index:
        index_table.append(
            [
                row["feature"],
                fnum(row["token_accuracy"], 4),
                fnum(row["sequence_accuracy"], 4),
                fnum(row["mean_confidence"], 4),
            ]
        )
    story.append(make_table(index_table, [1.7 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch], font_size=8))
    story.append(Spacer(1, 0.12 * inch))
    story.append(para("Suggested replacement feature-analysis paragraph:", h2))
    story.append(
        para(
            "We probe the geometric content of deployment-equivalent representations on "
            "138,090 LIBERO-10 frame pairs using a video-disjoint 80/20 split. The provided "
            "RGB representation reaches R2=0.454, indicating that this cache corresponds to "
            "the pre-VQ LAPA-LAQ feature scale rather than the finetuned-LAPA reference used "
            "in the earlier appendix. Concatenating Stage-2.5 depth features gives a selective "
            "improvement: RGB + Model 2 reaches R2=0.520, a +0.066 absolute gain over RGB; "
            "Model 6.1 is effectively tied at R2=0.520. The no-depth controls Model 3 and "
            "Model 5 remain close to RGB (+0.003 and +0.001), supporting the interpretation "
            "that the main gain comes from explicit depth-image variants rather than capacity "
            "alone. Model 4 is weaker in linear R2 but produces the smoothest UMAP magnitude "
            "neighborhoods, so the feature analysis should separate linear decodability, "
            "visual smoothness, and downstream policy success.",
            body,
        )
    )
    story.append(
        para(
            "Main conclusion: depth-derived representations add measurable geometric "
            "decodability to the provided pre-finetune RGB feature cache, but the result is "
            "incremental and must be reported with the cache mismatch and downstream policy "
            "evaluation caveats.",
            note,
        )
    )

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.65 * inch,
        title="Redone Feature Analysis",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
