"""Professional PDF assessment summary generation for StrokeGuard."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any

from app_core import APP_NAME, APP_SUBTITLE, CREDIT_NAME, SYMPTOM_FIELDS


def generate_pdf_report(result: dict[str, Any]) -> bytes:
    """Generate a polished clinical decision support summary."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        HRFlowable,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    import reportlab

    font_dir = Path(reportlab.__file__).resolve().parent / "fonts"
    if "SGSans" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("SGSans", str(font_dir / "Vera.ttf")))
        pdfmetrics.registerFont(
            TTFont("SGSans-Bold", str(font_dir / "VeraBd.ttf"))
        )
        pdfmetrics.registerFontFamily(
            "SGSans",
            normal="SGSans",
            bold="SGSans-Bold",
            italic="SGSans",
            boldItalic="SGSans-Bold",
        )

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=19 * mm,
        title=f"{APP_NAME} - {APP_SUBTITLE} Assessment Summary",
        author=CREDIT_NAME,
        subject="Symptom-based stroke assessment summary",
    )

    styles = getSampleStyleSheet()
    navy = colors.HexColor("#10263A")
    teal = colors.HexColor("#168B82")
    pale_teal = colors.HexColor("#E9F7F5")
    pale_blue = colors.HexColor("#EEF4F8")
    muted = colors.HexColor("#5C6E7D")
    text = colors.HexColor("#243746")
    danger = colors.HexColor("#C63E52")

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="SGSans-Bold",
        fontSize=20,
        leading=24,
        textColor=navy,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="SGSans",
        fontSize=9,
        leading=13,
        textColor=muted,
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="SGSans-Bold",
        fontSize=12,
        leading=15,
        textColor=navy,
        spaceBefore=9,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName="SGSans",
        fontSize=9.2,
        leading=14,
        textColor=text,
        spaceAfter=5,
    )
    small_style = ParagraphStyle(
        "Small",
        parent=body_style,
        fontSize=8,
        leading=11,
        textColor=muted,
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=body_style,
        fontSize=8.3,
        leading=11,
        spaceAfter=0,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=body_style,
        fontSize=8.5,
        leading=12,
        textColor=danger,
        borderColor=colors.HexColor("#E9B9C0"),
        borderWidth=0.7,
        borderPadding=8,
        backColor=colors.HexColor("#FFF5F6"),
    )

    def page_footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D9E3EA"))
        canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
        canvas.setFont("SGSans", 7.5)
        canvas.setFillColor(muted)
        canvas.drawString(
            18 * mm,
            9.5 * mm,
            f"{APP_NAME} | Product design and application engineering by {CREDIT_NAME}",
        )
        canvas.drawRightString(192 * mm, 9.5 * mm, f"Page {doc.page}")
        canvas.restoreState()

    story: list[Any] = []
    assessed_at = datetime.fromisoformat(result["assessed_at"]).strftime(
        "%d %B %Y, %I:%M %p"
    )
    band = result["band"]
    inputs = result["inputs"]

    story.append(Paragraph(APP_NAME, title_style))
    story.append(
        Paragraph(
            f"{APP_SUBTITLE} | Doctor-guided decisions come first",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=teal))
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Important: This summary provides an experimental estimate based only on "
            "the information entered. Do not rely on it alone for diagnosis, treatment, "
            "emergency decisions, fitness, or medical clearance. A qualified doctor's "
            "assessment and advice should remain the primary basis for care.",
            disclaimer_style,
        )
    )
    story.append(Spacer(1, 10))

    summary_data = [
        ["Assessment summary", "Result"],
        ["Symptom-based indicator", f'{result["score"]:.1f}%'],
        ["Assessment range", band["label"]],
        ["Pattern summary", result["binary_label"]],
        ["Selected symptoms", str(len(result["active_symptoms"]))],
        ["Assessment time", assessed_at],
    ]
    summary = Table(summary_data, colWidths=[62 * mm, 105 * mm], repeatRows=1)
    summary.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "SGSans"),
                ("BACKGROUND", (0, 0), (-1, 0), navy),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "SGSans-Bold"),
                ("BACKGROUND", (0, 1), (0, -1), pale_blue),
                ("BACKGROUND", (1, 1), (1, 2), pale_teal),
                ("FONTNAME", (0, 1), (0, -1), "SGSans-Bold"),
                ("TEXTCOLOR", (0, 1), (-1, -1), text),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CFDCE5")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(summary)

    story.append(Paragraph("Patient inputs", heading_style))
    patient_rows = [
        ["Parameter", "Recorded value"],
        ["Age", str(inputs["Age"])],
        ["Gender", str(inputs["Gender"])],
    ]
    patient_rows.extend(
        [label, "Present" if inputs[name] else "Absent"]
        for name, label in SYMPTOM_FIELDS
    )
    patient_table = Table(
        patient_rows,
        colWidths=[93 * mm, 74 * mm],
        repeatRows=1,
    )
    patient_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "SGSans"),
                ("BACKGROUND", (0, 0), (-1, 0), teal),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "SGSans-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale_blue]),
                ("TEXTCOLOR", (0, 1), (-1, -1), text),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D7E2E9")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(patient_table)
    story.append(PageBreak())

    story.append(Paragraph("Indicator insights", heading_style))
    story.append(
        Paragraph(
            "These values show how the estimate changed when one selected input "
            "was altered while the other inputs stayed the same. They do not prove "
            "cause, severity, or an individual medical contribution.",
            small_style,
        )
    )
    sensitivity = result["sensitivity"]
    if sensitivity:
        sensitivity_rows = [["Input comparison", "Score change"]]
        for item in sensitivity[:10]:
            sensitivity_rows.append(
                [
                    Paragraph(
                        f'{item["feature"]}: {item["comparison"]}',
                        table_cell_style,
                    ),
                    Paragraph(
                        f'{item["effect_points"]:+.1f} points',
                        table_cell_style,
                    ),
                ]
            )
        sensitivity_table = Table(
            sensitivity_rows,
            colWidths=[130 * mm, 37 * mm],
            repeatRows=1,
        )
        sensitivity_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "SGSans"),
                    ("BACKGROUND", (0, 0), (-1, 0), navy),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "SGSans-Bold"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale_blue]),
                    ("TEXTCOLOR", (0, 1), (-1, -1), text),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D7E2E9")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(sensitivity_table)
    else:
        story.append(
            Paragraph(
                "No active indicator comparison was available for this assessment.",
                body_style,
            )
        )

    story.append(Paragraph("Guidance to discuss", heading_style))
    for item in result["guidance"]:
        story.append(
            KeepTogether(
                [
                    Paragraph(f'<b>{item["title"]}</b>', body_style),
                    Paragraph(item["text"], small_style),
                    Spacer(1, 4),
                ]
            )
        )

    story.append(Paragraph("Possible clinical discussion topics", heading_style))
    discussion_rows = [["Topic", "Why it may be discussed"]]
    discussion_rows.extend(
        [
            Paragraph(item["name"], table_cell_style),
            Paragraph(item["reason"], table_cell_style),
        ]
        for item in result["clinical_discussions"]
    )
    discussion_table = Table(
        discussion_rows,
        colWidths=[57 * mm, 110 * mm],
        repeatRows=1,
    )
    discussion_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "SGSans"),
                ("BACKGROUND", (0, 0), (-1, 0), teal),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "SGSans-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale_teal]),
                ("TEXTCOLOR", (0, 1), (-1, -1), text),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D7E2E9")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(discussion_table)

    document.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    buffer.seek(0)
    return buffer.getvalue()
