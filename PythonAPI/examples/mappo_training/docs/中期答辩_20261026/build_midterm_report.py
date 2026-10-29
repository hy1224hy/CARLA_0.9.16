"""Create the Word report from the reviewed Markdown source."""
from pathlib import Path
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


BASE = Path(__file__).resolve().parent
SOURCE = BASE / "研究生中期报告_20261027.md"
DEST = BASE / "研究生中期报告_20261027.docx"


def cjk_font(style, western="Times New Roman", chinese="宋体", size=12):
    style.font.name = western
    style.font.size = Pt(size)
    style.font.color.rgb = RGBColor(0, 0, 0)
    fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    fonts.set(qn("w:ascii"), western)
    fonts.set(qn("w:hAnsi"), western)
    fonts.set(qn("w:eastAsia"), chinese)
    language = OxmlElement("w:lang")
    language.set(qn("w:eastAsia"), "zh-CN")
    style.element.get_or_add_rPr().append(language)


def set_cell(cell, text, *, header=False, center=False):
    cell.text = text
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    pr = cell._tc.get_or_add_tcPr()
    margins = OxmlElement("w:tcMar")
    for side in ("top", "left", "bottom", "right"):
        element = OxmlElement("w:" + side)
        element.set(qn("w:w"), "95")
        element.set(qn("w:type"), "dxa")
        margins.append(element)
    pr.append(margins)
    for p in cell.paragraphs:
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.line_spacing = 1.12
        p.paragraph_format.keep_with_next = False
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
        for run in p.runs:
            run.font.size = Pt(10)
            run.bold = header
    if header:
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "E8EDF2")
        pr.append(shading)


def table(doc, rows):
    count = len(rows[0])
    tab = doc.add_table(rows=1, cols=count)
    tab.alignment = WD_TABLE_ALIGNMENT.CENTER
    tab.autofit = False
    widths = ([2.45, 2.45, 3.55, 4.0, 3.15] if count == 5 else
              [3.3, 4.0, 5.2, 3.1])
    for column, width in zip(tab.columns, widths):
        column.width = Cm(width)
    for index, texts in enumerate(rows):
        cells = tab.rows[0].cells if index == 0 else tab.add_row().cells
        for col, text in enumerate(texts):
            set_cell(cells[col], text, header=index == 0,
                     center=index == 0 or (count == 5 and col > 0))
            cells[col].width = Cm(widths[col])
        if index == 0:
            repeat = OxmlElement("w:tblHeader")
            tab.rows[0]._tr.get_or_add_trPr().append(repeat)
        no_split = OxmlElement("w:cantSplit")
        tab.rows[index]._tr.get_or_add_trPr().append(no_split)
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        line = OxmlElement("w:" + side)
        line.set(qn("w:val"), "single")
        line.set(qn("w:sz"), "4")
        line.set(qn("w:color"), "D9D9D9")
        borders.append(line)
    tab._tbl.tblPr.append(borders)
    after = doc.add_paragraph()
    after.paragraph_format.space_after = Pt(2)
    after.paragraph_format.space_before = Pt(0)
    after.paragraph_format.line_spacing = 0.5
    after.add_run().font.size = Pt(3)


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.top_margin, sec.bottom_margin = Cm(2.4), Cm(2.3)
    sec.left_margin, sec.right_margin = Cm(2.6), Cm(2.5)
    sec.footer_distance = Cm(1.1)
    cjk_font(doc.styles["Normal"], size=12)
    normal = doc.styles["Normal"].paragraph_format
    normal.line_spacing = 1.28
    normal.first_line_indent = Pt(24)
    normal.space_after = Pt(5)
    normal.widow_control = True
    for name, size in (("Title", 20), ("Heading 1", 15), ("Heading 2", 12)):
        style = doc.styles[name]
        cjk_font(style, chinese="黑体", size=size)
        style.font.bold = name != "Title"
        pf = style.paragraph_format
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(12 if name == "Heading 1" else 8)
        pf.space_after = Pt(6)
        pf.keep_with_next = True
        pf.keep_together = True
        pf.line_spacing = 1.15
    doc.core_properties.title = "基于对抗学习的多车协作驾驶算法研究 中期报告"
    doc.core_properties.subject = "研究生学位论文中期检查"
    doc.core_properties.author = ""
    doc.core_properties.keywords = "多智能体强化学习 Mut-MAPPO 通信 对抗训练"
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    index = 0
    reference_mode = False
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    rows.append(cells)
                index += 1
            table(doc, rows)
            continue
        if line.startswith("# "):
            paragraph = doc.add_paragraph(line[2:], "Title")
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif line.startswith("## "):
            text = line[3:]
            reference_mode = text.startswith("六 ")
            paragraph = doc.add_paragraph(text, "Heading 1")
            if reference_mode:
                paragraph.paragraph_format.page_break_before = True
        elif line.startswith("### "):
            paragraph = doc.add_paragraph(line[4:], "Heading 2")
        else:
            paragraph = doc.add_paragraph(line)
            if line.startswith(("论文题目", "学生姓名", "指导教师", "报告日期")):
                paragraph.paragraph_format.first_line_indent = Pt(0)
                paragraph.paragraph_format.space_after = Pt(5)
                if line.startswith("论文题目"):
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in paragraph.runs:
                        run.bold = True
            if reference_mode:
                paragraph.paragraph_format.first_line_indent = Pt(-18)
                paragraph.paragraph_format.left_indent = Pt(18)
                paragraph.paragraph_format.line_spacing = 1.15
                paragraph.paragraph_format.space_after = Pt(6)
                for run in paragraph.runs:
                    run.font.size = Pt(10.5)
        index += 1
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.paragraph_format.first_line_indent = Pt(0)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    for style in doc.styles:
        if style.type == 1:
            style.font.color.rgb = RGBColor(0, 0, 0)
        for border in list(style.element.iter(qn("w:pBdr"))):
            border.getparent().remove(border)
    for paragraph in doc.paragraphs:
        for border in list(paragraph._p.iter(qn("w:pBdr"))):
            border.getparent().remove(border)
    doc.save(DEST)
    print(DEST)


if __name__ == "__main__":
    main()
