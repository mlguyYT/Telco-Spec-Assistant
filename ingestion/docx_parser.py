from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}
W_VAL = f"{{{W_NS}}}val"


@dataclass
class Section:
    section_id: str
    title: str
    level: int
    paragraphs: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs).strip()


@dataclass(frozen=True)
class ParsedDocument:
    sections: list[Section]


def parse_docx(path: Path) -> ParsedDocument:
    with zipfile.ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")

    root = ET.fromstring(document_xml)
    body = root.find("w:body", NS)
    if body is None:
        raise ValueError("DOCX is missing word/document.xml body")

    sections: list[Section] = []
    current: Section | None = None
    body_started = False

    for child in body:
        tag = _local_name(child.tag)
        if tag == "p":
            style = _paragraph_style(child)
            text = _paragraph_text(child)
            if not text or _is_toc_style(style):
                continue

            heading = _heading_from_paragraph(style, text)
            if heading is not None:
                body_started = True
                current = Section(section_id=heading[0], title=heading[1], level=heading[2])
                sections.append(current)
                continue

            if not body_started:
                continue
            if current is None:
                current = Section(section_id="front-matter", title="Front matter", level=0)
                sections.append(current)
            current.paragraphs.append(text)

        elif tag == "tbl" and body_started and current is not None:
            for row in _table_rows(child):
                current.paragraphs.append(row)

    return ParsedDocument(sections=[section for section in sections if section.text])


def _paragraph_style(paragraph: ET.Element) -> str | None:
    style = paragraph.find("w:pPr/w:pStyle", NS)
    return style.attrib.get(W_VAL) if style is not None else None


def _paragraph_text(paragraph: ET.Element) -> str:
    text = "".join(node.text or "" for node in paragraph.findall(".//w:t", NS))
    return _normalize_text(text)


def _table_rows(table: ET.Element) -> list[str]:
    rows: list[str] = []
    for row in table.findall("w:tr", NS):
        cells = []
        for cell in row.findall("w:tc", NS):
            cell_text = " ".join(_paragraph_text(paragraph) for paragraph in cell.findall("w:p", NS))
            if cell_text:
                cells.append(cell_text)
        if cells:
            rows.append(" | ".join(cells))
    return rows


def _heading_from_paragraph(style: str | None, text: str) -> tuple[str, str, int] | None:
    if style == "Heading1" and text == "Foreword":
        return ("foreword", "Foreword", 1)
    if not style or not style.startswith("Heading"):
        return None

    level_match = re.search(r"(\d+)$", style)
    level = int(level_match.group(1)) if level_match else 1

    annex_match = re.match(r"^(Annex\s+[A-Z])\s*(.*)$", text)
    if annex_match:
        section_id = annex_match.group(1)
        title = annex_match.group(2).strip(" :-") or section_id
        return (section_id, title, level)

    section_match = re.match(r"^(\d+(?:\.\d+)*)(.*)$", text)
    if not section_match:
        return None
    section_id = section_match.group(1)
    title = section_match.group(2).strip() or section_id
    return (section_id, title, level)


def _is_toc_style(style: str | None) -> bool:
    return bool(style and style.startswith("TOC"))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
