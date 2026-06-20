from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from ingestion.chunker import chunks_from_document
from ingestion.docx_parser import parse_docx
from ingestion.manifest import load_manifest
from ingestion.staging import stage_documents
from scripts.run_phase1_local import main as run_phase1_main


class IngestionTests(unittest.TestCase):
    def test_manifest_loads_seed_document(self) -> None:
        docs = load_manifest(Path("specs/manifest.example.yaml"))

        self.assertEqual([doc.spec_id for doc in docs], ["3GPP TS 38.321", "3GPP TS 38.322", "3GPP TS 38.331"])
        self.assertTrue(all(doc.version == "v19.2.0" for doc in docs))

    def test_stage_documents_uses_local_seed_without_tracking_path(self) -> None:
        docs = load_manifest(Path("specs/manifest.example.yaml"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_dir = root / "seed"
            seed_dir.mkdir()
            for doc in docs:
                (seed_dir / doc.local_seed_filename).write_bytes(b"placeholder")

            staged = stage_documents(docs, data_dir=root / ".data", seed_dir=seed_dir, allow_download=False)

            self.assertEqual(len(staged), 3)
            self.assertTrue(all(item.path.exists() for item in staged))
            self.assertTrue(all(".data" in item.path.parts for item in staged))

    def test_docx_parser_extracts_heading_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docx_path = Path(tmp) / "sample.docx"
            _write_minimal_docx(docx_path)

            parsed = parse_docx(docx_path)

            self.assertEqual([section.section_id for section in parsed.sections], ["1", "1.1"])
            self.assertEqual(parsed.sections[0].title, "Scope")
            self.assertIn("scope body", parsed.sections[0].text)

    def test_chunks_include_mandatory_metadata(self) -> None:
        docs = load_manifest(Path("specs/manifest.example.yaml"))
        with tempfile.TemporaryDirectory() as tmp:
            docx_path = Path(tmp) / "sample.docx"
            _write_minimal_docx(docx_path)
            parsed = parse_docx(docx_path)

            chunks = chunks_from_document(parsed, docs[0], max_chars=500)
            row = json.loads(chunks[0].to_json())

            for field in [
                "spec_id",
                "release",
                "version",
                "section",
                "source_url",
                "chunk_hash",
                "doc_title",
            ]:
                self.assertIn(field, row)
            self.assertTrue(row["chunk_hash"].startswith("sha256:"))

    def test_phase1_runner_imports(self) -> None:
        self.assertTrue(callable(run_phase1_main))


def _write_minimal_docx(path: Path) -> None:
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:pPr><w:pStyle w:val="TOC1"/></w:pPr>
      <w:r><w:t>1Scope1</w:t></w:r>
    </w:p>
    <w:p>
      <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
      <w:r><w:t>1Scope</w:t></w:r>
    </w:p>
    <w:p><w:r><w:t>scope body</w:t></w:r></w:p>
    <w:p>
      <w:pPr><w:pStyle w:val="Heading2"/></w:pPr>
      <w:r><w:t>1.1Details</w:t></w:r>
    </w:p>
    <w:p><w:r><w:t>details body</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)


if __name__ == "__main__":
    unittest.main()
