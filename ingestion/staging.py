from __future__ import annotations

import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ingestion.manifest import SpecDocument


@dataclass(frozen=True)
class StagedDocument:
    spec: SpecDocument
    path: Path
    source: str


def stage_documents(
    documents: list[SpecDocument],
    data_dir: Path,
    seed_dir: Path | None = None,
    allow_download: bool = True,
) -> list[StagedDocument]:
    raw_dir = safe_data_subdir(data_dir, "raw")
    staged: list[StagedDocument] = []

    for doc in documents:
        destination = raw_dir / doc.local_seed_filename
        if destination.exists():
            staged.append(StagedDocument(spec=doc, path=destination, source="existing"))
            continue

        if seed_dir is not None:
            local_seed = seed_dir / doc.local_seed_filename
            if local_seed.exists():
                shutil.copy2(local_seed, destination)
                staged.append(StagedDocument(spec=doc, path=destination, source=f"seed:{local_seed.name}"))
                continue

        if allow_download:
            staged.append(_download_and_stage(doc, raw_dir))
            continue

        raise FileNotFoundError(
            f"{doc.local_seed_filename} was not found in {raw_dir}"
            + (f" or {seed_dir}" if seed_dir else "")
        )

    return staged


def safe_data_subdir(data_dir: Path, name: str) -> Path:
    root = data_dir.resolve()
    target = (root / name).resolve()
    if root == Path(".").resolve() or root in Path.cwd().resolve().parents:
        raise ValueError("data_dir must not be the repository root or one of its parents")
    if root not in target.parents:
        raise ValueError(f"unsafe data path: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _download_and_stage(doc: SpecDocument, raw_dir: Path) -> StagedDocument:
    if doc.source_url.startswith("TODO"):
        raise ValueError(f"source_url for {doc.id} is not verified")

    archive_name = doc.source_url.rstrip("/").split("/")[-1]
    if not archive_name:
        raise ValueError(f"source_url for {doc.id} does not contain a filename")
    archive_path = raw_dir / archive_name

    urllib.request.urlretrieve(doc.source_url, archive_path)
    if archive_path.suffix.lower() == ".zip":
        return _extract_doc_from_zip(doc, archive_path, raw_dir)

    if archive_path.name != doc.local_seed_filename:
        destination = raw_dir / doc.local_seed_filename
        shutil.copy2(archive_path, destination)
        return StagedDocument(spec=doc, path=destination, source=doc.source_url)

    return StagedDocument(spec=doc, path=archive_path, source=doc.source_url)


def _extract_doc_from_zip(doc: SpecDocument, archive_path: Path, raw_dir: Path) -> StagedDocument:
    destination = raw_dir / doc.local_seed_filename
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        exact = [name for name in names if Path(name).name == doc.local_seed_filename]
        docx_files = [name for name in names if name.lower().endswith(".docx")]
        candidates = exact or docx_files
        if not candidates:
            raise FileNotFoundError(f"no DOCX file found in {archive_path.name}")
        with archive.open(candidates[0]) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)
    return StagedDocument(spec=doc, path=destination, source=f"{doc.source_url}#{candidates[0]}")
