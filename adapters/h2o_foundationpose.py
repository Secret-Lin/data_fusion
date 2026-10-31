from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import zipfile

import trimesh
from PIL import Image

from adapters.foundationpose_runner import HopeModel


@dataclass(frozen=True)
class H2OModel(HopeModel):
    """H2O mesh metadata; H2O object meshes are already expressed in metres."""


class H2OModelRepository:
    """Expose H2O's ``object.zip`` or extracted object directory to FoundationPose.

    The official archive contains ``object/<name>/<name>.obj`` files.  The
    archive is extracted into a project-local cache so the vendored
    FoundationPose code can load textures and material files by path.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        extraction_root: str | Path | None = None,
        scale_to_m: float = 1.0,
    ) -> None:
        self.source = Path(source).expanduser().resolve()
        self.extraction_root = Path(
            extraction_root or self.source.parent / ".h2o_models"
        ).expanduser().resolve()
        self.scale_to_m = float(scale_to_m)
        if self.scale_to_m <= 0:
            raise ValueError("scale_to_m must be positive")
        self._resolved_root: Path | None = None
        self._model_index: dict[str, H2OModel] | None = None

    def names(self) -> list[str]:
        return sorted(self._models())

    def resolve(self, name: str) -> H2OModel:
        key = str(name).strip().casefold()
        matches = [model for model in self._models().values() if model.name.casefold() == key]
        if not matches:
            raise KeyError(
                f"Unknown H2O object {name!r}; available: {', '.join(self.names())}"
            )
        return matches[0]

    def load_mesh(self, name: str) -> tuple[H2OModel, trimesh.Trimesh]:
        model = self.resolve(name)
        mesh = trimesh.load(model.mesh_path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"H2O mesh did not load as Trimesh: {model.mesh_path}")
        mesh = mesh.copy()
        # Official cocoa MTL references cocoa.png, but object.zip ships
        # ovomaltine.png. Repair only this known asset alias in memory.
        material = getattr(mesh.visual, "material", None)
        if model.name.casefold() == "cocoa" and getattr(material, "image", None) is None:
            texture = model.mesh_path.parent / "ovomaltine.png"
            if not texture.is_file() or material is None:
                raise FileNotFoundError(f"Missing H2O cocoa texture: {texture}")
            with Image.open(texture) as image:
                material.image = image.convert("RGB")
        mesh.apply_scale(model.scale_to_m)
        if len(mesh.vertices) == 0 or not mesh.is_volume and not mesh.faces.size:
            raise ValueError(f"H2O mesh is empty: {model.mesh_path}")
        return model, mesh

    def _models(self) -> dict[str, H2OModel]:
        if self._model_index is not None:
            return self._model_index
        root = self._materialize()
        candidates: list[Path] = []
        for base in (root / "object", root):
            if not base.is_dir():
                continue
            candidates.extend(sorted(base.glob("*/" + "*.obj")))
        found: dict[str, H2OModel] = {}
        for mesh_path in candidates:
            # ``root/object/name/name.obj`` is the official layout.  For an
            # already flattened directory, use the parent directory name.
            name = mesh_path.parent.name
            if name.casefold() == "object":
                name = mesh_path.stem
            found.setdefault(
                name,
                H2OModel(name, mesh_path.resolve(), self.scale_to_m),
            )
        if not found:
            raise FileNotFoundError(
                "No H2O object meshes found. Expected object.zip or object/<name>/*.obj "
                f"under {self.source}"
            )
        self._model_index = found
        return found

    def _materialize(self) -> Path:
        if self._resolved_root is not None:
            return self._resolved_root
        if self.source.is_dir():
            self._resolved_root = self.source
            return self._resolved_root
        if not self.source.is_file() or self.source.suffix.casefold() != ".zip":
            raise FileNotFoundError(f"H2O mesh source not found: {self.source}")

        marker = self.extraction_root / ".complete"
        if not marker.is_file():
            self.extraction_root.mkdir(parents=True, exist_ok=True)
            root_resolved = self.extraction_root.resolve()
            with zipfile.ZipFile(self.source) as archive:
                for member in archive.infolist():
                    destination = (self.extraction_root / member.filename).resolve()
                    if os.path.commonpath((str(root_resolved), str(destination))) != str(root_resolved):
                        raise ValueError(f"Unsafe H2O mesh archive member: {member.filename}")
                archive.extractall(self.extraction_root)
            marker.write_text("H2O object archive extracted\n", encoding="utf-8")
        self._resolved_root = self.extraction_root
        return self._resolved_root
