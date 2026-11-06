import json
import logging
import math
import os
import re
import sys
from typing import Annotated, Optional, Sequence

import vtk

import qt
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
    Choice,
)

from slicer import vtkMRMLMarkupsFiducialNode
from slicer import vtkMRMLScalarVolumeNode

SEED_LINE_ATTRIBUTE = "SeqSeg.SeedLine"
# Per seed, in millimetres. The line length is the tracking direction, so the
# radius cannot live there: two points far apart inside one vessel would be
# read as a radius larger than the lumen. An attribute on the line node keeps
# each seed's radius with that seed across scene saves.
SEED_RADIUS_ATTRIBUTE = "SeqSeg.RadiusMm"


def _normalize_loaded_volume_node(loaded):
    """Return a single volume node from slicer.util.loadVolume/loadLabelVolume (node or list)."""
    if loaded is None:
        return None
    if isinstance(loaded, (list, tuple)):
        return loaded[0] if loaded else None
    return loaded


_VOLUME_SUFFIXES = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd")


def _files_under_output(output_dir: str, suffixes: Sequence[str]):
    """Files in the output folder and one level down.

    The finished mask is written in the output folder. The per-step masks are
    written in a case subfolder. Listing only the top folder hides this run
    when it never reached the final write, and the newest older file is then
    what gets loaded.
    """
    if not output_dir or not os.path.isdir(output_dir):
        return []
    found = []
    root = output_dir.rstrip(os.sep)
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isfile(path) and name.lower().endswith(suffixes):
            found.append(path)
            continue
        if not os.path.isdir(path):
            continue
        try:
            children = os.listdir(path)
        except OSError:
            continue
        for child in children:
            child_path = os.path.join(path, child)
            if os.path.isfile(child_path) and child.lower().endswith(suffixes):
                found.append(child_path)
    return found


def _written_since(path: str, not_before: float) -> bool:
    """True when the file was written during this run.

    One second of slack covers a filesystem timestamp that rounds down. More
    than that would accept the previous run when two runs are back to back.
    """
    try:
        return os.path.getmtime(path) >= not_before - 1.0
    except OSError:
        return False


def choose_seqseg_segmentation(output_dir: str, case_name: str, not_before: float) -> Optional[str]:
    """Return the final mask this run wrote for this case, or None.

    SeqSeg keeps every previous run in the same output folder. The old picker
    loaded the newest file whose name contained "seg", so a previous volume's
    mask was shown on the image that had just been run. The finished mask is
    ``{case}_segmentation_*_steps``, written at the end of the run. Step
    masks (``binary_seg``, ``raw_seg``, ``sweep_seg``) are not that file.
    """
    if not case_name:
        return None
    prefix = f"{case_name}_segmentation_"
    written = [
        path for path in _files_under_output(output_dir, _VOLUME_SUFFIXES)
        if os.path.basename(path).startswith(prefix)
        and "_steps." in os.path.basename(path)
        and _written_since(path, not_before)
    ]
    if not written:
        return None
    written.sort(key=os.path.getmtime)
    return written[-1]


def choose_seqseg_surface(output_dir: str, case_name: str, not_before: float) -> Optional[str]:
    """Return the smoothed surface this run wrote for this case, or None.

    Same rule as the mask: a previous run's ``.vtp`` is left in the folder,
    and the newest file overall is not necessarily this case. The nonsmooth
    mesh is an intermediate; the smoothed one is ``surface_mesh`` (classic)
    or ``_surface_`` (plus), written beside the finished mask.
    """
    if not case_name:
        return None
    prefix = f"{case_name}_"
    written = []
    for path in _files_under_output(output_dir, (".vtp",)):
        name = os.path.basename(path)
        if not name.startswith(prefix) or "surface" not in name or "nonsmooth" in name:
            continue
        if not _written_since(path, not_before):
            continue
        written.append(path)
    if not written:
        return None

    def rank(path: str):
        name = os.path.basename(path)
        # surface_mesh is the smoothed classic mesh. A plain ``_surface_`` is
        # the plus mesh. Prefer the smoothed name when both were just written.
        smoothed = 1 if "surface_mesh" in name else 0
        return (smoothed, os.path.getmtime(path))

    written.sort(key=rank)
    return written[-1]


#
# seqseg
#


class seqseg(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SeqSeg Vessel Segmentation")
        self.parent.categories = [
            translate("qSlicerAbstractCoreModule", "SimVascular"),
            translate("qSlicerAbstractCoreModule", "Segmentation"),
        ]
        # Keep empty: declaring PyTorch/NNUNet here makes Slicer load them before seqseg opens; if they are missing or fail,
        # the module never loads and setupPythonRequirements cannot show install guidance (Total Segmentator uses [] too).
        self.parent.dependencies = []
        self.parent.contributors = ["Numi Sveinsson Cepero (UC Berkeley & UT Austin)"]
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
SeqSeg Vessel Segmentation is a seed-based vascular segmentation module for 3D Slicer. Choose a model, then add one seed per vessel. Each seed has its own radius in millimetres. The line on that seed is only the start and the tracking direction. The first <b>Run SeqSeg</b> installs <b>torch 2.2.2</b> and <b>nnunetv2</b> if they are missing, through the PyTorch and Slicer NNUNet extensions when those are installed and with pip otherwise, and may ask Slicer to restart. Pretrained weights for the selected model are downloaded on that run, after a confirm dialog, if they are not already in Slicer's SeqSeg folder.
See more information in <a href="https://github.com/numisveinsson/SeqSeg">SeqSeg package documentation</a>.
""")
        self.parent.acknowledgementText = _("""
This module was developed by Numi Sveinsson Cepero at UC Berkeley & UT Austin.
It integrates the SeqSeg Python package for seed-based vascular segmentation into 3D Slicer.
See the paper <a href="https://rdcu.be/ePeZD">here</a> for more details.
Cite: Sveinsson Cepero, N., Shadden, S.C. SeqSeg: Learning Local Segments for Automatic Vascular Model Construction. Ann Biomed Eng 53, 158–179 (2025). https://doi.org/10.1007/s10439-024-03611-z
                                        
                                            
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)
        
        # Check for SeqSeg package availability on module load
        self._checkDependencies()
    
    def _checkDependencies(self):
        """Check if required dependencies are available."""
        import importlib.metadata

        try:
            # Use distribution metadata — `import seqseg` resolves to this scripted module, not the pip package.
            importlib.metadata.version("seqseg")
            logging.info("SeqSeg dependency check passed")
        except importlib.metadata.PackageNotFoundError:
            logging.warning("SeqSeg package not found. It will be installed automatically when first used.")
            # Note: We don't fail here to allow the module to load,
            # dependency installation will happen when the user tries to run the algorithm


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add the CTA-cardio tutorial volume to Sample Data.

    The two previous entries pointed at thumbnails (seqseg1.png, seqseg2.png)
    that are not in Resources/Icons, so Sample Data showed a broken icon for
    volumes the tutorial never used. CTA-cardio is the walkthrough volume.
    The thumbnail is seqseg.jpg, which CMake actually installs.
    """

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="SeqSeg Vessel Segmentation",
        sampleName="CTA-cardio",
        thumbnailFileName=os.path.join(iconsPath, "seqseg.jpg"),
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/3b0d4eb1a7d8ebb0c5a89cc0504640f76a030b4e869e33ff34c564c3d3b88ad2",
        fileNames="CTA-cardio.nrrd",
        checksums="SHA256:3b0d4eb1a7d8ebb0c5a89cc0504640f76a030b4e869e33ff34c564c3d3b88ad2",
        nodeNames="CTA-cardio",
    )


#
# seqsegParameterNode
#

# Train dataset ids passed to SeqSeg / nnUNet (stored on the parameter node).
# The Train Dataset combo shows friendly labels; itemData holds these real ids.
KNOWN_TRAIN_DATASETS = (
    "Dataset005_SEQAORTANDFEMOMR",
    "Dataset006_SEQAORTANDFEMOCT",
    "Dataset002_CASXCT",
    "Dataset091_VMR_FONTAN_003MR",
    "Dataset090_VMR_FONTAN_ORIGMR",
    # Kept so a scene saved against the previous coronary archive still loads.
    # The combo does not list it; canonical_dataset_id sends it to Dataset002.
    "Dataset010_SEQCOROASOCACT",
)

# The 2026-10-30 coronary archive is Dataset002_CASXCT. The Zenodo record text
# still says Dataset010_SEQCOROASOCACT, but that folder is not in the zip, and
# SeqSeg joins the results path with this name.
LEGACY_TRAIN_DATASETS = {
    "Dataset010_SEQCOROASOCACT": "Dataset002_CASXCT",
}

# User-facing labels for the Train Dataset / weights combo (id -> display text).
TRAIN_DATASET_DISPLAY_NAMES = {
    "Dataset005_SEQAORTANDFEMOMR": "Aorta MR",
    "Dataset006_SEQAORTANDFEMOCT": "Aorta CT",
    "Dataset002_CASXCT": "Coronary CT",
    "Dataset091_VMR_FONTAN_003MR": "Fontan MR (003)",
    "Dataset090_VMR_FONTAN_ORIGMR": "Fontan MR (orig)",
}


def train_dataset_display_name(dataset_id: str) -> str:
    """Friendly label for a known train dataset id (falls back to the id)."""
    return TRAIN_DATASET_DISPLAY_NAMES.get(dataset_id, dataset_id)


def canonical_dataset_id(dataset_id: str) -> str:
    """Map a retired coronary id onto the folder in the current archive."""
    if not dataset_id:
        return dataset_id
    return LEGACY_TRAIN_DATASETS.get(dataset_id, dataset_id)


def train_dataset_id_from_ui_value(value: str) -> Optional[str]:
    """Resolve combo display text or a raw dataset id to a known train dataset id."""
    if not value:
        return None
    resolved = None
    if value in KNOWN_TRAIN_DATASETS:
        resolved = value
    else:
        for dataset_id, label in TRAIN_DATASET_DISPLAY_NAMES.items():
            if label == value:
                resolved = dataset_id
                break
    if resolved is None:
        return None
    return canonical_dataset_id(resolved)


# trained_unit is the spacing the network saw. image_unit and scale are what a
# millimetre header needs for that network. A centimetre header (spacing about
# ten times smaller) is handled by settings_for_volume, not by a second preset.
MODEL_PRESETS = {
    "Dataset005_SEQAORTANDFEMOMR": {
        "trained_unit": "cm",
        "image_unit": "mm",
        "scale": "0.1",
        "radius_mm": 10.0,
        "max_steps": 10,
    },
    "Dataset006_SEQAORTANDFEMOCT": {
        "trained_unit": "cm",
        "image_unit": "mm",
        "scale": "0.1",
        "radius_mm": 10.0,
        "max_steps": 10,
    },
    "Dataset002_CASXCT": {
        "trained_unit": "mm",
        "image_unit": "mm",
        "scale": "1",
        "radius_mm": 2.0,
        "max_steps": 10,
        # The archive ships one trainer folder, 3d_fullres. 2d looks for a
        # folder that is not there and the run stops before tracking.
        "nnunet_type": "3d_fullres",
    },
    "Dataset090_VMR_FONTAN_ORIGMR": {
        "trained_unit": "cm",
        "image_unit": "mm",
        "scale": "0.1",
        "radius_mm": 10.0,
        "max_steps": 10,
    },
    "Dataset091_VMR_FONTAN_003MR": {
        "trained_unit": "cm",
        "image_unit": "mm",
        "scale": "0.1",
        "radius_mm": 10.0,
        "max_steps": 10,
    },
}

# No Zenodo archive. Showing it beside the models that download makes Run fail
# for anyone who has not already copied that dataset folder onto disk.
LOCAL_ONLY_DATASETS = frozenset({"Dataset091_VMR_FONTAN_003MR"})


def model_preset(dataset_id: str) -> dict:
    """Preset for a known model, or a millimetre fallback for an unknown id."""
    preset = MODEL_PRESETS.get(canonical_dataset_id(dataset_id))
    if preset is not None:
        return preset
    return {
        "trained_unit": "mm",
        "image_unit": "mm",
        "scale": "1",
        "radius_mm": 5.0,
        "max_steps": 10,
    }


def header_unit_from_spacing(spacing) -> str:
    """Return 'cm' or 'mm' from the spacing numbers Slicer is holding.

    A vascular CT or MR in millimetres is about 0.4–2. The same scan stored in
    centimetres is about 0.04–0.2. The gap at 0.3 is what separates them.
    Nothing in the file format is consulted: many centimetre volumes are tagged
    as millimetres and only the magnitude gives them away. A micro-CT finer
    than 0.3 mm is classified as centimetres.
    """
    if not spacing:
        return "mm"
    values = []
    for component in spacing:
        try:
            value = abs(float(component))
        except (TypeError, ValueError):
            continue
        if value > 0:
            values.append(value)
    if not values:
        return "mm"
    if max(values) < 0.3:
        return "cm"
    return "mm"


def settings_for_volume(dataset_id: str, header_unit: str) -> dict:
    """Unit and scale for this model on a header that is already mm or cm.

    A centimetre-trained network shown a millimetre header needs scale 0.1,
    otherwise a 0.8 mm voxel is read as 0.8 cm and the vessel is missed. The
    same network on a centimetre header needs scale 1. A millimetre-trained
    network on a centimetre header needs scale 10.
    """
    preset = model_preset(dataset_id)
    trained = preset.get("trained_unit", "cm")
    header = header_unit if header_unit in ("mm", "cm") else "mm"
    if trained == header:
        scale = "1"
    elif trained == "cm" and header == "mm":
        scale = "0.1"
    else:
        scale = "10"
    settings = {
        "image_unit": header,
        "scale": scale,
        "radius_mm": float(preset["radius_mm"]),
        "max_steps": int(preset["max_steps"]),
        "trained_unit": trained,
    }
    if preset.get("nnunet_type"):
        settings["nnunet_type"] = preset["nnunet_type"]
    return settings


def radius_in_header_units(radius_mm: float, header_unit: str) -> float:
    """Radius written to seeds.json, in the same numbers as the volume spacing.

    The spin box is millimetres. SeqSeg sizes the patch as radius divided by
    spacing. A centimetre header stores 0.08 where a millimetre header stores
    0.8, so a radius of 10 against that header is ten times too wide.
    """
    if header_unit == "cm":
        return float(radius_mm) / 10.0
    return float(radius_mm)


def _format_spacing(spacing) -> str:
    if not spacing:
        return ""
    return " × ".join(f"{float(component):.3g}" for component in spacing)


def datasets_shown_in_combo(installed_ids) -> tuple:
    """Dataset ids the model combo should list.

    Fontan MR (003) stays hidden until its folder is already on disk. The
    other ids have archives and can be chosen before the download.
    """
    installed = set(installed_ids or ())
    return tuple(
        dataset_id
        for dataset_id in KNOWN_TRAIN_DATASETS
        if dataset_id not in LEGACY_TRAIN_DATASETS
        and (dataset_id not in LOCAL_ONLY_DATASETS or dataset_id in installed)
    )


def volume_hint(dataset_id: str, spacing, header_unit: str, image_unit: str, scale: str, radius_mm: float) -> str:
    """One sentence under the model. Unit and scale stay in the collapsed section.

    The sentence names the spacing and what will happen to it, so a centimetre
    volume is not described as millimetres. A hand edit that disagrees with the
    spacing is stated as a disagreement, not applied back over the edit.
    """
    label = train_dataset_display_name(dataset_id)
    expected = settings_for_volume(dataset_id, header_unit)
    trained_word = "centimetres" if expected["trained_unit"] == "cm" else "millimetres"
    header_word = "centimetres" if header_unit == "cm" else "millimetres"
    if spacing:
        lead = f"Spacing {_format_spacing(spacing)} is in {header_word}. "
    else:
        lead = f"No spacing on the volume; treating it as {header_word}. "
    if header_unit == "cm":
        sent = radius_in_header_units(radius_mm, "cm")
        lead += (
            f"New seeds start at {float(radius_mm):g} mm, sent as {sent:g}. "
            f"Each seed has its own radius."
        )
    elif expected["scale"] == "0.1":
        lead += (
            f"{label} was trained in {trained_word}, so the spacing is divided by 10 "
            f"before the network sees it. New seeds start at {float(radius_mm):g} mm, "
            f"and each seed has its own radius."
        )
    else:
        lead += (
            f"{label} was trained in millimetres, so the spacing is used as stored. "
            f"New seeds start at {float(radius_mm):g} mm, and each seed has its own radius."
        )
    if image_unit != expected["image_unit"] or scale != expected["scale"]:
        lead += (
            f" The run is set to Image Unit {image_unit} and Scale {scale}; "
            f"this spacing wants {expected['image_unit']} and {expected['scale']}."
        )
    return lead


# Aorta MR and Aorta CT are one Zenodo archive, so choosing either model installs
# both. Coronary CT and Fontan MR (orig) are separate archives. Fontan MR (003)
# has no archive: Run only proceeds when a results folder already contains it.
WEIGHT_PROFILES = (
    {
        "datasets": (
            "Dataset005_SEQAORTANDFEMOMR",
            "Dataset006_SEQAORTANDFEMOCT",
        ),
        # The archive is named nnUNet_results.zip, but the dataset folders
        # (Dataset005_…, Dataset006_…) sit at the top of the zip. There is no
        # nnUNet_results directory inside it.
        "url": "https://zenodo.org/records/15020477/files/nnUNet_results.zip?download=1",
        "zip_filename": "nnUNet_results.zip",
        "extracted_dir_name": "nnUNet_results",
        "record_url": "https://zenodo.org/records/15020477",
        "progress_label": "Downloading Aorta weights (CT/MR) (236.3 MB)...",
        "confirm_detail": (
            "Aorta CT and Aorta MR share one archive (236.3 MB). "
            "Downloading either model installs both."
        ),
    },
    {
        "datasets": ("Dataset002_CASXCT",),
        # Record 23813298 replaced 19547894. The file is also called
        # nnUNet_results.zip, the same name as the aorta archive, so the local
        # copy keeps a distinct name or the two downloads overwrite each other.
        # Inside the zip the dataset is nnUNet_results/Dataset002_CASXCT, and
        # the only trainer folder is 3d_fullres. A Dataset010 tree already on
        # disk is the previous model and is not reused.
        "url": "https://zenodo.org/records/23813298/files/nnUNet_results.zip?download=1",
        "zip_filename": "nnUNet_results_coronary.zip",
        "extracted_dir_name": "nnUNet_results_coronary",
        "record_url": "https://zenodo.org/records/23813298",
        "progress_label": "Downloading Coronary CT weights (2.2 MB)...",
        "confirm_detail": (
            "The Coronary CT archive is 2.2 MB. "
            "It contains a 3d_fullres model only."
        ),
    },
    {
        "datasets": ("Dataset090_VMR_FONTAN_ORIGMR",),
        # The zip's top folder is the dataset id, not a results bundle like
        # nnUNet_results.zip. extract_into_results_dir unpacks it under its own
        # parent so SeqSeg receives a folder that contains Dataset090_…
        "url": "https://zenodo.org/records/20056802/files/Dataset090_VMR_FONTAN_ORIGMR.zip?download=1",
        "zip_filename": "Dataset090_VMR_FONTAN_ORIGMR.zip",
        "extracted_dir_name": "nnUNet_results_fontan_orig",
        "extract_into_results_dir": True,
        "record_url": "https://zenodo.org/records/20056802",
        "progress_label": "Downloading Fontan MR (orig) weights (~10.7 MB)...",
        "confirm_detail": (
            "The Fontan MR (orig) archive is about 10.7 MB. "
            "The model was trained in cm."
        ),
    },
)

# Trees written by the old download buttons, before the path was chosen for the user.
LEGACY_WEIGHTS_DIR = os.path.join(os.path.expanduser("~"), "SeqSeg_Weights")


def weight_profile_for_dataset(dataset_id: str):
    """Return the downloadable archive that contains this dataset, or None."""
    dataset_id = canonical_dataset_id(dataset_id)
    for profile in WEIGHT_PROFILES:
        if dataset_id in profile["datasets"]:
            return profile
    return None


def dataset_weights_present(results_path: str, dataset_id: str) -> bool:
    """A results tree counts only when it contains this dataset's folder.

    Aorta and coronary unpack to different parents. Treating the parent
    directory as enough would let a coronary tree satisfy an aorta run.
    """
    if not results_path or not dataset_id:
        return False
    return os.path.isdir(os.path.join(results_path, dataset_id))


def results_dir_containing_dataset(extract_root: str, dataset_id: str) -> Optional[str]:
    """Return the folder SeqSeg should receive, or None.

    The aorta archive puts Dataset005_… and Dataset006_… directly in the
    weights folder, so that folder itself is the results path. One extra
    directory is allowed when a zip wraps the dataset, and a parent that
    merely exists is not a match.
    """
    if dataset_weights_present(extract_root, dataset_id):
        return extract_root
    if not extract_root or not os.path.isdir(extract_root):
        return None
    for name in os.listdir(extract_root):
        nested = os.path.join(extract_root, name)
        if dataset_weights_present(nested, dataset_id):
            return nested
    return None


def managed_weights_root() -> str:
    """Folder beside Slicer's user settings file. The user never picks it."""
    settings_file = slicer.app.slicerUserSettingsFilePath
    return os.path.join(os.path.dirname(settings_file), "SeqSeg", "weights")


def find_existing_weights(dataset_id: str, stored_path: str = "") -> Optional[str]:
    """Return a results folder that already contains this dataset, or None.

    New downloads live under Slicer's settings directory. A tree left in
    ~/SeqSeg_Weights by the old download buttons is reused so that archive
    is not fetched again. Both roots are searched even when the dataset has
    no archive: that is how a copied Fontan MR (003) folder becomes visible.
    A path stored on the parameter node is last, so a results tree saved
    with a scene is still found.
    """
    dataset_id = canonical_dataset_id(dataset_id)
    for root in (managed_weights_root(), LEGACY_WEIGHTS_DIR):
        found = results_dir_containing_dataset(root, dataset_id)
        if found:
            return found
    if stored_path:
        found = results_dir_containing_dataset(stored_path, dataset_id)
        if found:
            return found
    return None


def line_length(start, end) -> float:
    """Euclidean length between two 3D points."""
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    dz = float(end[2]) - float(start[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def max_branches_for_seeds(max_branches: int, n_seeds: int) -> int:
    """SeqSeg stops starting new seeds once Max Branches is reached.

    VesselTree always consumes the next JSON seed as a new branch, so a cap
    smaller than the seed count (Slicer default is 1) traces only the first
    two starts and ignores the rest.
    """
    return max(int(max_branches), int(n_seeds))


@parameterNodeWrapper
class seqsegParameterNode:
    """
    The parameters needed by SeqSeg module.

    inputVolume - The input volume for segmentation.
    outputSegmentation - The segmentation node that will contain the results.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    # Ten steps is enough to leave the seed. The old default of 1 finished before tracking started.
    maxSteps: Annotated[int, WithinRange(1, 10000)] = 10
    maxBranches: Annotated[int, WithinRange(1, 100)] = 1  # Max number of branches
    maxStepsPerBranch: Annotated[int, WithinRange(1, 1000)] = 20  # Max steps per branch
    # Millimetres first. connectGui falls back to the first Choice when a value does not bind,
    # and a centimetre default made a Slicer volume (always millimetres) run at the wrong scale.
    imageUnit: Annotated[str, Choice(["mm", "cm"])] = "mm"
    # Aorta MR is the default model and was trained in centimetres, so a millimetre volume needs 0.1.
    scale: Annotated[str, Choice(["0.1", "1", "10"])] = "0.1"
    # Lumen radius in millimetres. Not the seed-line length: that length is the tracking direction.
    radiusMm: float = 10.0
    coordinateSystem: Annotated[str, Choice(["LPS World", "RAS World"])] = "LPS World"  # Coordinate system for seed points
    nnunetResultsPath: str = ""  # Set at run from the selected model; not edited in the panel
    nnunetType: Annotated[str, Choice(["3d_fullres", "2d"])] = "3d_fullres"  # Type of nnUNet model
    trainDataset: Annotated[str, Choice(list(KNOWN_TRAIN_DATASETS))] = "Dataset005_SEQAORTANDFEMOMR"  # Training dataset name
    outputDirectory: str = ""  # Directory for SeqSeg outputs (data_dir)
    outputSegmentation: str = ""  # Segmentation node ID
    simvascular: bool = False  # SeqSeg -simvascular 0/1
    forceCpu: bool = False  # SeqSeg -cpu (run nnUNet inference on the CPU)


class SeqSegCancelled(Exception):
    """The user cancelled the running SeqSeg process."""


class _SeqSegRunResult:
    """stdout, stderr, and exit code from the seqseg process. Same fields subprocess.run returned."""

    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class WeightsDownloadCancelled(Exception):
    """The user declined the weights download, so the run stops before torch or SeqSeg."""


class WeightsDownloadError(Exception):
    """Raised when a pretrained-weights download or extract fails."""


class InstallError(Exception):
    """Raised when dependency installation fails or requires user action (aligned with common Slicer AI-module patterns)."""

    def __init__(self, message, restartRequired=False):
        super().__init__(message)
        self.message = message
        self.restartRequired = restartRequired

    def __str__(self):
        return self.message


#
# seqsegWidget
#


class seqsegWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self._seedRows = []
        self._clearingSeeds = False
        self._seqsegRunning = False
        self._headerAppliedKey = None
        self._applyingHeader = False

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)
        if hasattr(self, "reloadCollapsibleButton"):
            self.reloadCollapsibleButton.collapsed = True

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/seqseg.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = seqsegLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        if hasattr(self.ui, "addSeedButton"):
            self.ui.addSeedButton.connect("clicked(bool)", self.onAddSeedButton)
        if hasattr(self.ui, "clearSeedsButton"):
            self.ui.clearSeedsButton.connect("clicked(bool)", self.onClearSeedsButton)

        # One reload path. Run already loads the segmentation and the surface.
        # The old Load Segmentation button created a second node beside that result.
        if hasattr(self.ui, 'browseOutputsButton'):
            self.ui.browseOutputsButton.connect("clicked(bool)", self.onBrowseOutputsButton)
        
        # A path line edit rather than a ctkDirectoryButton: the button asks the layout for the
        # full width of its path, so a long output folder widened the whole module panel.
        # connectGui binds ctkPathLineEdit only to pathlib.Path parameters, so this one is synced
        # by hand, in _syncUiWithParameterNode and here.
        self.ui.outputDirectoryPathLineEdit.connect('currentPathChanged(QString)', self.onOutputDirectoryChanged)

        # Connect image unit combo box to sync with parameter node
        if hasattr(self.ui, 'unitComboBox') and hasattr(self.ui.unitComboBox, 'currentTextChanged'):
            self.ui.unitComboBox.currentTextChanged.connect(self.onImageUnitChanged)
            logging.info("Connected image unit combo box signal")
        else:
            logging.warning("Could not connect image unit combo box signal")
        
        # Connect nnUNet type combo box to sync with parameter node
        if hasattr(self.ui, 'nnunetTypeComboBox') and hasattr(self.ui.nnunetTypeComboBox, 'currentTextChanged'):
            self.ui.nnunetTypeComboBox.currentTextChanged.connect(self.onNnUnetTypeChanged)
            logging.info("Connected nnUNet type combo box signal")
        else:
            logging.warning("Could not connect nnUNet type combo box signal")

        # Connect train dataset combo box to sync with parameter node
        if hasattr(self.ui, 'trainDatasetComboBox') and hasattr(self.ui.trainDatasetComboBox, 'currentTextChanged'):
            self.ui.trainDatasetComboBox.currentTextChanged.connect(self.onTrainDatasetChanged)
            logging.info("Connected train dataset combo box signal")
        else:
            logging.warning("Could not connect train dataset combo box signal")

        if hasattr(self.ui, "scaleComboBox") and hasattr(self.ui.scaleComboBox, "currentTextChanged"):
            self.ui.scaleComboBox.currentTextChanged.connect(self.onScaleChanged)
            logging.info("Connected scale combo box signal")
        else:
            logging.warning("Could not connect scale combo box signal")

        if hasattr(self.ui, "radiusSpinBox"):
            try:
                self.ui.radiusSpinBox.connect("valueChanged(double)", self.onRadiusChanged)
                logging.info("Connected radiusSpinBox")
            except Exception as e:
                logging.warning(f"Could not connect radiusSpinBox: {e}")

        if hasattr(self.ui, "inputSelector"):
            try:
                self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputVolumeChanged)
                logging.info("Connected inputSelector")
            except Exception as e:
                logging.warning(f"Could not connect inputSelector: {e}")

        if hasattr(self.ui, "outputSegmentationSelector"):
            try:
                self.ui.outputSegmentationSelector.connect(
                    "currentNodeChanged(vtkMRMLNode*)", self.onOutputSegmentationChanged
                )
                logging.info("Connected outputSegmentationSelector")
            except Exception as e:
                logging.warning(f"Could not connect outputSegmentationSelector: {e}")

        # QSpinBox values are not always pushed to the parameter node by connectGui; sync explicitly.
        for spin_name, handler, label in (
            ("maxStepsSpinBox", self.onMaxStepsChanged, "Nr. of Steps"),
            ("maxBranchesSpinBox", self.onMaxBranchesChanged, "Max branches"),
            ("maxStepsPerBranchSpinBox", self.onMaxStepsPerBranchChanged, "Max steps per branch"),
        ):
            if hasattr(self.ui, spin_name):
                spin = getattr(self.ui, spin_name)
                try:
                    spin.connect("valueChanged(int)", handler)
                    logging.info(f"Connected {spin_name} ({label})")
                except Exception as e:
                    logging.warning(f"Could not connect {spin_name}: {e}")

        for box_name, handler in (
            ("simvascularCheckBox", self.onSimvascularChanged),
            ("forceCpuCheckBox", self.onForceCpuChanged),
        ):
            if hasattr(self.ui, box_name):
                try:
                    getattr(self.ui, box_name).connect("toggled(bool)", handler)
                    logging.info(f"Connected {box_name}")
                except Exception as e:
                    logging.warning(f"Could not connect {box_name}: {e}")
            else:
                logging.warning(f"{box_name} not found in UI")

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        
        # Initialize status updates
        self._updateStatusMessage("Ready to run vessel segmentation")

    def _updateStatusMessage(self, message: str, isError: bool = False) -> None:
        """Update the status label with current operation status."""
        if hasattr(self.ui, 'statusLabel'):
            self.ui.statusLabel.text = message
            if isError:
                self.ui.statusLabel.setStyleSheet("""
                    QLabel {
                        color: #D32F2F;
                        font-style: italic;
                        font-weight: bold;
                        padding: 5px;
                    }
                """)
            else:
                self.ui.statusLabel.setStyleSheet("""
                    QLabel {
                        color: #2E7D32;
                        font-style: italic;
                        padding: 5px;
                    }
                """)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self._clearSeedRows(delete_nodes=False)
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        self.initializeParameterNode()
        self.refreshSeedRowsFromScene()
        self._fitWrappingLabels()
        self._checkCanApply()

    def _fitWrappingLabels(self) -> None:
        """Give word-wrapped hints the height of the text at the panel width.

        A wrapping label reports a one-line size hint. The seed rows are then
        placed on the following lines, so they cover "Each seed is…".
        """
        if not getattr(self, "ui", None):
            return
        for name in ("seedsHintLabel", "presetHintLabel"):
            label = getattr(self.ui, name, None)
            if label is None or not hasattr(label, "heightForWidth"):
                continue
            width = label.width
            if width < 80:
                parent = label.parentWidget()
                width = parent.width if parent is not None else 0
            if width < 80:
                width = 260
            needed = int(label.heightForWidth(width))
            if needed > 0:
                label.minimumHeight = needed

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

        # Set default output directory if not set, before the sync so the field shows it
        if not self._parameterNode.outputDirectory:
            default_output_dir = os.path.join(os.path.expanduser("~"), "SeqSeg_Output")
            # Ensure directory ends with '/'
            if not default_output_dir.endswith(os.sep):
                default_output_dir = default_output_dir + os.sep
            self._parameterNode.outputDirectory = default_output_dir
            logging.info(f"Using default output directory: {default_output_dir}")

        # Sync UI with parameter node values
        self._syncUiWithParameterNode()
        # Match Image Unit and Scale to this volume once per volume/model pair.
        # A later hand edit of those controls is left alone until the volume or
        # the model changes, which is what the sentence under Model describes.
        self.applyHeaderToCurrentModel(reset_radius=False)

    def _installedLocalDatasetIds(self) -> tuple:
        """Local-only models whose dataset folder is already on disk."""
        stored = ""
        if self._parameterNode and self._parameterNode.nnunetResultsPath:
            stored = self._parameterNode.nnunetResultsPath
        return tuple(
            dataset_id
            for dataset_id in LOCAL_ONLY_DATASETS
            if find_existing_weights(dataset_id, stored)
        )

    def _populateTrainDatasetComboFromKnown(self) -> None:
        """Rebuild the model combo with friendly labels; itemData stores real dataset ids.

        The parameter node still stores Dataset*_… ids (Choice / SeqSeg -train_dataset).
        The combo is synced manually (no SlicerParameterName) so users see labels like
        "Aorta CT" while runtime still receives Dataset006_SEQAORTANDFEMOCT.
        Fontan MR (003) is omitted until its folder is on disk: it has no archive,
        and choosing it otherwise stops the run.
        """
        if not getattr(self, "ui", None) or not hasattr(self.ui, "trainDatasetComboBox"):
            return
        combo = self.ui.trainDatasetComboBox
        expected = list(datasets_shown_in_combo(self._installedLocalDatasetIds()))
        current = [combo.itemData(i) for i in range(combo.count)]
        if current == expected:
            return
        combo.blockSignals(True)
        combo.clear()
        for ds in expected:
            combo.addItem(train_dataset_display_name(ds), ds)
        combo.blockSignals(False)

    def _trainDatasetIdFromCombo(self, combo=None) -> Optional[str]:
        """Read the real train dataset id from the combo (itemData, then display text)."""
        if combo is None:
            if not getattr(self, "ui", None) or not hasattr(self.ui, "trainDatasetComboBox"):
                return None
            combo = self.ui.trainDatasetComboBox
        data = combo.currentData
        if isinstance(data, str) and data in KNOWN_TRAIN_DATASETS:
            return data
        return train_dataset_id_from_ui_value(combo.currentText)

    def _setTrainDatasetComboToId(self, dataset_id: str) -> None:
        """Select the combo row whose itemData matches dataset_id."""
        if not getattr(self, "ui", None) or not hasattr(self.ui, "trainDatasetComboBox"):
            return
        if not dataset_id:
            return
        combo = self.ui.trainDatasetComboBox
        for index in range(combo.count):
            if combo.itemData(index) == dataset_id:
                if combo.currentIndex != index:
                    combo.setCurrentIndex(index)
                return
        label = train_dataset_display_name(dataset_id)
        index = combo.findText(label)
        if index < 0:
            index = combo.findText(dataset_id)
        if index >= 0 and combo.currentIndex != index:
            combo.setCurrentIndex(index)

    def setParameterNode(self, inputParameterNode: Optional[seqsegParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self._populateTrainDatasetComboFromKnown()
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            # connectGui rebuilds Choice combo boxes (clear + add first item). Extra
            # currentTextChanged handlers in setup() would then overwrite the stored
            # parameter with the first Choice value unless widget signals are blocked.
            blocked = []
            for widget in self._widgetsWithManualParameterHandlers():
                blocked.append((widget, widget.signalsBlocked()))
                widget.blockSignals(True)
            try:
                self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            finally:
                for widget, was_blocked in blocked:
                    widget.blockSignals(was_blocked)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()
        else:
            self._clearSeedRows(delete_nodes=False)

    def _widgetsWithManualParameterHandlers(self):
        """Widgets whose setup() signals write to the parameter node outside connectGui."""
        names = (
            "unitComboBox",
            "nnunetTypeComboBox",
            "trainDatasetComboBox",
            "scaleComboBox",
            "maxStepsSpinBox",
            "maxBranchesSpinBox",
            "maxStepsPerBranchSpinBox",
            "simvascularCheckBox",
            "forceCpuCheckBox",
        )
        widgets = []
        ui = getattr(self, "ui", None)
        if ui is None:
            return widgets
        for name in names:
            if hasattr(ui, name):
                widgets.append(getattr(ui, name))
        return widgets

    def _checkCanApply(self, caller=None, event=None) -> None:
        """Check if we can run SeqSeg and update UI accordingly."""
        # The run pumps the Qt event loop so Cancel works. Parameter writes during
        # that run emit ModifiedEvent, and this observer would otherwise re-enable
        # the button and start a second seqseg process.
        if getattr(self, "_seqsegRunning", False):
            if hasattr(self.ui, "applyButton"):
                self.ui.applyButton.enabled = False
                self.ui.applyButton.text = "Running..."
            return
        if not self._parameterNode:
            if hasattr(self.ui, 'applyButton'):
                self.ui.applyButton.enabled = False
            self._updateStatusMessage("Parameter node not initialized", isError=True)
            return
            
        # Check requirements and update status
        missing_requirements = []
        
        if not self._parameterNode.inputVolume:
            missing_requirements.append("Input Volume")

        if not self._completeSeedLines():
            missing_requirements.append("Seed line (Add Seed, then place start and direction)")
        
        self._showRaisedMaxBranches()

        # The missing list goes to the status line, which wraps, not onto the button,
        # whose label width becomes the module panel's minimum.
        if missing_requirements:
            self.ui.applyButton.text = "Run SeqSeg"
            self.ui.applyButton.enabled = False
            self._updateStatusMessage(f"Please provide: {', '.join(missing_requirements)}")
        else:
            self.ui.applyButton.text = "Run SeqSeg"
            self.ui.applyButton.enabled = True
            self._updateStatusMessage("Ready for vessel segmentation.")

    def _showRaisedMaxBranches(self) -> None:
        """Show the branch cap SeqSeg will actually use.

        VesselTree stops starting seeds once Max Branches is reached, and the
        spin box default is 1, so a second seed line was ignored while the
        panel still said 1. Raising the spin box is the same rule as
        max_branches_for_seeds, visible before Run.
        """
        if not self._parameterNode or not hasattr(self.ui, "maxBranchesSpinBox"):
            return
        raised = max_branches_for_seeds(
            int(self._parameterNode.maxBranches),
            len(self._completeSeedLines()),
        )
        spin = self.ui.maxBranchesSpinBox
        if int(spin.value) != raised:
            spin.setValue(raised)

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        
        # Update UI for processing
        self._seqsegRunning = True
        self.ui.applyButton.enabled = False
        self.ui.applyButton.text = "Running..."
        self._updateStatusMessage("Starting SeqSeg processing...")
        weights_status = None
        weights_error = False

        try:
            inputVolume = self._parameterNode.inputVolume
            if not inputVolume:
                logging.warning("No input volume selected, trying to find one...")
                firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
                if firstVolumeNode:
                    inputVolume = firstVolumeNode
                    self._parameterNode.inputVolume = firstVolumeNode
                    logging.info(f"Using volume: {firstVolumeNode.GetName()}")
                else:
                    raise ValueError("No volume found in scene. Please load a volume first.")

            if hasattr(self.ui, "radiusSpinBox"):
                try:
                    self._parameterNode.radiusMm = float(self.ui.radiusSpinBox.value)
                except Exception as e:
                    logging.warning(f"Could not read radius from UI: {e}")

            seedPairs = self._collectSeedPairs()
            if not seedPairs:
                raise ValueError("Add at least one complete seed line (start and direction points).")
            logging.info("Collected %s complete seed line(s) for SeqSeg", len(seedPairs))
            seedPoint1Node = None
            seedPoint2Node = None
            radiusEstimate = seedPairs[0]["radius"]
            maxSteps = self._parameterNode.maxSteps
            maxBranches = self._parameterNode.maxBranches
            maxStepsPerBranch = self._parameterNode.maxStepsPerBranch
            imageUnit = self._parameterNode.imageUnit
            nnunetType = self._parameterNode.nnunetType
            trainDataset = self._parameterNode.trainDataset

            # Fallback: Try to get nnUNet type directly from UI if parameter node has default/empty value
            if hasattr(self.ui, 'nnunetTypeComboBox'):
                try:
                    ui_nnunet_type = self.ui.nnunetTypeComboBox.currentText
                    if ui_nnunet_type and ui_nnunet_type != nnunetType:
                        logging.info(f"nnUNet Type mismatch - Parameter: '{nnunetType}', UI: '{ui_nnunet_type}'")
                        if ui_nnunet_type in ["3d_fullres", "2d"]:
                            nnunetType = ui_nnunet_type
                            self._parameterNode.nnunetType = ui_nnunet_type
                            logging.info(f"Updated nnUNet Type from UI: '{ui_nnunet_type}'")
                except Exception as e:
                    logging.warning(f"Could not read nnUNet type from UI: {e}")

            if hasattr(self.ui, "trainDatasetComboBox"):
                try:
                    ui_train_ds = self._trainDatasetIdFromCombo()
                    # A dataset that is not in the combo (Fontan MR (003) with no
                    # weights on disk) must not be replaced by whatever row the
                    # combo fell back to, or Run would download a different model.
                    combo_ids = [
                        self.ui.trainDatasetComboBox.itemData(i)
                        for i in range(self.ui.trainDatasetComboBox.count)
                    ]
                    if (
                        ui_train_ds
                        and ui_train_ds != trainDataset
                        and trainDataset in combo_ids
                        and ui_train_ds in KNOWN_TRAIN_DATASETS
                    ):
                        logging.info(f"Train Dataset mismatch - Parameter: '{trainDataset}', UI: '{ui_train_ds}'")
                        trainDataset = ui_train_ds
                        self._parameterNode.trainDataset = ui_train_ds
                        logging.info(f"Updated Train Dataset from UI: '{ui_train_ds}'")
                except Exception as e:
                    logging.warning(f"Could not read train dataset from UI: {e}")

            # Spin boxes: read current UI values before run (connectGui may not update the parameter node).
            if hasattr(self.ui, "maxStepsSpinBox"):
                try:
                    maxSteps = int(self.ui.maxStepsSpinBox.value)
                    self._parameterNode.maxSteps = maxSteps
                except Exception as e:
                    logging.warning(f"Could not read Nr. of Steps from UI: {e}")
            if hasattr(self.ui, "maxBranchesSpinBox"):
                try:
                    maxBranches = int(self.ui.maxBranchesSpinBox.value)
                    self._parameterNode.maxBranches = maxBranches
                except Exception as e:
                    logging.warning(f"Could not read max branches from UI: {e}")
            if hasattr(self.ui, "maxStepsPerBranchSpinBox"):
                try:
                    maxStepsPerBranch = int(self.ui.maxStepsPerBranchSpinBox.value)
                    self._parameterNode.maxStepsPerBranch = maxStepsPerBranch
                except Exception as e:
                    logging.warning(f"Could not read max steps per branch from UI: {e}")

            # Scale: read from UI before run (connectGui may not update the parameter node).
            if hasattr(self.ui, "scaleComboBox"):
                try:
                    ui_scale = self.ui.scaleComboBox.currentText
                    if ui_scale in ("0.1", "1", "10"):
                        self._parameterNode.scale = ui_scale
                    elif ui_scale:
                        logging.warning(f"Scale '{ui_scale}' not in allowed list; keeping parameter node value")
                except Exception as e:
                    logging.warning(f"Could not read Scale from UI: {e}")

            if hasattr(self.ui, "simvascularCheckBox"):
                try:
                    self._parameterNode.simvascular = bool(self.ui.simvascularCheckBox.checked)
                except Exception as e:
                    logging.warning(f"Could not read SimVascular toggle from UI: {e}")

            if hasattr(self.ui, "forceCpuCheckBox"):
                try:
                    self._parameterNode.forceCpu = bool(self.ui.forceCpuCheckBox.checked)
                except Exception as e:
                    logging.warning(f"Could not read Force CPU toggle from UI: {e}")

            simvascular = bool(self._parameterNode.simvascular)
            forceCpu = bool(self._parameterNode.forceCpu)

            # Debug: Log image unit with extra details
            logging.info(f"Image Unit from parameter node: '{imageUnit}'")
            
            # Validate image unit - should be automatically synced via parameter binding
            if imageUnit not in ["mm", "cm"]:
                logging.warning(f"Invalid image unit '{imageUnit}', defaulting to 'mm'")
                imageUnit = "mm"
                self._parameterNode.imageUnit = "mm"
            
            logging.info(f"nnUNet Type (final): '{nnunetType}'")
            logging.info(f"Train Dataset: '{trainDataset}'")
            logging.info(f"Nr. of Steps: {maxSteps}, Max branches: {maxBranches}, Max steps/branch: {maxStepsPerBranch}")

            # Resolve weights before torch install and SeqSeg. A cancel stops here.
            self._updateStatusMessage("Checking model weights...")
            nnunetResultsPath = self.ensureWeightsForDataset(trainDataset)
            logging.info(f"nnUNet Results Path: '{nnunetResultsPath}'")
            outputDirectory = self._parameterNode.outputDirectory
            logging.info(f"Output directory from parameter node: '{outputDirectory}'")
            
            # If output directory is empty, try to get it from the UI directly
            if not outputDirectory:
                outputDirectory = self.ui.outputDirectoryPathLineEdit.currentPath
                logging.info(f"Output directory from UI field: '{outputDirectory}'")
                if outputDirectory:
                    # Ensure directory ends with '/'
                    if not outputDirectory.endswith(os.sep):
                        outputDirectory = outputDirectory + os.sep
                    self._parameterNode.outputDirectory = outputDirectory
            
            # Final check - if still empty, use default
            if not outputDirectory:
                default_output_dir = os.path.join(os.path.expanduser("~"), "SeqSeg_Output")
                outputDirectory = default_output_dir
                self._parameterNode.outputDirectory = default_output_dir
                logging.info(f"Using default output directory: {default_output_dir}")
            
            # Ensure output directory ends with '/'
            if outputDirectory and not outputDirectory.endswith(os.sep):
                outputDirectory = outputDirectory + os.sep
            
            outputSegmentationNode = self._outputSegmentationNode(create=True)
            
            # Run SeqSeg segmentation
            self._updateStatusMessage("Running SeqSeg algorithm...")
            self.logic.runSeqSeg(inputVolume, seedPoint1Node, seedPoint2Node, radiusEstimate, 
                               maxSteps, maxBranches, maxStepsPerBranch,
                               imageUnit, self._parameterNode.scale, self._parameterNode.coordinateSystem, nnunetResultsPath, nnunetType, trainDataset, 
                               outputDirectory, outputSegmentationNode, simvascular=simvascular,
                               forceCpu=forceCpu, seedPairs=seedPairs)

            if getattr(self.logic, "lastRunLoadedSurface", False):
                finished = _(
                    "SeqSeg finished. The segmentation is on the volume in the slice views, and the surface mesh is in the scene."
                )
            else:
                finished = _(
                    "SeqSeg finished. The segmentation is on the volume in the slice views. SeqSeg did not write a surface mesh."
                )
            # _checkCanApply in finally replaces the status line. Keep the
            # finished sentence the same way a failure is written back.
            weights_status = finished

        except SeqSegCancelled as e:
            logging.info("SeqSeg run cancelled: %s", e)
            weights_status = str(e)
        except WeightsDownloadCancelled as e:
            logging.info("SeqSeg run cancelled: %s", e)
            weights_status = str(e)
        except WeightsDownloadError as e:
            logging.error("Failed to download weights: %s", e)
            weights_status = _("Failed to download model weights.")
            weights_error = True
            slicer.util.errorDisplay(
                _("Failed to download model weights."),
                detailedText=str(e),
            )
        except InstallError as e:
            logging.error(f"SeqSeg dependency installation failed: {e}")
            self._updateStatusMessage(f"SeqSeg dependencies: {str(e)}", isError=True)
            if e.restartRequired:
                self._updateStatusMessage(_("Application restart required to finish Python package setup."), isError=False)
                if slicer.util.confirmOkCancelDisplay(
                    _("Application restart is required to complete installation of required Python packages.\nPress OK to restart."),
                    _("Confirm application restart"),
                    detailedText=str(e),
                ):
                    slicer.util.restart()
            else:
                slicer.util.errorDisplay(_("SeqSeg dependency installation failed."), detailedText=str(e))
        except Exception as e:
            logging.error(f"SeqSeg processing failed: {e}")
            # The status line is one sentence. stdout and stderr are in the dialog,
            # because a failed SeqSeg often prints the reason on stdout and leaves stderr empty.
            weights_status = _("SeqSeg failed.")
            weights_error = True
            slicer.util.errorDisplay(_("SeqSeg failed."), detailedText=str(e))

        finally:
            # Reset button state. _checkCanApply replaces the status line, so a
            # weights cancel or failure is written again afterwards.
            self._seqsegRunning = False
            self.ui.applyButton.enabled = True
            self._checkCanApply()
            if weights_status:
                self._updateStatusMessage(weights_status, isError=weights_error)

    def onOutputDirectoryChanged(self, directory: str) -> None:
        """Called when the output directory field changes - sync with parameter node."""
        if self._parameterNode and directory:
            # Ensure directory ends with '/'
            if not directory.endswith(os.sep):
                directory = directory + os.sep
            self._parameterNode.outputDirectory = directory
            logging.info(f"Output directory updated to: {directory}")

    def onImageUnitChanged(self, unit: str) -> None:
        """Called when image unit combo box changes - sync with parameter node."""
        if self._parameterNode and unit:
            self._parameterNode.imageUnit = unit
            logging.info(f"Image Unit updated to: {unit}")
            self.refreshSeedStatusLabels()
            if not self._applyingHeader:
                self._updatePresetHint()

    def onNnUnetTypeChanged(self, nnunet_type: str) -> None:
        """Called when nnUNet type combo box changes - sync with parameter node."""
        if self._parameterNode and nnunet_type:
            self._parameterNode.nnunetType = nnunet_type
            logging.info(f"nnUNet Type updated to: {nnunet_type}")

    def onTrainDatasetChanged(self, train_dataset: str) -> None:
        """Map the combo label to a dataset id, and apply that model's preset.

        The preset is applied only when the id actually changes. Syncing the
        combo from a saved parameter node fires this too, and reapplying then
        would wipe a unit or radius the user had edited.
        """
        if not self._parameterNode:
            return
        dataset_id = self._trainDatasetIdFromCombo()
        if not dataset_id:
            dataset_id = train_dataset_id_from_ui_value(train_dataset)
        if not dataset_id or dataset_id not in KNOWN_TRAIN_DATASETS:
            return
        previous = self._parameterNode.trainDataset
        self._parameterNode.trainDataset = dataset_id
        logging.info(
            f"Train Dataset updated to: {dataset_id} "
            f"({train_dataset_display_name(dataset_id)})"
        )
        if dataset_id != previous:
            self.applyModelPreset(dataset_id)
        else:
            self._updatePresetHint(dataset_id)

    def applyModelPreset(self, dataset_id: str) -> None:
        """Set radius and steps for the model, and unit and scale for its header.

        The aorta and Fontan networks were trained in centimetres. Leaving
        scale at 1 with a millimetre header feeds them spacing ten times too
        coarse, and the segmentation misses the vessel. A centimetre header
        is the opposite case and must not receive that millimetre scale.
        """
        self._headerAppliedKey = None
        self.applyHeaderToCurrentModel(reset_radius=True)

    def onInputVolumeChanged(self, node=None) -> None:
        """A new volume can be centimetres while the previous one was millimetres."""
        if self._applyingHeader:
            return
        self.applyHeaderToCurrentModel(reset_radius=False)

    def _volumeSpacing(self):
        volume = self._parameterNode.inputVolume if self._parameterNode else None
        if volume is None or not hasattr(volume, "GetSpacing"):
            return None
        try:
            spacing = volume.GetSpacing()
            return (float(spacing[0]), float(spacing[1]), float(spacing[2]))
        except Exception:
            logging.debug("Could not read volume spacing", exc_info=True)
            return None

    def _headerUnit(self) -> str:
        return header_unit_from_spacing(self._volumeSpacing())

    def _volumeModelKey(self):
        volume = self._parameterNode.inputVolume if self._parameterNode else None
        volume_id = ""
        if volume is not None and hasattr(volume, "GetID"):
            try:
                volume_id = volume.GetID() or ""
            except Exception:
                volume_id = ""
        dataset_id = self._parameterNode.trainDataset if self._parameterNode else ""
        return (volume_id, dataset_id, self._volumeSpacing())

    def applyHeaderToCurrentModel(self, reset_radius: bool = False) -> None:
        """Set Image Unit and Scale from the volume spacing and the selected model.

        Applied when the volume or the model changes. A hand edit of Image Unit
        or Scale after that is kept: the sentence under Model says when that
        edit no longer matches the spacing.
        """
        if not self._parameterNode or self._applyingHeader:
            return
        dataset_id = self._parameterNode.trainDataset
        if not dataset_id:
            self._updatePresetHint()
            return
        key = self._volumeModelKey()
        if key == self._headerAppliedKey and not reset_radius:
            self._updatePresetHint(dataset_id)
            return
        settings = settings_for_volume(dataset_id, self._headerUnit())
        self._applyingHeader = True
        try:
            self._parameterNode.imageUnit = settings["image_unit"]
            self._parameterNode.scale = settings["scale"]
            if reset_radius:
                self._parameterNode.maxSteps = int(settings["max_steps"])
                if settings.get("nnunet_type"):
                    self._parameterNode.nnunetType = settings["nnunet_type"]
                # radiusMm is the default for a new seed. Seeds the user has
                # already given a different radius keep it; ones still on the
                # previous default follow the new model.
                previous_default = self._radiusMm()
                new_default = float(settings["radius_mm"])
                stored = [
                    (node, self._storedSeedRadiusMm(node))
                    for node in self._seedLineNodesFromScene()
                ]
                self._parameterNode.radiusMm = new_default
                for node, current in stored:
                    if current is None or abs(current - previous_default) < 1e-3:
                        self._setSeedRadiusMm(node, new_default)
            self._headerAppliedKey = key
            logging.info(
                "Matched %s to a %s header: unit %s, scale %s",
                train_dataset_display_name(dataset_id),
                settings["image_unit"],
                settings["image_unit"],
                settings["scale"],
            )
            self._syncUiWithParameterNode()
            self.refreshSeedStatusLabels()
        finally:
            self._applyingHeader = False
        self._updatePresetHint(dataset_id)

    def _updatePresetHint(self, dataset_id: str = "") -> None:
        if not getattr(self, "ui", None) or not hasattr(self.ui, "presetHintLabel"):
            return
        if not self._parameterNode:
            return
        if not dataset_id:
            dataset_id = self._parameterNode.trainDataset
        if not dataset_id:
            return
        self.ui.presetHintLabel.text = volume_hint(
            dataset_id,
            self._volumeSpacing(),
            self._headerUnit(),
            self._parameterNode.imageUnit,
            self._parameterNode.scale,
            self._radiusMm(),
        )

    def onScaleChanged(self, scale: str) -> None:
        """Called when Scale combo box changes - sync with parameter node."""
        if self._parameterNode and scale in ("0.1", "1", "10"):
            self._parameterNode.scale = scale
            logging.info(f"Scale updated to: {scale}")
            if not self._applyingHeader:
                self._updatePresetHint()

    def onRadiusChanged(self, value: float) -> None:
        if self._parameterNode is not None:
            self._parameterNode.radiusMm = float(value)
            if not self._applyingHeader:
                self._updatePresetHint()

    def onOutputSegmentationChanged(self, node) -> None:
        """Remember the segmentation Run should fill.

        The parameter is a node id string, not a node reference, so connectGui
        cannot bind the combo. An unwired combo used to ignore the node the
        user created and load the result into a different one.
        """
        if self._parameterNode is None:
            return
        self._parameterNode.outputSegmentation = node.GetID() if node is not None else ""

    def onMaxStepsChanged(self, value: int) -> None:
        if self._parameterNode is not None:
            self._parameterNode.maxSteps = int(value)

    def onMaxBranchesChanged(self, value: int) -> None:
        if self._parameterNode is not None:
            self._parameterNode.maxBranches = int(value)

    def onMaxStepsPerBranchChanged(self, value: int) -> None:
        if self._parameterNode is not None:
            self._parameterNode.maxStepsPerBranch = int(value)

    def onSimvascularChanged(self, checked: bool) -> None:
        if self._parameterNode is not None:
            self._parameterNode.simvascular = bool(checked)

    def onForceCpuChanged(self, checked: bool) -> None:
        if self._parameterNode is not None:
            self._parameterNode.forceCpu = bool(checked)

    def _syncUiWithParameterNode(self) -> None:
        """Manually sync UI controls with parameter node values."""
        if not self._parameterNode:
            return

        # Output directory field (not bound by connectGui; see setup)
        if self._parameterNode.outputDirectory:
            self.ui.outputDirectoryPathLineEdit.currentPath = self._parameterNode.outputDirectory

        # Sync image unit combo box (backup for parameter binding)
        if hasattr(self.ui, 'unitComboBox') and self._parameterNode.imageUnit:
            try:
                current_unit = self._parameterNode.imageUnit
                combo_box = self.ui.unitComboBox
                # Only update if different to avoid circular updates
                if combo_box.currentText != current_unit:
                    index = combo_box.findText(current_unit)
                    if index >= 0:
                        combo_box.setCurrentIndex(index)
                        logging.info(f"Synced image unit to UI: {current_unit}")
                    else:
                        logging.warning(f"Image unit '{current_unit}' not found in combo box options")
            except Exception as e:
                logging.warning(f"Could not sync image unit to UI: {e}")
        
        # Sync nnUNet type combo box (backup for parameter binding)
        if hasattr(self.ui, 'nnunetTypeComboBox') and self._parameterNode.nnunetType:
            try:
                current_type = self._parameterNode.nnunetType
                combo_box = self.ui.nnunetTypeComboBox
                # Only update if different to avoid circular updates
                if combo_box.currentText != current_type:
                    index = combo_box.findText(current_type)
                    if index >= 0:
                        combo_box.setCurrentIndex(index)
                        logging.info(f"Synced nnUNet type to UI: {current_type}")
                    else:
                        logging.warning(f"nnUNet type '{current_type}' not found in combo box options")
            except Exception as e:
                logging.warning(f"Could not sync nnUNet type to UI: {e}")

        # Sync train dataset combo box (friendly labels; parameter stores real ids)
        if hasattr(self.ui, 'trainDatasetComboBox') and self._parameterNode.trainDataset:
            try:
                current_ds = canonical_dataset_id(self._parameterNode.trainDataset)
                if current_ds != self._parameterNode.trainDataset:
                    self._parameterNode.trainDataset = current_ds
                combo_box = self.ui.trainDatasetComboBox
                selected_id = self._trainDatasetIdFromCombo(combo_box)
                if selected_id != current_ds:
                    self._setTrainDatasetComboToId(current_ds)
                    if self._trainDatasetIdFromCombo(combo_box) == current_ds:
                        logging.info(
                            f"Synced train dataset to UI: {current_ds} "
                            f"({train_dataset_display_name(current_ds)})"
                        )
                    else:
                        logging.warning(
                            f"Train dataset '{current_ds}' not in combo options; "
                            f"parameter kept, UI may not match until you pick a listed dataset."
                        )
            except Exception as e:
                logging.warning(f"Could not sync train dataset to UI: {e}")

        # Sync scale combo box (backup for parameter binding)
        if hasattr(self.ui, "scaleComboBox") and self._parameterNode.scale:
            try:
                current_scale = self._parameterNode.scale
                combo_box = self.ui.scaleComboBox
                if combo_box.currentText != current_scale:
                    index = combo_box.findText(current_scale)
                    if index >= 0:
                        combo_box.setCurrentIndex(index)
                        logging.info(f"Synced scale to UI: {current_scale}")
                    else:
                        logging.warning(f"Scale '{current_scale}' not found in combo box options")
            except Exception as e:
                logging.warning(f"Could not sync scale to UI: {e}")

        # Sync integer spin boxes when parameter node was restored or updated in code
        for attr, spin_name in (
            ("maxSteps", "maxStepsSpinBox"),
            ("maxBranches", "maxBranchesSpinBox"),
            ("maxStepsPerBranch", "maxStepsPerBranchSpinBox"),
        ):
            if not hasattr(self.ui, spin_name):
                continue
            try:
                want = int(getattr(self._parameterNode, attr))
                spin = getattr(self.ui, spin_name)
                if int(spin.value) != want:
                    spin.setValue(want)
                    logging.info(f"Synced {attr} to UI: {want}")
            except Exception as e:
                logging.warning(f"Could not sync {attr} to UI: {e}")

        for box_name, attr in (
            ("simvascularCheckBox", "simvascular"),
            ("forceCpuCheckBox", "forceCpu"),
        ):
            if not hasattr(self.ui, box_name):
                continue
            try:
                want = bool(getattr(self._parameterNode, attr))
                checkbox = getattr(self.ui, box_name)
                if bool(checkbox.checked) != want:
                    checkbox.setChecked(want)
                    logging.info(f"Synced {attr} to UI: {want}")
            except Exception as e:
                logging.warning(f"Could not sync {attr} to UI: {e}")

        if hasattr(self.ui, "radiusSpinBox"):
            try:
                want = float(getattr(self._parameterNode, "radiusMm", 10.0) or 10.0)
                spin = self.ui.radiusSpinBox
                if abs(float(spin.value) - want) > 1e-6:
                    spin.setValue(want)
                    logging.info(f"Synced radiusMm to UI: {want}")
            except Exception as e:
                logging.warning(f"Could not sync radius to UI: {e}")

        self._syncOutputSegmentationCombo()
        dataset_id = self._parameterNode.trainDataset if self._parameterNode else ""
        self._updatePresetHint(dataset_id)

    def _syncOutputSegmentationCombo(self) -> None:
        selector = getattr(self.ui, "outputSegmentationSelector", None)
        if selector is None or not self._parameterNode:
            return
        node = None
        node_id = self._parameterNode.outputSegmentation
        if node_id:
            node = slicer.mrmlScene.GetNodeByID(node_id)
        try:
            selector.blockSignals(True)
            selector.setCurrentNode(node)
        except Exception as e:
            logging.warning(f"Could not sync output segmentation to UI: {e}")
        finally:
            selector.blockSignals(False)

    def _outputSegmentationNode(self, create: bool = False):
        """Return the segmentation Run fills, creating it when the combo is empty.

        noneEnabled is on so the combo can sit on "Create on run". Creating the
        node here, and selecting it, is what makes the next run replace this
        result instead of adding SeqSeg_Segmentation_* beside it.
        """
        selector = getattr(self.ui, "outputSegmentationSelector", None)
        node = None
        if selector is not None:
            try:
                node = selector.currentNode()
            except Exception:
                node = None
        if node is None and self._parameterNode and self._parameterNode.outputSegmentation:
            node = slicer.mrmlScene.GetNodeByID(self._parameterNode.outputSegmentation)
        if node is None and create:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "SeqSeg Output")
            node.CreateDefaultDisplayNodes()
            volume = self._parameterNode.inputVolume if self._parameterNode else None
            if volume is not None:
                # Geometry string alone does not make the Segmentations module
                # treat this node as belonging to the image, and a segmentation
                # left at the scene root is a separate object.
                node.SetReferenceImageGeometryParameterFromVolumeNode(volume)
                self.logic.parentNodeUnderVolume(node, volume)
        if node is not None and self._parameterNode:
            self._parameterNode.outputSegmentation = node.GetID()
            self._syncOutputSegmentationCombo()
        return node

    def _radiusMm(self) -> float:
        """Default radius in millimetres for a newly added seed."""
        if self._parameterNode is not None:
            try:
                radius = float(self._parameterNode.radiusMm)
                if radius > 0:
                    return radius
            except (TypeError, ValueError):
                pass
        return 10.0

    def _storedSeedRadiusMm(self, node):
        """Radius stored on this seed, or None when the node has never had one."""
        if node is None or not hasattr(node, "GetAttribute"):
            return None
        raw = node.GetAttribute(SEED_RADIUS_ATTRIBUTE)
        if not raw:
            return None
        try:
            radius = float(raw)
        except (TypeError, ValueError):
            return None
        if radius <= 0:
            return None
        return radius

    def _setSeedRadiusMm(self, node, radius_mm: float) -> None:
        if node is None or not hasattr(node, "SetAttribute"):
            return
        node.SetAttribute(SEED_RADIUS_ATTRIBUTE, f"{float(radius_mm):.6g}")

    def _seedRadiusMm(self, node) -> float:
        """This seed's radius in millimetres.

        Missing attributes take the current default and are written onto the
        node, so a seed created before per-seed radii still has one value of
        its own rather than sharing later edits to the default.
        """
        stored = self._storedSeedRadiusMm(node)
        if stored is not None:
            return stored
        fallback = self._radiusMm()
        self._setSeedRadiusMm(node, fallback)
        return fallback

    def onSeedRadiusChanged(self, node, value: float) -> None:
        if self._clearingSeeds or self._applyingHeader:
            return
        self._setSeedRadiusMm(node, value)

    def onAddSeedButton(self, _checked: bool = False) -> None:
        """Create a tagged seed line, add a row, and enter placement for start then direction."""
        node = self.createSeedLine()
        self._addSeedRowForNode(node, activate=True)
        self._checkCanApply()

    def onClearSeedsButton(self, _checked: bool = False) -> None:
        """Remove all SeqSeg seed lines from the scene and the widget list."""
        self._clearSeedRows(delete_nodes=True)
        self._updateStatusMessage("Cleared seed lines.")
        self._checkCanApply()

    def _stopPlacingSeeds(self) -> None:
        """Leave Place mode so deleting the active markup line does not crash Slicer."""
        try:
            interaction_node = slicer.app.applicationLogic().GetInteractionNode()
            if interaction_node is not None:
                if hasattr(interaction_node, "SetPlaceModePersistence"):
                    interaction_node.SetPlaceModePersistence(0)
                view_transform = getattr(slicer.vtkMRMLInteractionNode, "ViewTransform", None)
                if view_transform is not None and hasattr(interaction_node, "SetCurrentInteractionMode"):
                    interaction_node.SetCurrentInteractionMode(view_transform)
            selection_node = slicer.app.applicationLogic().GetSelectionNode()
            if selection_node is not None:
                if hasattr(selection_node, "SetActivePlaceNodeID"):
                    selection_node.SetActivePlaceNodeID("")
                if hasattr(selection_node, "SetReferenceActivePlaceNodeID"):
                    selection_node.SetReferenceActivePlaceNodeID("")
        except Exception:
            logging.debug("Could not leave seed placement mode", exc_info=True)

    def _markupPointRAS(self, node, index: int = 0):
        if not node or node.GetNumberOfControlPoints() <= index:
            return None
        ras = [0.0, 0.0, 0.0]
        node.GetNthControlPointPosition(index, ras)
        return [float(ras[0]), float(ras[1]), float(ras[2])]

    def createSeedLine(self):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
        node.CreateDefaultDisplayNodes()
        node.SetName(slicer.mrmlScene.GenerateUniqueName("Seed"))
        node.SetAttribute(SEED_LINE_ATTRIBUTE, "true")
        self._setSeedRadiusMm(node, self._radiusMm())
        if hasattr(node, "SetMaximumNumberOfControlPoints"):
            node.SetMaximumNumberOfControlPoints(2)
        display = node.GetDisplayNode()
        if display:
            display.SetGlyphScale(3.0)
            display.SetTextScale(2.0)
            # Slice intersections make the start and the direction visible in the
            # 2D views. Without them the line exists only in 3D and is easy to
            # place off the lumen.
            if hasattr(display, "SetVisibility2D"):
                display.SetVisibility2D(True)
        return node

    def _addSeedRowForNode(self, node, activate: bool = False):
        row = qt.QWidget()
        row_layout = qt.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        name_label = qt.QLabel(node.GetName() if node else "Seed")
        row_layout.addWidget(name_label)

        radius_spin = qt.QDoubleSpinBox()
        radius_spin.setDecimals(1)
        radius_spin.setMinimum(0.1)
        radius_spin.setMaximum(100.0)
        radius_spin.setSingleStep(0.5)
        radius_spin.setSuffix(" mm")
        radius_spin.setMaximumWidth(96)
        radius_spin.setToolTip(
            "Radius of this seed in millimetres. The two clicks set the start and "
            "the tracking direction, not this radius. Each seed keeps its own value."
        )
        radius_spin.blockSignals(True)
        radius_spin.setValue(self._seedRadiusMm(node))
        radius_spin.blockSignals(False)
        radius_spin.connect(
            "valueChanged(double)",
            lambda value, node=node: self.onSeedRadiusChanged(node, value),
        )
        row_layout.addWidget(radius_spin)

        radius_label = qt.QLabel("Click the start inside the vessel")
        # Ignored width: the sentence is longer than the module panel. Its
        # minimum width otherwise widens the row until it is drawn on top of
        # the hint above it.
        radius_label.setWordWrap(True)
        radius_label.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred))
        row_layout.addWidget(radius_label, 1)

        reposition_button = qt.QPushButton("Reposition")
        reposition_button.setToolTip("Clear this line and click the start, then a point along the vessel")
        reposition_button.connect(
            "clicked(bool)",
            lambda _checked=False, node=node: self.repositionSeed(node),
        )
        row_layout.addWidget(reposition_button)

        remove_button = qt.QPushButton("X")
        remove_button.setFixedWidth(28)
        remove_button.flat = True
        remove_button.setStyleSheet(
            "QPushButton { color: #b00020; font-weight: bold; border: none; }"
            "QPushButton:hover { background-color: #f8d7da; }"
        )
        remove_button.toolTip = "Delete seed"
        remove_button.connect(
            "clicked(bool)",
            lambda _checked=False, row=row: self.removeSeedRow(row),
        )
        row_layout.addWidget(remove_button)

        seed_row = {
            "row": row,
            "node": node,
            "name_label": name_label,
            "radius_label": radius_label,
            "radius_spin": radius_spin,
            "observers": [],
        }
        self._seedRows.append(seed_row)
        layout = self._seedRowsLayout()
        if layout is not None:
            layout.addWidget(row)
        self._fitWrappingLabels()
        self.observeSeedNode(seed_row)
        self.refreshSeedRowStatus(seed_row)
        if activate:
            self.activateLinePlacement(node)
            self._updateStatusMessage(
                "Click the start inside the vessel, then a short step along it. "
                "Set this seed's radius on its row; the line does not set it."
            )
        return seed_row

    def _seedRowsLayout(self):
        if getattr(self, "ui", None) and hasattr(self.ui, "seedRowsWidget"):
            return self.ui.seedRowsWidget.layout()
        return None

    def removeSeedRow(self, row) -> None:
        if self._clearingSeeds:
            return
        removed_rows = [seed_row for seed_row in self._seedRows if seed_row["row"] is row]
        self._stopPlacingSeeds()
        for seed_row in removed_rows:
            self.removeSeedNodeObservers(seed_row)
            node = seed_row["node"]
            seed_row["node"] = None
            if node:
                try:
                    slicer.mrmlScene.RemoveNode(node)
                except Exception:
                    logging.debug("Could not remove seed line node", exc_info=True)
        self._seedRows = [seed_row for seed_row in self._seedRows if seed_row["row"] is not row]
        row.deleteLater()
        self._checkCanApply()

    def _nodeInScene(self, node) -> bool:
        if node is None:
            return False
        try:
            if hasattr(node, "GetScene"):
                return node.GetScene() is not None
            if hasattr(node, "GetID") and hasattr(slicer.mrmlScene, "GetNodeByID"):
                node_id = node.GetID()
                return bool(node_id and slicer.mrmlScene.GetNodeByID(node_id))
        except Exception:
            return False
        return True

    def _seedLineNodesFromScene(self):
        try:
            nodes = slicer.util.getNodesByClass("vtkMRMLMarkupsLineNode")
        except Exception:
            nodes = []
            try:
                collection = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsLineNode")
                collection.InitTraversal()
                node = collection.GetNextItemAsObject()
                while node:
                    nodes.append(node)
                    node = collection.GetNextItemAsObject()
            except Exception:
                pass
        return [node for node in nodes if node.GetAttribute(SEED_LINE_ATTRIBUTE) == "true"]

    def _nodeKey(self, node):
        try:
            return node.GetID() or id(node)
        except Exception:
            return id(node)

    def _clearSeedRows(self, delete_nodes: bool = False) -> None:
        self._stopPlacingSeeds()
        self._clearingSeeds = True
        rows = list(self._seedRows)
        self._seedRows = []
        try:
            for seed_row in rows:
                self.removeSeedNodeObservers(seed_row)
                node = seed_row.get("node")
                seed_row["node"] = None
                if delete_nodes and node:
                    try:
                        slicer.mrmlScene.RemoveNode(node)
                    except Exception:
                        logging.debug("Could not remove seed line node", exc_info=True)
                widget = seed_row.get("row")
                if widget is not None:
                    # deleteLater leaves the row in the layout until the event
                    # loop runs. Run holds that loop inside the progress dialog,
                    # so a rebuild showed every seed twice until the run ended.
                    layout = self._seedRowsLayout()
                    if layout is not None:
                        layout.removeWidget(widget)
                    widget.setParent(None)
                    widget.deleteLater()
        finally:
            self._clearingSeeds = False

    def refreshSeedRowsFromScene(self) -> None:
        if not getattr(self, "ui", None) or not hasattr(self.ui, "seedRowsWidget"):
            return
        live_nodes = []
        seen = set()
        for seed_row in self._seedRows:
            node = seed_row["node"]
            if self._nodeInScene(node):
                if hasattr(node, "SetAttribute"):
                    node.SetAttribute(SEED_LINE_ATTRIBUTE, "true")
                key = self._nodeKey(node)
                if key not in seen:
                    live_nodes.append(node)
                    seen.add(key)
        for node in self._seedLineNodesFromScene():
            if self._nodeInScene(node):
                key = self._nodeKey(node)
                if key not in seen:
                    live_nodes.append(node)
                    seen.add(key)

        # Rebuilding when the lines are already listed creates a second row
        # for each seed. The old row is only removed on a later event-loop
        # turn, which during Run is when the progress dialog closes.
        row_nodes = [seed_row.get("node") for seed_row in self._seedRows]
        if (
            row_nodes
            and all(self._nodeInScene(node) for node in row_nodes)
            and [self._nodeKey(node) for node in row_nodes] == [self._nodeKey(node) for node in live_nodes]
        ):
            self.refreshSeedStatusLabels()
            return

        self._clearSeedRows(delete_nodes=False)
        for node in live_nodes:
            self._addSeedRowForNode(node, activate=False)
        self.refreshSeedStatusLabels()

    def refreshSeedStatusLabels(self) -> None:
        for seed_row in self._seedRows:
            self.refreshSeedRowStatus(seed_row)

    def refreshSeedRowStatus(self, seed_row) -> None:
        if self._clearingSeeds:
            return
        node = seed_row.get("node")
        radius_label = seed_row.get("radius_label")
        if radius_label is None:
            return
        # The line is start plus tracking direction. Its length is not the radius;
        # labelling that length with Image Unit turned a 10 mm click into "10 cm".
        count = self._seedLinePointCount(node) if node and self._nodeInScene(node) else 0
        if count < 1:
            radius_label.text = "Click the start inside the vessel"
        elif count < 2:
            radius_label.text = "Click a point along the vessel"
        else:
            radius_label.text = "Start and direction set"
        radius_spin = seed_row.get("radius_spin")
        if radius_spin is not None and node and self._nodeInScene(node):
            want = self._seedRadiusMm(node)
            if abs(float(radius_spin.value) - want) > 1e-3:
                blocked = radius_spin.blockSignals(True)
                radius_spin.setValue(want)
                radius_spin.blockSignals(blocked)
        name_label = seed_row.get("name_label")
        if name_label is not None and node:
            name_label.setText(node.GetName())

    def _markupsEvent(self, event_name):
        markups_node = getattr(slicer, "vtkMRMLMarkupsNode", None)
        if markups_node is None:
            return None
        return getattr(markups_node, event_name, None)

    def observeSeedNode(self, seed_row) -> None:
        node = seed_row["node"]
        if not node or not hasattr(node, "AddObserver"):
            return
        event_names = (
            "PointPositionDefinedEvent",
            "PointModifiedEvent",
            "PointEndInteractionEvent",
            "PointRemovedEvent",
        )
        for event_name in event_names:
            event_id = self._markupsEvent(event_name)
            if event_id is None:
                continue
            observer_tag = node.AddObserver(
                event_id,
                lambda _caller, _event, row=seed_row: self._onSeedLineModified(row),
            )
            seed_row["observers"].append((node, observer_tag))

    def _onSeedLineModified(self, seed_row) -> None:
        if self._clearingSeeds or seed_row not in self._seedRows:
            return
        self.refreshSeedRowStatus(seed_row)
        self._checkCanApply()

    def removeSeedNodeObservers(self, seed_row) -> None:
        for node, observer_tag in seed_row.get("observers", []):
            try:
                node.RemoveObserver(observer_tag)
            except Exception:
                pass
        seed_row["observers"] = []

    def repositionSeed(self, node) -> None:
        if node is None:
            return
        try:
            node.RemoveAllControlPoints()
        except Exception:
            pass
        self.refreshSeedStatusLabels()
        self.activateLinePlacement(node)
        self._updateStatusMessage(f"Reposition {node.GetName()}: click the start, then a point along the vessel.")
        self._checkCanApply()

    def activateLinePlacement(self, node) -> None:
        if node is None:
            return
        try:
            slicer.modules.markups.logic().SetActiveListID(node)
        except Exception:
            selection_node = slicer.app.applicationLogic().GetSelectionNode()
            if selection_node is not None:
                if hasattr(selection_node, "SetReferenceActivePlaceNodeClassName"):
                    selection_node.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsLineNode")
                if hasattr(selection_node, "SetReferenceActivePlaceNodeID"):
                    selection_node.SetReferenceActivePlaceNodeID(node.GetID())
                elif hasattr(selection_node, "SetActivePlaceNodeID"):
                    selection_node.SetActivePlaceNodeID(node.GetID())

        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        if hasattr(interaction_node, "SetPlaceModePersistence"):
            interaction_node.SetPlaceModePersistence(0)
        if hasattr(interaction_node, "SetCurrentInteractionMode"):
            interaction_node.SetCurrentInteractionMode(slicer.vtkMRMLInteractionNode.Place)
        elif hasattr(interaction_node, "SwitchToPersistentPlaceMode"):
            interaction_node.SwitchToPersistentPlaceMode()

    def _seedLinePointCount(self, node) -> int:
        if node is None:
            return 0
        if hasattr(node, "GetNumberOfDefinedControlPoints"):
            return int(node.GetNumberOfDefinedControlPoints())
        if hasattr(node, "GetNumberOfControlPoints"):
            return int(node.GetNumberOfControlPoints())
        return 0

    def _completeSeedLines(self):
        """Complete SeqSeg lines from the widget list and any tagged scene nodes."""
        complete = []
        seen = set()

        def add_if_complete(node) -> None:
            if not node or not self._nodeInScene(node):
                return
            if self._seedLinePointCount(node) < 2:
                return
            key = self._nodeKey(node)
            if key in seen:
                return
            seen.add(key)
            complete.append(node)

        for seed_row in self._seedRows:
            add_if_complete(seed_row.get("node"))
        for node in self._seedLineNodesFromScene():
            add_if_complete(node)
        return complete

    def _collectSeedPairs(self):
        """Return start/direction/radius dicts from complete SeqSeg seed lines.

        Each seed carries its own radius, in millimetres, on the line node.
        That value is converted into the volume's spacing units. The line
        length is only the direction the tracker walks. Sharing one radius
        across seeds made a narrow branch use the aorta's radius.
        """
        header_unit = self._headerUnit()
        pairs = []
        for node in self._completeSeedLines():
            start = self._markupPointRAS(node, 0)
            direction = self._markupPointRAS(node, 1)
            if start is None or direction is None:
                logging.warning("Skipping incomplete seed line %s", node.GetName())
                continue
            radius_mm = self._seedRadiusMm(node)
            radius = radius_in_header_units(radius_mm, header_unit)
            if header_unit == "cm":
                logging.info(
                    "Seed %s spacing is in centimetres. Radius %s mm is written as %s.",
                    node.GetName(),
                    radius_mm,
                    radius,
                )
            pairs.append({
                "start": start,
                "direction": direction,
                "radius": radius,
            })
        return pairs

    def ensureWeightsForDataset(self, dataset_id: str) -> str:
        """Return the nnUNet results folder for this model, downloading it if needed.

        Missing pretrained archives are fetched into Slicer's settings directory
        after a confirm dialog. Cancelling raises WeightsDownloadCancelled so the
        run stops before torch is installed or SeqSeg starts. Fontan MR (003)
        has no archive: the run continues only when a stored results folder
        already contains that dataset directory. A download does not change the
        model the user picked.
        """
        stored = ""
        if self._parameterNode and self._parameterNode.nnunetResultsPath:
            stored = self._parameterNode.nnunetResultsPath
        existing = find_existing_weights(dataset_id, stored)
        if existing:
            self._parameterNode.nnunetResultsPath = existing
            logging.info(
                "Using %s weights at %s",
                train_dataset_display_name(dataset_id),
                existing,
            )
            return existing

        profile = weight_profile_for_dataset(dataset_id)
        label = train_dataset_display_name(dataset_id)
        if profile is None:
            raise ValueError(
                f"{label} weights are not installed, and this model has no automatic download."
            )

        accepted = slicer.util.confirmYesNoDisplay(
            (
                f"{label} weights are not installed.\n\n"
                f"{profile['confirm_detail']}\n\n"
                "Download them now? They are saved in Slicer's SeqSeg folder and used automatically."
            ),
            windowTitle="Download model weights",
        )
        if not accepted:
            raise WeightsDownloadCancelled(f"{label} weights were not downloaded.")

        extracted = self._download_weight_profile(profile)
        # The selected dataset must be inside the archive. A parent directory
        # alone is what used to let the wrong tree through.
        results_path = results_dir_containing_dataset(extracted, dataset_id)
        if not results_path:
            raise WeightsDownloadError(
                f"Downloaded weights at {extracted} do not contain {dataset_id}.\n\n"
                f"You can manually download from:\n{profile['record_url']}"
            )
        self._parameterNode.nnunetResultsPath = results_path
        return results_path

    def _download_weight_profile(self, profile: dict) -> str:
        """Download one Zenodo archive into Slicer's SeqSeg weights folder.

        The caller has already confirmed. There is no directory dialog: the
        archive unpacks beside Slicer's user settings, and SeqSeg is pointed
        at the extracted tree.
        """
        import urllib.request
        import zipfile

        download_dir = managed_weights_root()
        os.makedirs(download_dir, exist_ok=True)
        weights_path = os.path.join(download_dir, profile["extracted_dir_name"])
        zip_file = os.path.join(download_dir, profile["zip_filename"])
        # Aorta and coronary archives put dataset folders at the top of the zip,
        # so they unpack into the weights folder. Requiring a child named
        # nnUNet_results made that successful extract look like a failure.
        # Fontan MR (orig) is unpacked into its own results folder.
        extract_target = weights_path if profile.get("extract_into_results_dir") else download_dir
        if profile.get("extract_into_results_dir"):
            os.makedirs(weights_path, exist_ok=True)
        progressBar = None
        try:
            progressBar = slicer.util.createProgressDialog(
                labelText=profile["progress_label"],
                maximum=0,
                windowTitle="Downloading Weights",
            )
            slicer.app.processEvents()
            logging.info("Downloading weights from %s to %s", profile["url"], zip_file)

            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    progressBar.setValue(percent)
                    progressBar.setLabelText(f"Downloading... {percent}%")
                    slicer.app.processEvents()

            urllib.request.urlretrieve(profile["url"], zip_file, show_progress)

            progressBar.setLabelText("Extracting weights...")
            slicer.app.processEvents()
            logging.info("Extracting %s to %s", zip_file, extract_target)
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(extract_target)
        except Exception as e:
            raise WeightsDownloadError(
                f"Failed to download model weights:\n{e}\n\n"
                f"You can manually download from:\n{profile['record_url']}"
            ) from e
        finally:
            if progressBar is not None:
                progressBar.close()

        if os.path.exists(zip_file):
            try:
                os.remove(zip_file)
            except OSError as e:
                logging.warning("Could not remove weights archive %s: %s", zip_file, e)

        if not os.path.isdir(extract_target):
            raise WeightsDownloadError(
                f"Extraction failed: {extract_target} not found.\n\n"
                f"You can manually download from:\n{profile['record_url']}"
            )
        logging.info("Downloaded weights to %s", extract_target)
        return extract_target

    def onBrowseOutputsButton(self) -> None:
        """Load one output file the user picks.

        The previous dialog listed the files and then loaded all of them, and
        the segmentation path created a new node instead of filling the one
        from the run. A surface is a model; a segmentation refills the output
        node, same as Run.
        """
        try:
            output_dir = self._parameterNode.outputDirectory if self._parameterNode else ""
            output_subdir = os.path.join(output_dir, "output") if output_dir else ""
            if not output_subdir or not os.path.isdir(output_subdir):
                slicer.util.warningDisplay("No SeqSeg output found. Run SeqSeg first, or check the output directory.")
                return
            selected = qt.QFileDialog.getOpenFileName(
                None,
                "Load SeqSeg output",
                output_subdir,
                "SeqSeg outputs (*.mha *.nii.gz *.vtp *.nii)",
            )
            if isinstance(selected, (tuple, list)):
                selected = selected[0] if selected else ""
            path = str(selected or "")
            if not path:
                return
            if path.endswith(".vtp"):
                node = self.logic.loadSurfaceModel(path)
                if node is None:
                    slicer.util.errorDisplay("Failed to load surface mesh.")
                return
            node = self._outputSegmentationNode(create=True)
            volume = self._parameterNode.inputVolume if self._parameterNode else None
            if not self.logic.loadSegmentationToNode(path, node, volume):
                slicer.util.errorDisplay("Failed to load segmentation file.")
                return
            if volume is not None:
                self.logic.showSegmentationOverVolume(node, volume)
        except Exception as e:
            logging.error(f"Error loading SeqSeg output: {e}")
            slicer.util.errorDisplay(f"Error loading SeqSeg output: {e}")



#
# seqsegLogic
#


class seqsegLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        self.lastRunLoadedSurface = False

    def getParameterNode(self):
        return seqsegParameterNode(super().getParameterNode())

    # 2.1.1 is the first release whose device selection honors -cpu instead of always taking an
    # unusable CUDA device. Installed with --no-deps then selective deps (TotalSegmentator-style).
    # Newer PyPI releases are installed automatically when setupPythonRequirements() runs.
    SEQSEG_PYTHON_PACKAGE_NAME = "seqseg"
    MINIMUM_SEQSEG_VERSION = "2.1.1"

    def pipInstallSelective(self, packageToInstall, installCommand, packagesToSkip, recursive=False, _seen=None):
        """Install a Python distribution without deps, strip skipped Requires-Dist lines, then pip-install remaining requires.

        ``recursive`` walks the whole tree. nnunetv2 needs that: its dependencies
        declare torch themselves, and a one-level install would let pip replace
        the build installed for this module. The seqseg install stays one level.
        """
        if _seen is None:
            _seen = set()
        seenKey = packageToInstall.lower().replace("_", "-")
        if seenKey in _seen:
            return []
        _seen.add(seenKey)

        self._pipInstall(f"{installCommand} --no-deps")
        skippedRequirements = []

        import importlib.metadata

        metadataPath = [p for p in importlib.metadata.files(packageToInstall) if "METADATA" in str(p)][0]

        filteredMetadata = ""
        with open(metadataPath.locate(), "r+", encoding="latin1") as file:
            for line in file:
                skipThisPackage = False
                requirementPrefix = "Requires-Dist: "
                if line.startswith(requirementPrefix):
                    for packageToSkip in packagesToSkip:
                        if packageToSkip in line:
                            skipThisPackage = True
                            break
                if skipThisPackage:
                    skippedRequirements.append(line.removeprefix(requirementPrefix))
                    continue
                filteredMetadata += line
            file.seek(0)
            file.write(filteredMetadata)
            file.truncate()

        requirements = importlib.metadata.requires(packageToInstall)
        if requirements is None:
            return skippedRequirements

        from packaging.requirements import Requirement

        for requirement in requirements:
            skipThisPackage = False
            for packageToSkip in packagesToSkip:
                if requirement.startswith(packageToSkip):
                    skipThisPackage = True
                    break

            if skipThisPackage:
                logging.info(_("Skipping dependency (provided by Slicer or installed separately): {requirement}").format(requirement=requirement))
            elif not self._markerApplies(requirement):
                # Linux CUDA wheels and dev extras are requirements for some other
                # install. Evaluating the marker here keeps them out of this Slicer.
                logging.info(_("Skipping dependency (marker does not apply): {requirement}").format(requirement=requirement))
            elif self._requirementSatisfied(requirement):
                logging.info(_("Dependency already satisfied: {requirement}").format(requirement=requirement))
            elif recursive:
                logging.info(_("Installing {package} dependency: {requirement}").format(package=packageToInstall, requirement=requirement))
                try:
                    dependencyName = Requirement(requirement).name
                except Exception:
                    self._pipInstall(requirement)
                else:
                    self.pipInstallSelective(
                        dependencyName, requirement, packagesToSkip, recursive=True, _seen=_seen
                    )
            else:
                logging.info(_("Installing {package} dependency: {requirement}").format(package=packageToInstall, requirement=requirement))
                self._pipInstall(requirement)

        return skippedRequirements

    def _pipRequirementArg(self, requirement):
        """One argument for ``slicer.util.pip_install``.

        That helper shlex-splits its string. The old rewriter deleted the space
        after ``;``, so ``numexpr>=2.14.1; platform_machine != "wasm32"`` became
        ``numexpr>=2.14.1;platform_machine != "wasm32"``. shlex then split on the
        space before ``!=``, and pip parsed a marker whose whole text was
        ``platform_machine``.
        """
        import shlex

        spec, sep, flags = requirement.partition(" --")
        if any(ch in spec for ch in ' ;"\'()'):
            spec = shlex.quote(spec)
        return f"{spec}{sep}{flags}" if sep else spec

    def _numpyConstraintFile(self):
        """A pip constraint that keeps NumPy on 1.x.

        Written once. Every later install is run with ``-c`` this file, so a
        dependency that asks for NumPy 2 cannot replace the downgrade and
        force another restart.
        """
        path = getattr(self, "_numpyConstraintPath", None)
        if path and os.path.isfile(path):
            return path
        import tempfile

        path = os.path.join(tempfile.gettempdir(), "slicer-seqseg-numpy-lt2.txt")
        with open(path, "w", encoding="utf8") as handle:
            handle.write("numpy<2\n")
        self._numpyConstraintPath = path
        return path

    def _pipInstall(self, requirement):
        """Install with pip, refusing NumPy 2.

        torch 2.2.2 cannot import against NumPy 2. imagecodecs and a new
        scipy will otherwise resolve NumPy 2 after the downgrade. The run
        then fails, and the next Run downgrades NumPy and asks for a second
        restart.
        """
        import shlex

        arg = self._pipRequirementArg(requirement)
        spec = requirement.partition(" --")[0]
        from packaging.requirements import Requirement

        pin_numpy = True
        try:
            pin_numpy = Requirement(spec).name.lower().replace("_", "-") != "numpy"
        except Exception:
            pin_numpy = not spec.lower().startswith("numpy")
        if pin_numpy:
            arg = f"{arg} -c {shlex.quote(self._numpyConstraintFile())}"
        slicer.util.pip_install(arg)

    def _markerApplies(self, requirement):
        """False when a Requires-Dist marker excludes this interpreter.

        ``extra == "dev"`` and ``platform_system == "Linux"`` are requirements
        for some other install, not missing packages.
        """
        from packaging.requirements import Requirement

        try:
            req = Requirement(requirement)
        except Exception:
            return True
        if req.marker is None:
            return True
        return bool(req.marker.evaluate())

    def _installedDistributionVersion(self, packageName):
        import importlib.metadata

        try:
            return importlib.metadata.version(packageName)
        except importlib.metadata.PackageNotFoundError:
            return None

    def _requirementSatisfied(self, requirement):
        """True when this environment already meets a Requires-Dist line.

        A marker that does not apply counts as satisfied, so the installer
        does not fetch a package this Slicer will never import.
        """
        from packaging.requirements import Requirement
        from packaging.version import InvalidVersion
        from packaging.version import parse

        try:
            req = Requirement(requirement)
        except Exception:
            return False
        if req.marker is not None and not req.marker.evaluate():
            return True
        installed = self._installedDistributionVersion(req.name)
        if installed is None:
            return False
        if not req.specifier:
            return True
        try:
            return parse(installed) in req.specifier
        except InvalidVersion:
            return False

    def _slicerTesting(self):
        """True under ``slicer_add_python_unittest``.

        That run must not download torch: the test only checks that a missing
        install raises, and a PyTorch wheel is hundreds of megabytes.
        """
        try:
            return bool(slicer.app.testingEnabled())
        except Exception:
            return False

    def _installTorchWithoutExtension(self, requiredTorchVersion, requiredTorchVisionVersion):
        """Install the pinned torch wheel without the PyTorch extension.

        ``pip install torch==2.2.2`` takes whatever wheel is the default on
        PyPI. On Linux and Windows that wheel often has no CUDA kernels for
        the GPU in the machine, and nnU-Net then reports the failure as an
        out-of-memory error. light-the-torch is what PyTorchUtils uses to
        pick a wheel from the driver version instead.
        """
        import importlib

        # The Slicer fork's setuptools-scm version is not in the zip. Without
        # this, pip fails before light-the-torch is importable.
        os.environ["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LIGHT_THE_TORCH"] = "0.8.0"
        try:
            import light_the_torch._patch  # noqa: F401
        except ImportError:
            logging.info(_("Installing light-the-torch so PyTorch matches this machine's GPU..."))
            slicer.util.pip_install("https://github.com/Slicer/light-the-torch/archive/main.zip")
            importlib.invalidate_caches()
            import light_the_torch._patch  # noqa: F401

        # torch 2.2.2 is built against NumPy 1.x. Without the pin, resolving
        # the wheel can pull NumPy 2 and the next import dies with _ARRAY_API not found.
        args = [
            "install",
            f"torch=={requiredTorchVersion}",
            f"torchvision=={requiredTorchVisionVersion}",
            "numpy<2",
        ]
        logging.info("Install PyTorch using light-the-torch with arguments: %s", args)
        execute = getattr(slicer.util, "executePythonModule", None) or getattr(slicer.util, "_executePythonModule", None)
        if execute is None:
            raise InstallError(_("This Slicer build cannot run light-the-torch (no executePythonModule)."))
        execute("light_the_torch", args)
        importlib.invalidate_caches()

    def _nnunetPackagesToSkip(self):
        """Packages pip must not replace while installing nnunetv2.

        Slicer's SimpleITK has its own IO factory, torch 2.2.2 cannot import
        against NumPy 2, and replacing requests forces a Slicer restart.
        """
        packagesToSkip = [
            "SimpleITK",
            "torch",
            "torchvision",
            "numpy",
            "requests",
        ]
        if sys.version_info < (3, 12):
            # acvl-utils 0.2.1 and 0.2.2 fail on the Python 3.9 Slicer still
            # ships (5.8 and earlier). Installed afterwards at 0.2.0.
            packagesToSkip.extend(["acvl-utils", "acvl_utils"])
        if sys.platform == "darwin":
            packagesToSkip.append("imagecodecs")
        return packagesToSkip

    def _dependencyTreeCovered(self, packageName, packagesToSkip, seen=None):
        """False when a failed install left a package without a dependency.

        nnunetv2 can already be at the required version while numexpr, which
        blosc2 asks for, never got installed: the marker line crashed pip
        before that requirement was reached. The version check would then
        skip the tree, and the next Run would not try again.
        """
        if seen is None:
            seen = set()
        seenKey = packageName.lower().replace("_", "-")
        if seenKey in seen:
            return True
        seen.add(seenKey)
        if self._installedDistributionVersion(packageName) is None:
            return False
        import importlib.metadata
        from packaging.requirements import Requirement

        try:
            requirements = importlib.metadata.requires(packageName) or []
        except importlib.metadata.PackageNotFoundError:
            return False
        for requirement in requirements:
            if any(requirement.startswith(packageToSkip) for packageToSkip in packagesToSkip):
                continue
            if not self._markerApplies(requirement):
                continue
            if not self._requirementSatisfied(requirement):
                return False
            try:
                dependencyName = Requirement(requirement).name
            except Exception:
                continue
            if not self._dependencyTreeCovered(dependencyName, packagesToSkip, seen):
                return False
        return True

    def _installNnunetWithoutExtension(self, nnunetRequirement):
        """Install nnunetv2 without the NNUNet extension."""
        packagesToSkip = self._nnunetPackagesToSkip()
        if sys.platform == "darwin":
            self._installImagecodecsForDarwin()

        logging.info(_("Installing nnunetv2 with pip ({spec})...").format(spec=nnunetRequirement))
        self.pipInstallSelective("nnunetv2", nnunetRequirement, packagesToSkip, recursive=True)
        if sys.version_info < (3, 12):
            self._installWorkingAcvlUtils()

    def _installImagecodecsForDarwin(self):
        """imagecodecs wheels after 2026.3.6 are arm64-only.

        An x86_64 Slicer, including one running under Rosetta, would otherwise
        try to compile imagecodecs from source when nnunetv2 is installed, and
        that build fails.
        """
        if self._installedDistributionVersion("imagecodecs") is not None:
            return
        logging.info(_("Installing imagecodecs compatible with this Slicer build..."))
        # --no-deps: a current imagecodecs wheel requires NumPy 2, and a normal
        # install would undo the NumPy 1 pin before SeqSeg ever runs.
        self._pipInstall("imagecodecs<=2026.3.6 --no-deps")

    def _installWorkingAcvlUtils(self):
        """acvl-utils 0.2.1 and 0.2.2 do not import on Python 3.9.

        0.2.0 is the last release that does, which is what SlicerNNUNet
        installs on Slicer 5.8 and earlier.
        """
        from packaging.version import parse

        installed = self._installedDistributionVersion("acvl_utils")
        if installed is None:
            installed = self._installedDistributionVersion("acvl-utils")
        if installed is not None and parse(installed) == parse("0.2"):
            return
        logging.info(_("Installing a working acvl-utils package version..."))
        self._pipInstall("acvl_utils==0.2.0")

    def _latest_pypi_version(self, package_name):
        """Return the current version on PyPI, or None if the lookup fails."""
        import urllib.error
        import urllib.request

        url = f"https://pypi.org/pypi/{package_name}/json"
        request = urllib.request.Request(url, headers={"User-Agent": "SlicerSeqSeg"})
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode())
            version_str = payload.get("info", {}).get("version")
            return version_str or None
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
            logging.warning("Could not look up latest %s version on PyPI: %s", package_name, exc)
            return None

    def _seqseg_install_command(self, installed_version, upgrade=False):
        """Return a pip install command for seqseg, or None if the installed version is current."""
        from packaging import version as pkg_version

        minimum_v = pkg_version.parse(self.MINIMUM_SEQSEG_VERSION)
        installed_v = pkg_version.parse(installed_version) if installed_version else None
        latest_str = self._latest_pypi_version(self.SEQSEG_PYTHON_PACKAGE_NAME)
        latest_v = None
        if latest_str:
            try:
                latest_v = pkg_version.parse(latest_str)
            except pkg_version.InvalidVersion:
                logging.warning("Ignoring invalid SeqSeg version reported by PyPI: %s", latest_str)

        target_spec = None
        if latest_v is not None and latest_v >= minimum_v:
            if installed_v is None:
                logging.info(_("SeqSeg Python package is not installed; installing {version} from PyPI.").format(version=latest_str))
                target_spec = f"{self.SEQSEG_PYTHON_PACKAGE_NAME}=={latest_str}"
            elif installed_v < latest_v:
                logging.info(
                    _("Newer SeqSeg Python package available ({installed} → {latest}); updating.").format(
                        installed=installed_version, latest=latest_str
                    )
                )
                target_spec = f"{self.SEQSEG_PYTHON_PACKAGE_NAME}=={latest_str}"
            elif upgrade:
                logging.info(_("Reinstalling SeqSeg Python package {version} from PyPI.").format(version=latest_str))
                target_spec = f"{self.SEQSEG_PYTHON_PACKAGE_NAME}=={latest_str}"
            else:
                logging.info(_("SeqSeg Python package is up to date ({version}).").format(version=installed_version))
        elif installed_v is None or installed_v < minimum_v or upgrade:
            if installed_v is not None and installed_v < minimum_v:
                logging.info(
                    _("Installed SeqSeg version ({installed}) is older than the minimum ({minimum}); updating.").format(
                        installed=installed_version, minimum=self.MINIMUM_SEQSEG_VERSION
                    )
                )
            elif installed_v is None:
                logging.info(
                    _("SeqSeg Python package is not installed; installing {spec}.").format(
                        spec=f"{self.SEQSEG_PYTHON_PACKAGE_NAME}>={self.MINIMUM_SEQSEG_VERSION}"
                    )
                )
            target_spec = f"{self.SEQSEG_PYTHON_PACKAGE_NAME}>={self.MINIMUM_SEQSEG_VERSION}"
        else:
            logging.info(
                _("Keeping installed SeqSeg Python package ({version}); could not check PyPI for a newer release.").format(
                    version=installed_version
                )
            )

        if target_spec is None:
            return None
        if installed_v is not None or upgrade:
            return f"{target_spec} --upgrade"
        return target_spec

    def setupPythonRequirements(self, upgrade=False):
        """Install torch, nnunetv2, and the seqseg package.

        The PyTorch and NNUNet extensions are used when they are loaded, because
        that is how the rest of Slicer's nnUNet modules install these packages.
        Without them, the same versions are installed here. A plain
        ``pip install torch`` often fetches a wheel with no kernels for this
        GPU, which nnU-Net then reports as an out-of-memory error, so that
        path goes through light-the-torch the way PyTorchUtils does.
        """

        import importlib.metadata

        from packaging import version as pkg_version
        from packaging.requirements import Requirement

        requiredTorchVersion = "2.2.2"
        requiredTorchVisionVersion = "0.17.2"
        minimumNnunetV2Version = "2.5.1"
        minimum_nnunet_v = pkg_version.parse(minimumNnunetV2Version)

        def _torch_base_matches(version_str):
            return pkg_version.parse(version_str).base_version == requiredTorchVersion

        # torch 2.2.2 is built against NumPy 1.x; NumPy 2.x fails import with _ARRAY_API not found.
        # Downgrade first, but keep going: restarting here installs the other packages on the
        # next launch, those packages pull NumPy 2 back, the run fails, and the launch after
        # that downgrades NumPy and asks for a second restart.
        numpy_version_str = importlib.metadata.version("numpy")
        restart_for_numpy = pkg_version.parse(numpy_version_str) >= pkg_version.parse("2.0.0")
        if restart_for_numpy:
            logging.info(_("Downgrading NumPy from {current} to <2 for PyTorch 2.2.2 compatibility...").format(current=numpy_version_str))
            self._pipInstall("numpy<2")

        packagesToSkip = [
            "SimpleITK",
            "torch",
            "torchvision",
            "nnunetv2",
            "numpy",
            "requests",
            "rt_utils",
        ]

        confirmPackagesToInstall = []

        try:
            import PyTorchUtils
        except ModuleNotFoundError:
            PyTorchUtils = None
            logging.info("PyTorch extension is not installed. SeqSeg will install torch itself if it is missing.")

        # SlicerNNUNet's installer stops until the PyTorch extension is present
        # and then asks for a restart. Without that extension, install nnunet here.
        try:
            import SlicerNNUNetLib
        except ModuleNotFoundError:
            SlicerNNUNetLib = None
            logging.info("NNUNet extension is not installed. SeqSeg will install nnunetv2 itself if it is missing.")
        useNnunetExtension = SlicerNNUNetLib is not None and PyTorchUtils is not None
        # NumPy 2 is still the module loaded in this process. torchInstalled()
        # and SlicerNNUNet both import torch, which fails with _ARRAY_API not
        # found until Slicer is restarted onto NumPy 1.
        if restart_for_numpy:
            useNnunetExtension = False

        torchLogic = None
        nnunetlogic = None
        if not restart_for_numpy and PyTorchUtils is not None:
            torchLogic = PyTorchUtils.PyTorchUtilsLogic()
            if not torchLogic.torchInstalled():
                confirmPackagesToInstall.append("PyTorch")
            elif not _torch_base_matches(torchLogic.torch.__version__):
                confirmPackagesToInstall.append("PyTorch")
        else:
            torchVersion = self._installedDistributionVersion("torch")
            torchvisionVersion = self._installedDistributionVersion("torchvision")
            torchOk = torchVersion is not None and _torch_base_matches(torchVersion)
            visionOk = (
                torchvisionVersion is not None
                and pkg_version.parse(torchvisionVersion).base_version == requiredTorchVisionVersion
            )
            if not torchOk or not visionOk:
                confirmPackagesToInstall.append("PyTorch")

        # nnunetv2 below 2.7 does not run under Rosetta2. SlicerNNUNet raises
        # the floor itself; the pip path has to do the same or a Mac without
        # the extension installs a version that cannot import.
        nnunetFloor = minimumNnunetV2Version
        if not useNnunetExtension and sys.platform == "darwin" and pkg_version.parse("2.7.0") > minimum_nnunet_v:
            nnunetFloor = "2.7.0"
        nnunet_floor_v = pkg_version.parse(nnunetFloor)

        if useNnunetExtension:
            nnunetlogic = SlicerNNUNetLib.InstallLogic(doAskConfirmation=False)
            if not nnunetlogic.isPackageInstalled(Requirement("nnunetv2")):
                confirmPackagesToInstall.append("nnunetv2")
            elif pkg_version.parse(str(nnunetlogic.getInstalledNNUnetVersion())) < minimum_nnunet_v:
                confirmPackagesToInstall.append("nnunetv2")
        else:
            installedNnunet = self._installedDistributionVersion("nnunetv2")
            treeReady = (
                installedNnunet is not None
                and self._dependencyTreeCovered("nnunetv2", self._nnunetPackagesToSkip())
            )
            if installedNnunet is None or pkg_version.parse(installedNnunet) < nnunet_floor_v or not treeReady:
                confirmPackagesToInstall.append("nnunetv2")

        pipFallback = ("PyTorch" in confirmPackagesToInstall and PyTorchUtils is None) or (
            "nnunetv2" in confirmPackagesToInstall and not useNnunetExtension
        )
        if pipFallback and self._slicerTesting():
            raise InstallError(
                _("torch and nnunetv2 are not installed. Automatic installation is skipped while Slicer is running with --testing."),
                restartRequired=False,
            )

        if confirmPackagesToInstall:
            if not slicer.util.confirmOkCancelDisplay(
                _(
                    "SeqSeg needs PyTorch and nnUNet in Slicer's Python. "
                    "The download can be hundreds of megabytes and take several minutes. "
                    "If NumPy is 2.x it is downgraded to 1.x, and Slicer then asks to restart. "
                    "The selected model weights are a separate download after that. "
                    "Click OK to install."
                ),
                _("Confirm Python package installation"),
                detailedText=_("Python packages that will be installed: {package_list}").format(
                    package_list=", ".join(confirmPackagesToInstall)
                ),
            ):
                if restart_for_numpy:
                    raise InstallError(
                        _("NumPy was downgraded to 1.x for PyTorch 2.2.2. Restart Slicer. Package installation was cancelled."),
                        restartRequired=True,
                    )
                raise InstallError(_("User cancelled installation."), restartRequired=False)

        torchAlreadyImported = "torch" in sys.modules
        restart_for_torch = False
        if "PyTorch" in confirmPackagesToInstall:
            logging.info(_("Installing PyTorch Python packages (may take several minutes)..."))
            # installTorch imports torch. Skip it while this process still has NumPy 2 loaded.
            if PyTorchUtils is not None and not restart_for_numpy:
                torch_result = torchLogic.installTorch(
                    askConfirmation=False,
                    torchVersionRequirement=f"=={requiredTorchVersion}",
                    torchVisionVersionRequirement=f"=={requiredTorchVisionVersion}",
                )
                if torch_result is None:
                    raise InstallError(_("PyTorch installation failed. Install the PyTorch extension from the Extensions Manager."))
            else:
                try:
                    self._installTorchWithoutExtension(requiredTorchVersion, requiredTorchVisionVersion)
                except InstallError:
                    raise
                except Exception as exc:
                    raise InstallError(
                        _("PyTorch installation failed ({error}). See the Python console for pip output.").format(error=exc)
                    ) from exc
            # importlib still has the previous scan cached. An already-imported
            # torch module is also not the wheel pip just wrote.
            import importlib
            importlib.invalidate_caches()
            installedTorchVersion = self._installedDistributionVersion("torch")
            if not installedTorchVersion or not _torch_base_matches(installedTorchVersion):
                remedy = (
                    _("Use the PyTorch Util module to install a compatible build.")
                    if PyTorchUtils is not None
                    else _("See the Python console for pip output.")
                )
                raise InstallError(
                    _("PyTorch version {current} was expected after install ({required}). {remedy}").format(
                        current=installedTorchVersion,
                        required=requiredTorchVersion,
                        remedy=remedy,
                    )
                )
            # A torch already imported in this process stays the old module
            # after pip replaces the files. Finish the other installs first, then
            # restart once, so this is not a failed run followed by another downgrade.
            if torchAlreadyImported:
                restart_for_torch = True
        elif (
            not restart_for_numpy
            and PyTorchUtils is not None
            and not _torch_base_matches(torchLogic.torch.__version__)
        ):
            raise InstallError(
                _("PyTorch version {current} is required to be {required}. Use the PyTorch Util module to install a compatible build.").format(
                    current=torchLogic.torch.__version__,
                    required=requiredTorchVersion,
                )
            )

        if "nnunetv2" in confirmPackagesToInstall:
            logging.info(_("Installing nnunetv2 (may take several minutes)..."))
            if useNnunetExtension:
                nnunet_ok = nnunetlogic.setupPythonRequirements(f"nnunetv2>={minimumNnunetV2Version}")
                if not nnunet_ok:
                    raise InstallError(_("nnUNet v2 installation failed. Install the Slicer NNUNet extension from the Extensions Manager."))
                installed_nnunet_version = pkg_version.parse(str(nnunetlogic.getInstalledNNUnetVersion()))
                version_floor = minimum_nnunet_v
                version_floor_text = minimumNnunetV2Version
            else:
                try:
                    self._installNnunetWithoutExtension(f"nnunetv2>={nnunetFloor}")
                except InstallError:
                    raise
                except Exception as exc:
                    raise InstallError(
                        _("nnUNet v2 installation failed ({error}). See the Python console for pip output.").format(error=exc)
                    ) from exc
                installed_nnunet_text = self._installedDistributionVersion("nnunetv2")
                if not installed_nnunet_text:
                    raise InstallError(_("nnUNet v2 installation failed. See the Python console for pip output."))
                installed_nnunet_version = pkg_version.parse(installed_nnunet_text)
                version_floor = nnunet_floor_v
                version_floor_text = nnunetFloor
            if installed_nnunet_version < version_floor:
                remedy = (
                    _("Use the nnUNet module to install a compatible version.")
                    if useNnunetExtension
                    else _("See the Python console for pip output.")
                )
                raise InstallError(
                    _("nnUNet v2 version {current} was installed but at least {required} is required. {remedy}").format(
                        current=installed_nnunet_version,
                        required=version_floor_text,
                        remedy=remedy,
                    )
                )
        elif useNnunetExtension:
            installed_nnunet_version = pkg_version.parse(str(nnunetlogic.getInstalledNNUnetVersion()))
            if installed_nnunet_version < minimum_nnunet_v:
                raise InstallError(
                    _("nnUNet v2 version {current} is installed but at least {required} is required. Use the nnUNet module to install a compatible version.").format(
                        current=installed_nnunet_version,
                        required=minimumNnunetV2Version,
                    )
                )

        installed_seqseg_version = None
        try:
            installed_seqseg_version = importlib.metadata.version(self.SEQSEG_PYTHON_PACKAGE_NAME)
        except importlib.metadata.PackageNotFoundError:
            pass

        seqseg_install_command = self._seqseg_install_command(installed_seqseg_version, upgrade=upgrade)
        if seqseg_install_command:
            logging.info(_("Installing SeqSeg Python package ({spec})...").format(spec=seqseg_install_command))
            self.pipInstallSelective(self.SEQSEG_PYTHON_PACKAGE_NAME, seqseg_install_command, packagesToSkip)

        try:
            installed_seqseg_version = importlib.metadata.version(self.SEQSEG_PYTHON_PACKAGE_NAME)
        except importlib.metadata.PackageNotFoundError:
            raise InstallError(_("SeqSeg Python package installation failed. See the Python console for pip output."))

        if pkg_version.parse(installed_seqseg_version) < pkg_version.parse(self.MINIMUM_SEQSEG_VERSION):
            raise InstallError(
                _("SeqSeg Python package version {current} is installed but at least {required} is required.").format(
                    current=installed_seqseg_version,
                    required=self.MINIMUM_SEQSEG_VERSION,
                )
            )

        import importlib
        importlib.invalidate_caches()
        disk_numpy = self._installedDistributionVersion("numpy")
        if disk_numpy and pkg_version.parse(disk_numpy) >= pkg_version.parse("2.0.0"):
            logging.info(_("NumPy {current} was installed by a dependency; downgrading to <2.").format(current=disk_numpy))
            self._pipInstall("numpy<2")
            restart_for_numpy = True
        if restart_for_numpy:
            raise InstallError(
                _("NumPy 2.x is incompatible with PyTorch 2.2.2. NumPy was downgraded to 1.x and the Python packages were installed. Restart Slicer and run SeqSeg again."),
                restartRequired=True,
            )
        if restart_for_torch:
            raise InstallError(
                _("PyTorch was upgraded in this session. Restart Slicer so SeqSeg imports the new torch."),
                restartRequired=True,
            )

    def parentNodeUnderVolume(self, node, volumeNode) -> None:
        """Put a result under the volume in the data tree.

        A segmentation added with AddNewNodeByClass is a sibling of the image.
        The data module then shows SeqSeg Output as its own object, which is
        the "not associated with the image" state.
        """
        if node is None or volumeNode is None:
            return
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        if shNode is None:
            return
        volumeItem = shNode.GetItemByDataNode(volumeNode)
        nodeItem = shNode.GetItemByDataNode(node)
        if not volumeItem or not nodeItem:
            return
        if shNode.GetItemParent(nodeItem) != volumeItem:
            shNode.SetItemParent(nodeItem, volumeItem)

    def associateSegmentationWithVolume(self, segmentationNode, volumeNode) -> None:
        """Resample the segmentation onto the volume and record that volume as its source.

        The SeqSeg file has its own grid. Copying that array onto the image
        with updateSegmentBinaryLabelmapFromArray resizes the image data when
        the shapes differ and leaves the segment with no grid the slice views
        can draw. Setting the reference volume without resampling does the
        same: the Segmentations module then says the segmentation does not
        fit the image, and the slices stay empty.
        """
        if segmentationNode is None or volumeNode is None:
            return
        geometryLogic = slicer.vtkSlicerSegmentationGeometryLogic()
        geometryLogic.SetInputSegmentationNode(segmentationNode)
        geometryLogic.SetSourceGeometryNode(volumeNode)
        message = geometryLogic.CalculateOutputGeometry()
        if message:
            logging.warning("SeqSeg segmentation could not use the volume grid: %s", message)
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        else:
            resampled = geometryLogic.ResampleLabelmapsInSegmentationNode()
            referenced = geometryLogic.SetReferenceImageGeometryInSegmentationNode()
            if not resampled or not referenced:
                logging.warning(
                    "SeqSeg segmentation was not placed on the volume grid (resampled=%s, referenced=%s).",
                    resampled,
                    referenced,
                )
                segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        self.parentNodeUnderVolume(segmentationNode, volumeNode)

    def showSegmentationOverVolume(self, segmentationNode, volumeNode, overlayOpacity=0.45):
        """Draw the segmentation on the volume in the slice views and in 3D.

        A previous pass exported a second labelmap and turned this node's own
        slice drawing off. That labelmap did not share the image, so the
        slices showed neither one.
        """
        if not segmentationNode or not volumeNode:
            logging.warning("showSegmentationOverVolume: missing segmentation or volume node")
            return
        self._discardGeneratedLabelmap(segmentationNode)
        segmentationNode.CreateDefaultDisplayNodes()
        displayNode = segmentationNode.GetDisplayNode()
        segmentation = segmentationNode.GetSegmentation()
        if displayNode is not None and segmentation is not None:
            displayNode.SetVisibility(True)
            if hasattr(displayNode, "SetVisibility2D"):
                displayNode.SetVisibility2D(True)
            if hasattr(displayNode, "SetVisibility2DFill"):
                displayNode.SetVisibility2DFill(True)
            if hasattr(displayNode, "SetVisibility2DOutline"):
                displayNode.SetVisibility2DOutline(True)
            if hasattr(displayNode, "SetOpacity2D"):
                displayNode.SetOpacity2D(overlayOpacity)
            if hasattr(displayNode, "SetVisibility3D"):
                displayNode.SetVisibility3D(True)
            if hasattr(displayNode, "SetOpacity3D"):
                displayNode.SetOpacity3D(overlayOpacity)
            else:
                displayNode.SetOpacity(overlayOpacity)
            for index in range(segmentation.GetNumberOfSegments()):
                segmentId = segmentation.GetNthSegmentID(index)
                if hasattr(displayNode, "SetSegmentVisibility"):
                    displayNode.SetSegmentVisibility(segmentId, True)
                if hasattr(displayNode, "SetSegmentVisibility2DFill"):
                    displayNode.SetSegmentVisibility2DFill(segmentId, True)
                if hasattr(displayNode, "SetSegmentVisibility2DOutline"):
                    displayNode.SetSegmentVisibility2DOutline(segmentId, True)
                if hasattr(displayNode, "SetSegmentOpacity2DFill"):
                    displayNode.SetSegmentOpacity2DFill(segmentId, overlayOpacity)
                if hasattr(displayNode, "SetSegmentOpacity3D"):
                    displayNode.SetSegmentOpacity3D(segmentId, overlayOpacity)
        # An empty label left selected from the earlier export hides the image
        # under a blank layer. The segmentation's own slice drawing is the overlay.
        slicer.util.setSliceViewerLayers(background=volumeNode, label="", fit=True)

    def _discardGeneratedLabelmap(self, segmentationNode) -> None:
        """Remove the labelmap an earlier run created beside the segmentation."""
        if segmentationNode is None or not hasattr(segmentationNode, "GetAttribute"):
            return
        stored = segmentationNode.GetAttribute("SeqSeg.Labelmap")
        if not stored:
            return
        labelmap = slicer.mrmlScene.GetNodeByID(stored)
        if labelmap is not None:
            slicer.mrmlScene.RemoveNode(labelmap)
        if hasattr(segmentationNode, "RemoveAttribute"):
            segmentationNode.RemoveAttribute("SeqSeg.Labelmap")

    def loadSegmentationToNode(self, segmentation_file_path: str, outputSegmentationNode, inputVolume) -> bool:
        """Load a SeqSeg segmentation file into the output node, on the volume's grid.

        :param segmentation_file_path: Full path to the segmentation file
        :param outputSegmentationNode: The segmentation node to update
        :param inputVolume: Volume the segmentation is shown on
        :return: True if successful, False otherwise
        """
        labelmap = None
        try:
            if not os.path.exists(segmentation_file_path):
                logging.error(f"Segmentation file not found: {segmentation_file_path}")
                return False

            labelmap = _normalize_loaded_volume_node(slicer.util.loadLabelVolume(segmentation_file_path))
            if not labelmap:
                logging.error(f"Failed to load segmentation file: {segmentation_file_path}")
                return False
            if outputSegmentationNode is None or inputVolume is None:
                logging.error("Segmentation load needs an output node and a reference volume")
                return False

            # A rerun used to add another segment beside the previous one, so the
            # overlay stacked every run. This node is the output, so replace it.
            segmentation = outputSegmentationNode.GetSegmentation()
            segmentation.RemoveAllSegments()
            # Import keeps the file's world position. Copying the voxel array
            # onto the volume does not: a different shape resizes the image
            # data and the segment never lands on the scan.
            imported = slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                labelmap, outputSegmentationNode
            )
            if not imported:
                logging.error("Importing SeqSeg labelmap into the segmentation failed")
                return False

            if segmentation.GetNumberOfSegments() > 0:
                segment = segmentation.GetNthSegment(0)
                if segment is not None:
                    segment.SetColor(0.9, 0.1, 0.1)

            self.associateSegmentationWithVolume(outputSegmentationNode, inputVolume)
            logging.info(f"Successfully loaded segmentation from: {segmentation_file_path}")
            return True

        except Exception as e:
            logging.error(f"Error loading segmentation file {segmentation_file_path}: {e}")
            return False
        finally:
            if labelmap is not None:
                slicer.mrmlScene.RemoveNode(labelmap)

    def loadSurfaceModel(self, surface_path: str):
        """Load a SeqSeg .vtp as a model. Run does this so the surface is not a second button."""
        if not surface_path or not os.path.exists(surface_path):
            return None
        node = slicer.util.loadModel(surface_path)
        if node is None:
            return None
        display = node.GetDisplayNode()
        if display is not None:
            display.SetVisibility(True)
            display.SetColor(1, 0, 0)
            display.SetOpacity(0.8)
        logging.info("Loaded surface mesh: %s", surface_path)
        return node

    @staticmethod
    def latest_surface_mesh(output_dir: str, case_name: str = "", not_before: float = 0) -> Optional[str]:
        return choose_seqseg_surface(output_dir, case_name, not_before)

    @staticmethod
    def _combined_seqseg_log(result) -> str:
        return "\n".join(part for part in (result.stdout, result.stderr) if part)

    @staticmethod
    def _seqseg_log_indicates_inference_failure(text: str) -> Optional[str]:
        """Return a user-facing reason if SeqSeg exited 0 but tracking did not run."""
        if not text:
            return None
        if "no kernel image is available for execution on the device" in text:
            return (
                "PyTorch CUDA cannot execute kernels on this GPU "
                "(no kernel image is available for execution on the device). "
                "The installed torch build does not match this GPU architecture. "
                "Install a PyTorch build that has kernels for this GPU, or run on CPU."
            )
        if "Didnt work for first surface" in text and "Ratio of processed voxels: 0.00%" in text:
            return (
                "SeqSeg tracking produced no vessel (0.00% voxels processed); "
                "the first surface prediction failed. Check GPU/CUDA compatibility, "
                "seed placement, and that the selected model weights match the image."
            )
        return None

    def _should_force_seqseg_cpu(self) -> bool:
        """True when CUDA is advertised but a tiny kernel cannot run on the GPU."""
        try:
            import torch
        except ImportError:
            return False
        if not torch.cuda.is_available():
            return False
        try:
            sample = torch.zeros(8, device="cuda")
            sample = (sample + 1) * 2
            torch.cuda.synchronize()
            _ = float(sample.sum().item())
            return False
        except Exception as error:
            logging.warning(
                "CUDA is available but a test kernel failed (%s). Forcing SeqSeg onto CPU.",
                error,
            )
            return True

    def _run_seqseg_cli(self, seqseg_cmd, cwd, progress=None):
        """Run seqseg without blocking Slicer's event loop.

        subprocess.run captured the pipes and froze the panel until tracking
        finished, so a long 3d_fullres run looked hung and could not be
        cancelled. QProcess keeps the progress dialog's Cancel button alive.
        The pipes are drained while the process runs: if they fill, the child
        blocks and Cancel never arrives.
        """
        logging.info(f"Running SeqSeg command: {' '.join(seqseg_cmd)}")
        process = qt.QProcess()
        process.setWorkingDirectory(cwd)
        process.setProcessChannelMode(qt.QProcess.SeparateChannels)
        # PythonQt does not expose QStringList. start() accepts a list of strings.
        arguments = [str(argument) for argument in seqseg_cmd[1:]]
        process.start(seqseg_cmd[0], arguments)
        if not process.waitForStarted(15000):
            raise RuntimeError(
                f"Could not start '{seqseg_cmd[0]}'. {process.errorString()}\n\n"
                "SeqSeg is the console script installed into Slicer's Python. "
                "If that script is not on PATH, the run cannot find it."
            )

        stdout_parts = []
        stderr_parts = []
        cancelled = {"value": False}
        if progress is not None:
            try:
                progress.connect("canceled()", lambda: cancelled.__setitem__("value", True))
            except Exception:
                logging.debug("Could not connect SeqSeg progress cancel", exc_info=True)

        not_running = qt.QProcess.NotRunning
        while process.state() != not_running:
            self._drain_seqseg_process(process, stdout_parts, stderr_parts, progress)
            if self._seqseg_progress_cancelled(progress, cancelled["value"]):
                process.kill()
                process.waitForFinished(5000)
                self._drain_seqseg_process(process, stdout_parts, stderr_parts, progress)
                raise SeqSegCancelled("SeqSeg was cancelled.")
            slicer.app.processEvents(qt.QEventLoop.AllEvents, 100)
        self._drain_seqseg_process(process, stdout_parts, stderr_parts, progress)

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)
        if process.exitStatus() != qt.QProcess.NormalExit:
            returncode = -1
        else:
            returncode = int(process.exitCode())
        if stdout:
            logging.info(f"SeqSeg output:\n{stdout}")
        if stderr:
            logging.info(f"SeqSeg stderr:\n{stderr}")
        return _SeqSegRunResult(returncode, stdout, stderr)

    @staticmethod
    def _decode_qbytearray(data) -> str:
        if not data:
            return ""
        try:
            raw = data.data()
        except Exception:
            raw = data
        if isinstance(raw, str):
            return raw
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        try:
            return bytes(raw).decode("utf-8", errors="replace")
        except Exception:
            return str(data)

    def _drain_seqseg_process(self, process, stdout_parts, stderr_parts, progress) -> None:
        stdout = self._decode_qbytearray(process.readAllStandardOutput())
        stderr = self._decode_qbytearray(process.readAllStandardError())
        if stdout:
            stdout_parts.append(stdout)
            self._show_seqseg_progress_line(progress, stdout)
        if stderr:
            stderr_parts.append(stderr)
            self._show_seqseg_progress_line(progress, stderr)

    @staticmethod
    def _show_seqseg_progress_line(progress, text: str) -> None:
        if progress is None or not text or not str(text).strip():
            return
        line = str(text).strip().splitlines()[-1]
        if len(line) > 140:
            line = line[:137] + "..."
        try:
            progress.setLabelText(line)
        except Exception:
            logging.debug("Could not update SeqSeg progress text", exc_info=True)

    @staticmethod
    def _seqseg_progress_cancelled(progress, cancelled_flag: bool) -> bool:
        if cancelled_flag:
            return True
        if progress is None:
            return False
        was_canceled = getattr(progress, "wasCanceled", None)
        if was_canceled is None:
            return False
        try:
            return bool(was_canceled() if callable(was_canceled) else was_canceled)
        except Exception:
            return False

    def _open_run_progress(self, label: str):
        try:
            return slicer.util.createProgressDialog(
                value=0,
                maximum=0,
                labelText=label,
                windowTitle="SeqSeg",
            )
        except Exception:
            logging.warning("Could not open SeqSeg progress dialog", exc_info=True)
            return None

    @staticmethod
    def _close_run_progress(progress) -> None:
        if progress is None:
            return
        try:
            progress.close()
        except Exception:
            logging.debug("Could not close SeqSeg progress dialog", exc_info=True)

    @staticmethod
    def _markupPointRAS(value, point_index: int = 0):
        """Return RAS coords from a markup node or a 3-element coordinate list."""
        if value is None:
            raise ValueError("Seed point is not defined")
        if hasattr(value, "GetNthControlPointPosition"):
            if value.GetNumberOfControlPoints() <= point_index:
                raise ValueError("Seed point is not defined")
            ras = [0.0, 0.0, 0.0]
            value.GetNthControlPointPosition(point_index, ras)
            return [float(ras[0]), float(ras[1]), float(ras[2])]
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, IndexError, ValueError):
            raise ValueError("Seed point is not defined")

    @staticmethod
    def _rasToLps(ras):
        return [-float(ras[0]), -float(ras[1]), float(ras[2])]

    @classmethod
    def _normalizeSeedPairs(cls, seedPairs, seedPoint1Node, seedPoint2Node, radiusEstimate: float):
        """Return a list of {start, direction, radius} dicts from seedPairs or the pair-1 nodes."""
        pairs = []
        if seedPairs:
            for item in seedPairs:
                if isinstance(item, dict):
                    start = item.get("start")
                    direction = item.get("direction")
                    radius = item.get("radius", radiusEstimate)
                else:
                    start = item[0]
                    direction = item[1]
                    radius = item[2] if len(item) > 2 else radiusEstimate
                if start and direction:
                    pairs.append({
                        "start": start,
                        "direction": direction,
                        "radius": radius if radius and radius > 0 else radiusEstimate,
                    })
        if not pairs and seedPoint1Node and seedPoint2Node:
            pairs.append({
                "start": seedPoint1Node,
                "direction": seedPoint2Node,
                "radius": radiusEstimate,
            })
        return pairs

    @classmethod
    def buildSeedTriplets(cls, seedPairs, radiusEstimate: float, coordinateSystem: str):
        """Build SeqSeg seeds.json triplets: [start, direction, radius] in the chosen world system."""
        use_ras = coordinateSystem == "RAS World"
        triplets = []
        details = []
        for index, pair in enumerate(seedPairs, start=1):
            start_ras = cls._markupPointRAS(pair["start"])
            direction_ras = cls._markupPointRAS(pair["direction"])
            radius = float(pair.get("radius", radiusEstimate))
            if radius <= 0:
                radius = float(radiusEstimate)
            start_final = list(start_ras) if use_ras else cls._rasToLps(start_ras)
            direction_final = list(direction_ras) if use_ras else cls._rasToLps(direction_ras)
            triplet = [
                [float(start_final[0]), float(start_final[1]), float(start_final[2])],
                [float(direction_final[0]), float(direction_final[1]), float(direction_final[2])],
                radius,
            ]
            triplets.append(triplet)
            details.append({
                "index": index,
                "start_ras": start_ras,
                "direction_ras": direction_ras,
                "triplet": triplet,
                "radius": radius,
            })
        return triplets, details

    def runSeqSeg(self,
                  inputVolume: vtkMRMLScalarVolumeNode,
                  seedPoint1Node,
                  seedPoint2Node,
                  radiusEstimate: float,
                  maxSteps: int,
                  maxBranches: int,
                  maxStepsPerBranch: int,
                  imageUnit: str,
                  scale: str,
                  coordinateSystem: str,
                  nnunetResultsPath: str,
                  nnunetType: str,
                  trainDataset: str,
                  outputDirectory: str,
                  outputSegmentationNode,
                  simvascular: bool = False,
                  forceCpu: bool = False,
                  showResult: bool = True,
                  seedPairs: Optional[Sequence] = None) -> None:
        """
        Run the SeqSeg segmentation algorithm using CLI interface.
        Can be used without GUI widget.
        :param inputVolume: input volume for segmentation
        :param seedPoint1Node: start point of seed pair 1 (used if seedPairs is empty)
        :param seedPoint2Node: direction point of seed pair 1 (used if seedPairs is empty)
        :param radiusEstimate: lumen radius in millimetres. Slicer writes the NIfTI in millimetres and SeqSeg sizes the patch as radius / spacing, so this number is not divided when Image Unit is cm. Image Unit and Scale describe the header to the network; they do not rescale the seed.
        :param maxSteps: maximum number of steps for segmentation
        :param imageUnit: unit of the medical image (mm or cm)
        :param scale: SeqSeg -scale as "0.1", "1", or "10" (training vs. image unit conversion)
        :param nnunetResultsPath: path to nnUNet results folder
        :param nnunetType: type of nnUNet model (3d_fullres or 2d)
        :param trainDataset: name of dataset used to train nnUNet
        :param outputDirectory: directory for SeqSeg outputs (data_dir)
        :param outputSegmentationNode: output segmentation node
        :param simvascular: if True, pass -simvascular 1 to write SimVascular project layout
        :param forceCpu: if True, pass -cpu so nnUNet inference runs on the CPU
        :param showResult: show output segmentation in slice viewers
        :param seedPairs: optional list of {start, direction, radius} dicts or (start, direction[, radius]) tuples
        """

        # Validate minimum requirements
        if not inputVolume:
            raise ValueError("Input volume is required")

        normalized_pairs = self._normalizeSeedPairs(seedPairs, seedPoint1Node, seedPoint2Node, radiusEstimate)

        # Create default seed pair 1 if no complete pairs were provided
        if not normalized_pairs:
            logging.warning("Seed pair 1 not found, creating default start and direction points")
            if not seedPoint1Node:
                seedPoint1Node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Default Seed Point 1")
                seedPoint1Node.SetMaximumNumberOfControlPoints(1)
                bounds = [0] * 6
                inputVolume.GetBounds(bounds)
                centerX = (bounds[0] + bounds[1]) / 2
                centerY = (bounds[2] + bounds[3]) / 2
                centerZ = bounds[4] + (bounds[5] - bounds[4]) * 0.25
                seedPoint1Node.AddControlPoint([centerX, centerY, centerZ])
            if not seedPoint2Node:
                seedPoint2Node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Default Seed Point 2")
                seedPoint2Node.SetMaximumNumberOfControlPoints(1)
                bounds = [0] * 6
                inputVolume.GetBounds(bounds)
                centerX = (bounds[0] + bounds[1]) / 2
                centerY = (bounds[2] + bounds[3]) / 2
                centerZ = bounds[4] + (bounds[5] - bounds[4]) * 0.75
                seedPoint2Node.AddControlPoint([centerX, centerY, centerZ])
            normalized_pairs = self._normalizeSeedPairs(None, seedPoint1Node, seedPoint2Node, radiusEstimate)

        seedPoint1Node = normalized_pairs[0]["start"]
        seedPoint2Node = normalized_pairs[0]["direction"]
        requested_max_branches = maxBranches
        maxBranches = max_branches_for_seeds(maxBranches, len(normalized_pairs))
        if maxBranches > requested_max_branches:
            logging.warning(
                "Max Branches is %s but %s seed(s) were provided. "
                "Raising Max Branches to %s so every seed is used as a start.",
                requested_max_branches,
                len(normalized_pairs),
                maxBranches,
            )

        # Create default output segmentation if missing
        if not outputSegmentationNode:
            logging.warning("Output segmentation not found, creating default")
            outputSegmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "SeqSeg Output")
            outputSegmentationNode.CreateDefaultDisplayNodes()
            outputSegmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)
        
        # Use default values for missing parameters
        if radiusEstimate <= 0:
            radiusEstimate = 5.0
            logging.warning("Invalid radius, using default: 5.0")
        
        if not nnunetResultsPath or not os.path.exists(nnunetResultsPath):
            logging.warning("nnUNet results path not set or invalid - SeqSeg may fail without proper model path")
        
        if not trainDataset:
            raise ValueError("Train dataset name is required")

        allowed_scales = {"0.1", "1", "10"}
        if scale not in allowed_scales:
            scale = "1"
            logging.warning(f"Invalid scale, using default: 1 (got {scale!r})")
        scale_value = float(scale)

        # Validate output directory
        if not outputDirectory:
            raise ValueError("Output directory is required. Please select an output directory.")
        
        # Create output directory if it doesn't exist
        os.makedirs(outputDirectory, exist_ok=True)
        logging.info(f"Using output directory: {outputDirectory}")

        import time
        import numpy as np

        startTime = time.time()
        logging.info("SeqSeg processing started")

        try:
            self.setupPythonRequirements()

            # Use the specified output directory as data_dir
            data_dir = outputDirectory
            logging.info(f"Using data directory: {data_dir}")
            
            # Create images subdirectory (SeqSeg expects images/ subdirectory)
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            logging.info(f"Created images directory: {images_dir}")
            
            # Save input volume to images subdirectory
            # Use volume name or default name
            volume_name = inputVolume.GetName() if inputVolume.GetName() else "input_volume"
            # Remove spaces and special characters for filename
            case_name = "".join(c for c in volume_name if c.isalnum() or c in ('_', '-'))
            input_file = os.path.join(images_dir, f"{case_name}.nii.gz")
            slicer.util.saveNode(inputVolume, input_file)
            logging.info(f"Saved input volume to: {input_file}")

            seed_triplets, seed_details = self.buildSeedTriplets(
                normalized_pairs, radiusEstimate, coordinateSystem
            )
            if not seed_triplets:
                raise ValueError("At least one complete seed pair (start and direction) is required")
            coord_label = "RAS world" if coordinateSystem == "RAS World" else "LPS world"
            logging.info(f"Using {coord_label} coordinates for {len(seed_triplets)} seed pair(s)")
            for detail in seed_details:
                logging.info(
                    "Seed %s start (RAS): %s, direction (RAS): %s, radius: %s",
                    detail["index"],
                    detail["start_ras"],
                    detail["direction_ras"],
                    detail["radius"],
                )

            # Create seeds.json file in the data_dir (same level as images/)
            # Format: [{"name": "case_name", "seeds": [[start, direction, radius], ...]}]
            seeds_data = [
                {
                    "name": case_name,
                    "seeds": seed_triplets,
                }
            ]
            seeds_file = os.path.join(data_dir, "seeds.json")
            with open(seeds_file, 'w') as f:
                json.dump(seeds_data, f, indent=4)
            logging.info(f"Created seeds.json with {len(seed_triplets)} seed(s): {seeds_file}")
            logging.info("seeds.json contents:\n%s", json.dumps(seeds_data, indent=4))
            
            # Set up output directory (subdirectory within data_dir)
            output_dir = os.path.join(data_dir, "output")
            # Ensure output_dir ends with '/' for SeqSeg compatibility
            if not output_dir.endswith(os.sep):
                output_dir = output_dir + os.sep
            os.makedirs(output_dir, exist_ok=True)
            
            # Log all parameters before running
            logging.info("=== SeqSeg Parameters ===")
            logging.info(f"Input Volume: {inputVolume.GetName() if inputVolume else 'None'}")
            logging.info(f"Seed pairs: {len(seed_details)}")
            for detail in seed_details:
                logging.info(
                    "Seed %s start RAS: %s, direction RAS: %s, radius: %s",
                    detail["index"],
                    detail["start_ras"],
                    detail["direction_ras"],
                    detail["radius"],
                )
            logging.info(f"Radius Estimate: {radiusEstimate}")
            logging.info(f"Max Steps: {maxSteps}")
            logging.info(f"Max Branches: {maxBranches}")
            logging.info(f"Max Steps per Branch: {maxStepsPerBranch}")
            logging.info(f"Image Unit: {imageUnit}")
            logging.info(f"Scale: {scale_value}")
            logging.info(f"nnUNet Results Path: '{nnunetResultsPath}'")
            logging.info(f"nnUNet Type: {nnunetType}")
            logging.info(f"Train Dataset: {trainDataset}")
            logging.info(f"Data Directory: {data_dir}")
            logging.info(f"Output Directory: {output_dir}")
            logging.info(f"SimVascular Output: {simvascular}")
            logging.info(f"Force CPU: {forceCpu}")
            logging.info(f"Input File: {input_file}")
            logging.info(f"Seeds File: {seeds_file}")
            logging.info("========================")
            
            # Prepare SeqSeg command arguments according to actual SeqSeg CLI
            # Fold is omitted so SeqSeg uses its default ("all") and fold/checkpoint fallback search.
            seqseg_cmd = [
                "seqseg",
                "-data_dir", data_dir,
                "-nnunet_results_path", nnunetResultsPath,
                "-nnunet_type", nnunetType,
                "-train_dataset", trainDataset,
                "-img_ext", ".nii.gz",
                "-config_name", "aorta_tutorial",  # Default config name
                "-outdir", output_dir,
                "-unit", imageUnit,
                "-scale", str(scale_value),
                "-max_n_steps", str(maxSteps),
                "-max_n_branches", str(maxBranches),
                "-max_n_steps_per_branch", str(maxStepsPerBranch),
                # "-write_steps", "1",
            ]
            if simvascular:
                seqseg_cmd.extend(["-simvascular", "1"])
            
            # seqseg >= 2.1.1 keeps nnU-Net off the GPU with -cpu; earlier versions always took an
            # advertised CUDA device, even when no kernels were compiled for it.
            cpu_cmd = [*seqseg_cmd, "-cpu"]

            if forceCpu:
                force_cpu = True
                logging.info("Force CPU is enabled, running SeqSeg with -cpu")
            else:
                force_cpu = self._should_force_seqseg_cpu()
                if force_cpu:
                    logging.warning(
                        "PyTorch CUDA is visible but cannot execute kernels on this GPU. "
                        "Running SeqSeg with -cpu (slower). To use the GPU, install a PyTorch "
                        "build that has kernels for this GPU."
                    )

            self.lastRunLoadedSurface = False
            # Captured before the process starts so a mask already in the
            # output folder, from an earlier volume, is not this run's result.
            run_started = time.time()
            progress = self._open_run_progress(
                "Running SeqSeg. Tracking can take several minutes. Cancel stops the process."
            )
            try:
                result = self._run_seqseg_cli(
                    cpu_cmd if force_cpu else seqseg_cmd, cwd=data_dir, progress=progress
                )
                failure_reason = self._seqseg_log_indicates_inference_failure(
                    self._combined_seqseg_log(result)
                )
                if failure_reason and not force_cpu:
                    logging.warning("%s Retrying SeqSeg with -cpu.", failure_reason)
                    force_cpu = True
                    if progress is not None:
                        progress.setLabelText(
                            "This GPU cannot run the installed PyTorch. Retrying on CPU, which is slower."
                        )
                    result = self._run_seqseg_cli(cpu_cmd, cwd=data_dir, progress=progress)
                    failure_reason = self._seqseg_log_indicates_inference_failure(
                        self._combined_seqseg_log(result)
                    )
            finally:
                self._close_run_progress(progress)

            if result.returncode != 0:
                logging.error(f"SeqSeg failed with return code {result.returncode}")
                logging.error(f"Error output: {result.stderr}")
                detail = self._combined_seqseg_log(result).strip() or f"exit code {result.returncode}"
                raise RuntimeError(f"SeqSeg execution failed (exit {result.returncode}).\n\n{detail}")

            if failure_reason:
                raise RuntimeError(failure_reason)

            logging.info(
                "SeqSeg execution completed successfully%s",
                " (CPU)" if force_cpu else "",
            )

            output_file = choose_seqseg_segmentation(output_dir, case_name, run_started)
            if not output_file:
                raise RuntimeError(
                    f"SeqSeg did not write a new segmentation for {case_name} in {output_dir}. "
                    "An older segmentation in that folder was not loaded."
                )
            logging.info(f"Found output file: {output_file}")
            if maxSteps > 0 and re.search(r"_0_steps\.", os.path.basename(output_file)):
                raise RuntimeError(
                    "SeqSeg wrote a 0-step segmentation, so tracking never advanced from the seed. "
                    "Check the SeqSeg log for CUDA/GPU errors, seed placement, and model weights."
                )
            
            # Use the standardized loading method
            if not self.loadSegmentationToNode(output_file, outputSegmentationNode, inputVolume):
                raise RuntimeError("Failed to load SeqSeg output using standardized method")
            
            if showResult:
                self.showSegmentationOverVolume(outputSegmentationNode, inputVolume)

            surface = self.latest_surface_mesh(output_dir, case_name, run_started)
            if surface:
                try:
                    self.lastRunLoadedSurface = self.loadSurfaceModel(surface) is not None
                except Exception:
                    logging.warning("SeqSeg wrote a surface mesh but it could not be loaded", exc_info=True)
                    self.lastRunLoadedSurface = False

            logging.info("SeqSeg segmentation completed successfully")

        except InstallError:
            raise
        except SeqSegCancelled:
            raise
        except Exception as e:
            logging.error(f"SeqSeg processing failed: {e}")
            raise

        stopTime = time.time()
        logging.info(f"SeqSeg processing completed in {stopTime-startTime:.2f} seconds")


#
# seqsegTest
#


class seqsegTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_seqseg1()
        self.setUp()
        self.test_multiple_seed_triplets()

    def test_seqseg1(self):
        """Test basic functionality of the SeqSeg module.
        This test verifies that the module can handle seed points and volumes properly,
        and that the SeqSeg integration works as expected.
        """

        self.delayDisplay("Starting SeqSeg test")

        # Create test nodes
        inputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        inputVolume.SetName("TestVolume")
        
        # Create some test data
        import numpy as np
        testArray = np.random.rand(64, 64, 32) * 1000
        slicer.util.updateVolumeFromArray(inputVolume, testArray)
        
        # Create seed point nodes
        seedPoint1Node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        seedPoint1Node.SetName("SeedPoint1")
        seedPoint1Node.AddControlPoint([10, 10, 10])
        
        seedPoint2Node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode") 
        seedPoint2Node.SetName("SeedPoint2")
        seedPoint2Node.AddControlPoint([20, 20, 15])

        # Create output segmentation
        outputSegmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        outputSegmentation.SetName("SeqSegOutput")

        self.delayDisplay("Created test data")

        # Test the module logic
        logic = seqsegLogic()

        # Test that the SeqSeg method exists
        self.assertIsNotNone(logic.runSeqSeg, "runSeqSeg method should exist")

        # Note: We can't actually test the SeqSeg algorithm without the package installed
        # but we can test the basic structure and error handling
        try:
            # This will likely fail if SeqSeg package is not installed, which is expected
            radiusEstimate = 5.0  # Test radius in cm
            maxSteps = 1
            imageUnit = "cm"
            nnunetResultsPath = "/tmp/test_nnunet_results"  # Test path
            nnunetType = "3d_fullres"
            trainDataset = "Dataset005_SEQAORTANDFEMOMR"
            outputDirectory = "/tmp/test_seqseg_output/"  # Test output directory
            logic.runSeqSeg(inputVolume, seedPoint1Node, seedPoint2Node, radiusEstimate, 
                           maxSteps, 1, 20, imageUnit, "1", "LPS World", nnunetResultsPath, nnunetType, trainDataset, 
                           outputDirectory, outputSegmentation)
            self.delayDisplay("SeqSeg algorithm completed successfully")
        except InstallError as e:
            logging.info(f"SeqSeg dependency setup failed or incomplete for testing: {e}")
            self.delayDisplay(f"SeqSeg dependencies not satisfied - skipping algorithm run: {e}")
        except ImportError as e:
            # This is expected if SeqSeg package is not installed
            logging.info(f"SeqSeg package not available for testing: {e}")
            self.delayDisplay("SeqSeg package not installed - this is expected for basic testing")
        except Exception as e:
            logging.error(f"Unexpected error in SeqSeg test: {e}")

        # Clean up test nodes
        self.delayDisplay("Cleaning up test nodes...")
        
        # Remove test volume
        if inputVolume:
            slicer.mrmlScene.RemoveNode(inputVolume)
            
        # Remove seed point nodes
        if seedPoint1Node:
            slicer.mrmlScene.RemoveNode(seedPoint1Node)
            
        if seedPoint2Node:
            slicer.mrmlScene.RemoveNode(seedPoint2Node)
            
        # Remove output segmentation
        if outputSegmentation:
            slicer.mrmlScene.RemoveNode(outputSegmentation)

        self.delayDisplay("SeqSeg module test completed")

    def test_multiple_seed_triplets(self):
        """Verify seed-line start/direction/radius become SeqSeg seeds.json triplets."""
        self.delayDisplay("Starting multiple seed triplet test")

        line1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "Seed")
        line1.AddControlPoint([10, 20, 30])
        line1.AddControlPoint([11, 21, 31])
        line2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", "Seed_1")
        line2.AddControlPoint([40, 50, 60])
        line2.AddControlPoint([42, 50, 60])

        start1 = [10.0, 20.0, 30.0]
        direction1 = [11.0, 21.0, 31.0]
        start2 = [40.0, 50.0, 60.0]
        direction2 = [42.0, 50.0, 60.0]
        radius1 = line_length(start1, direction1)
        radius2 = line_length(start2, direction2)
        self.assertAlmostEqual(radius1, math.sqrt(3.0))
        self.assertAlmostEqual(radius2, 2.0)

        cm_start = [0.0, 0.0, 0.0]
        cm_direction = [0.7, 0.0, 0.0]
        cm_radius = line_length(cm_start, cm_direction)
        self.assertAlmostEqual(cm_radius, 0.7)
        cm_triplets, _ = seqsegLogic.buildSeedTriplets(
            [{"start": cm_start, "direction": cm_direction, "radius": cm_radius}],
            cm_radius,
            "RAS World",
        )
        self.assertAlmostEqual(cm_triplets[0][2], 0.7)

        # The radius written is the value supplied, not the distance between the
        # two points. That distance is the tracking direction. A 10 mm lumen on
        # a line of length sqrt(3) must stay 10 mm: Slicer's NIfTI is millimetres,
        # and dividing because Image Unit is cm would size the patch against the
        # wrong spacing.
        supplied_radius = 10.0
        supplied, _ = seqsegLogic.buildSeedTriplets(
            [{"start": start1, "direction": direction1, "radius": supplied_radius}],
            supplied_radius,
            "LPS World",
        )
        self.assertAlmostEqual(supplied[0][2], supplied_radius)
        self.assertNotAlmostEqual(supplied[0][2], radius1)

        aorta = model_preset("Dataset006_SEQAORTANDFEMOCT")
        self.assertEqual(aorta["trained_unit"], "cm")
        self.assertEqual(aorta["image_unit"], "mm")
        self.assertEqual(aorta["scale"], "0.1")
        self.assertEqual(aorta["radius_mm"], 10.0)
        self.assertEqual(aorta["max_steps"], 10)
        coronary = model_preset("Dataset002_CASXCT")
        self.assertEqual(coronary["trained_unit"], "mm")
        self.assertEqual(coronary["image_unit"], "mm")
        self.assertEqual(coronary["scale"], "1")
        self.assertEqual(coronary["radius_mm"], 2.0)
        self.assertEqual(coronary["nnunet_type"], "3d_fullres")
        self.assertEqual(canonical_dataset_id("Dataset010_SEQCOROASOCACT"), "Dataset002_CASXCT")
        self.assertEqual(model_preset("Dataset010_SEQCOROASOCACT")["radius_mm"], 2.0)
        fontan = model_preset("Dataset090_VMR_FONTAN_ORIGMR")
        self.assertEqual(fontan["scale"], "0.1")
        self.assertEqual(fontan["image_unit"], "mm")
        self.assertEqual(header_unit_from_spacing((0.75, 0.75, 1.25)), "mm")
        self.assertEqual(header_unit_from_spacing((0.075, 0.075, 0.125)), "cm")
        self.assertEqual(header_unit_from_spacing(None), "mm")
        mm_aorta = settings_for_volume("Dataset006_SEQAORTANDFEMOCT", "mm")
        self.assertEqual((mm_aorta["image_unit"], mm_aorta["scale"]), ("mm", "0.1"))
        cm_aorta = settings_for_volume("Dataset006_SEQAORTANDFEMOCT", "cm")
        self.assertEqual((cm_aorta["image_unit"], cm_aorta["scale"]), ("cm", "1"))
        cm_coronary = settings_for_volume("Dataset002_CASXCT", "cm")
        self.assertEqual((cm_coronary["image_unit"], cm_coronary["scale"]), ("cm", "10"))
        self.assertAlmostEqual(radius_in_header_units(10.0, "mm"), 10.0)
        self.assertAlmostEqual(radius_in_header_units(10.0, "cm"), 1.0)
        cm_hint = volume_hint(
            "Dataset006_SEQAORTANDFEMOCT", (0.08, 0.08, 0.12), "cm", "cm", "1", 10.0
        )
        self.assertIn("centimetres", cm_hint)
        self.assertIn("sent as 1", cm_hint)
        self.assertIn("its own radius", cm_hint)
        shown = datasets_shown_in_combo(())
        self.assertNotIn("Dataset091_VMR_FONTAN_003MR", shown)
        self.assertIn("Dataset002_CASXCT", shown)
        self.assertNotIn("Dataset010_SEQCOROASOCACT", shown)
        self.assertIn("Dataset006_SEQAORTANDFEMOCT", shown)
        self.assertIn("Dataset090_VMR_FONTAN_ORIGMR", shown)
        shown_local = datasets_shown_in_combo(("Dataset091_VMR_FONTAN_003MR",))
        self.assertIn("Dataset091_VMR_FONTAN_003MR", shown_local)

        logic = seqsegLogic()
        pairs = [
            {
                "start": logic._markupPointRAS(line1, 0),
                "direction": logic._markupPointRAS(line1, 1),
                "radius": radius1,
            },
            {
                "start": logic._markupPointRAS(line2, 0),
                "direction": logic._markupPointRAS(line2, 1),
                "radius": radius2,
            },
        ]
        triplets, details = logic.buildSeedTriplets(pairs, radius1, "LPS World")
        self.assertEqual(len(triplets), 2)
        self.assertEqual(len(details), 2)
        self.assertAlmostEqual(triplets[0][2], radius1)
        self.assertAlmostEqual(triplets[1][2], radius2)
        # RAS (10, 20, 30) -> LPS (-10, -20, 30)
        self.assertEqual(triplets[0][0], [-10.0, -20.0, 30.0])
        self.assertEqual(triplets[0][1], [-11.0, -21.0, 31.0])
        self.assertEqual(triplets[1][0], [-40.0, -50.0, 60.0])

        ras_triplets, _ = logic.buildSeedTriplets(pairs, radius1, "RAS World")
        self.assertEqual(ras_triplets[0][0], [10.0, 20.0, 30.0])

        normalized = logic._normalizeSeedPairs(pairs, None, None, radius1)
        self.assertEqual(len(normalized), 2)
        self.assertEqual(max_branches_for_seeds(1, 4), 4)
        self.assertEqual(max_branches_for_seeds(10, 4), 10)
        self.assertEqual(max_branches_for_seeds(1, 0), 1)

        self.delayDisplay("Multiple seed triplet test completed")
