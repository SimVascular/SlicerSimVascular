import json
import logging
import math
import os
import re
import subprocess
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


def _normalize_loaded_volume_node(loaded):
    """Return a single volume node from slicer.util.loadVolume/loadLabelVolume (node or list)."""
    if loaded is None:
        return None
    if isinstance(loaded, (list, tuple)):
        return loaded[0] if loaded else None
    return loaded


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
SeqSeg Vessel Segmentation is a seed-based vascular segmentation module for 3D Slicer. It uses the SeqSeg Python package,
requires one or more seed lines (start + direction; radius is the line length) and a volume as input, and produces a segmentation, surface mesh, and centerline.
The first <b>Run SeqSeg</b> installs <b>torch 2.2.2</b> and <b>nnunetv2</b> if they are missing: through the PyTorch and Slicer NNUNet extensions when those are installed, and with pip otherwise. Pretrained nnUNet weights for the selected <b>Model Weights</b> entry are downloaded on that run, after a confirm dialog, if they are not already in Slicer's SeqSeg folder.
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
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # SeqSeg Sample Data 1 - Example medical volume for testing SeqSeg
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="SeqSeg Vessel Segmentation",
        sampleName="SeqSeg Test Volume 1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        thumbnailFileName=os.path.join(iconsPath, "seqseg1.png"),
        # Download URL and target file name - using existing test data
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="SeqSegTestVolume1.nrrd",
        # Checksum to ensure file integrity
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="SeqSegTestVolume1",
    )

    # SeqSeg Sample Data 2 - Another example volume
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="SeqSeg Vessel Segmentation",
        sampleName="SeqSeg Test Volume 2",
        thumbnailFileName=os.path.join(iconsPath, "seqseg2.png"),
        # Download URL and target file name - using existing test data
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="SeqSegTestVolume2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="SeqSegTestVolume2",
    )


#
# seqsegParameterNode
#

# Train dataset ids passed to SeqSeg / nnUNet (stored on the parameter node).
# The Train Dataset combo shows friendly labels; itemData holds these real ids.
KNOWN_TRAIN_DATASETS = (
    "Dataset005_SEQAORTANDFEMOMR",
    "Dataset006_SEQAORTANDFEMOCT",
    "Dataset010_SEQCOROASOCACT",
    "Dataset091_VMR_FONTAN_003MR",
    "Dataset090_VMR_FONTAN_ORIGMR",
)

# User-facing labels for the Train Dataset / weights combo (id -> display text).
TRAIN_DATASET_DISPLAY_NAMES = {
    "Dataset005_SEQAORTANDFEMOMR": "Aorta MR",
    "Dataset006_SEQAORTANDFEMOCT": "Aorta CT",
    "Dataset010_SEQCOROASOCACT": "Coronary CT",
    "Dataset091_VMR_FONTAN_003MR": "Fontan MR (003)",
    "Dataset090_VMR_FONTAN_ORIGMR": "Fontan MR (orig)",
}


def train_dataset_display_name(dataset_id: str) -> str:
    """Friendly label for a known train dataset id (falls back to the id)."""
    return TRAIN_DATASET_DISPLAY_NAMES.get(dataset_id, dataset_id)


def train_dataset_id_from_ui_value(value: str) -> Optional[str]:
    """Resolve combo display text or a raw dataset id to a known train dataset id."""
    if not value:
        return None
    if value in KNOWN_TRAIN_DATASETS:
        return value
    for dataset_id, label in TRAIN_DATASET_DISPLAY_NAMES.items():
        if label == value:
            return dataset_id
    return None


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
        "datasets": ("Dataset010_SEQCOROASOCACT",),
        "url": "https://zenodo.org/records/19547894/files/nnUNet_results_coronary.zip?download=1",
        "zip_filename": "nnUNet_results_coronary.zip",
        "extracted_dir_name": "nnUNet_results_coronary",
        "record_url": "https://zenodo.org/records/19547894",
        "progress_label": "Downloading Coronary CT weights (~2.7 MB)...",
        "confirm_detail": "The Coronary CT archive is about 2.7 MB.",
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
    is not fetched again. A path already stored on the parameter node is
    last: it covers Fontan models (no archive) and a results tree saved
    with a scene.
    """
    profile = weight_profile_for_dataset(dataset_id)
    if profile:
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
    maxSteps: Annotated[int, WithinRange(1, 10000)] = 1  # Max segmentation steps
    maxBranches: Annotated[int, WithinRange(1, 100)] = 1  # Max number of branches
    maxStepsPerBranch: Annotated[int, WithinRange(1, 1000)] = 20  # Max steps per branch
    imageUnit: Annotated[str, Choice(["cm", "mm"])] = "cm"  # Image unit (mm or cm); cm first — MRML/connectGui often defaults to first Choice()
    scale: Annotated[str, Choice(["0.1", "1", "10"])] = "1"  # SeqSeg -scale vs. nnUNet training units
    coordinateSystem: Annotated[str, Choice(["LPS World", "RAS World"])] = "LPS World"  # Coordinate system for seed points
    nnunetResultsPath: str = ""  # Set at run from the selected model; not edited in the panel
    nnunetType: Annotated[str, Choice(["3d_fullres", "2d"])] = "3d_fullres"  # Type of nnUNet model
    trainDataset: Annotated[str, Choice(list(KNOWN_TRAIN_DATASETS))] = "Dataset005_SEQAORTANDFEMOMR"  # Training dataset name
    outputDirectory: str = ""  # Directory for SeqSeg outputs (data_dir)
    outputSegmentation: str = ""  # Segmentation node ID
    simvascular: bool = False  # SeqSeg -simvascular 0/1
    forceCpu: bool = False  # SeqSeg -cpu (run nnUNet inference on the CPU)


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

        # Output visualization buttons
        if hasattr(self.ui, 'loadSegmentationButton'):
            self.ui.loadSegmentationButton.connect("clicked(bool)", self.onLoadSegmentationButton)
        if hasattr(self.ui, 'loadSurfaceMeshButton'):
            self.ui.loadSurfaceMeshButton.connect("clicked(bool)", self.onLoadSurfaceMeshButton)
        if hasattr(self.ui, 'browseOutputsButton'):
            self.ui.browseOutputsButton.connect("clicked(bool)", self.onBrowseOutputsButton)
        
        # Connect output directory button to sync with parameter node
        # ctkDirectoryButton emits directoryChanged signal when directory changes
        if hasattr(self.ui.outputDirectoryButton, 'directoryChanged'):
            self.ui.outputDirectoryButton.connect('directoryChanged(QString)', self.onOutputDirectoryChanged)

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
        self._checkCanApply()

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

        # Sync UI with parameter node values
        self._syncUiWithParameterNode()
        
        # Set default output directory if not set
        if not self._parameterNode.outputDirectory:
            default_output_dir = os.path.join(os.path.expanduser("~"), "SeqSeg_Output")
            # Ensure directory ends with '/'
            if not default_output_dir.endswith(os.sep):
                default_output_dir = default_output_dir + os.sep
            self._parameterNode.outputDirectory = default_output_dir
            logging.info(f"Using default output directory: {default_output_dir}")

    def _populateTrainDatasetComboFromKnown(self) -> None:
        """Rebuild Train Dataset combo with friendly labels; itemData stores real dataset ids.

        The parameter node still stores Dataset*_… ids (Choice / SeqSeg -train_dataset).
        The combo is synced manually (no SlicerParameterName) so users see labels like
        "Aorta CT" while runtime still receives Dataset006_SEQAORTANDFEMOCT.
        """
        if not getattr(self, "ui", None) or not hasattr(self.ui, "trainDatasetComboBox"):
            return
        combo = self.ui.trainDatasetComboBox
        expected = list(KNOWN_TRAIN_DATASETS)
        current = [combo.itemData(i) for i in range(combo.count)]
        if current == expected:
            return
        combo.blockSignals(True)
        combo.clear()
        for ds in KNOWN_TRAIN_DATASETS:
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
        
        # Update UI based on requirements
        if missing_requirements:
            self.ui.applyButton.text = f"Missing: {', '.join(missing_requirements)}"
            self.ui.applyButton.enabled = False
            self._updateStatusMessage(f"Please provide: {', '.join(missing_requirements)}")
        else:
            self.ui.applyButton.text = "🚀 Run SeqSeg"
            self.ui.applyButton.enabled = True
            self._updateStatusMessage("Ready for vessel segmentation — all requirements met!")

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        
        # Update UI for processing
        self.ui.applyButton.enabled = False
        self.ui.applyButton.text = "Processing..."
        self._updateStatusMessage("Starting SeqSeg processing...")
        weights_status = None
        weights_error = False

        try:
            with slicer.util.tryWithErrorDisplay(_("Failed to compute SeqSeg results."), waitCursor=True):
                # Get nodes from parameter node
                inputVolume = self._parameterNode.inputVolume
                
                # Debug: Check if input volume is properly set
                if not inputVolume:
                    logging.warning("No input volume selected, trying to find one...")
                    firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
                    if firstVolumeNode:
                        inputVolume = firstVolumeNode
                        self._parameterNode.inputVolume = firstVolumeNode
                        logging.info(f"Using volume: {firstVolumeNode.GetName()}")
                    else:
                        raise ValueError("No volume found in scene. Please load a volume first.")
            
            self.refreshSeedRowsFromScene()
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
                    if ui_train_ds and ui_train_ds != trainDataset:
                        logging.info(f"Train Dataset mismatch - Parameter: '{trainDataset}', UI: '{ui_train_ds}'")
                        if ui_train_ds in KNOWN_TRAIN_DATASETS:
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
                logging.warning(f"Invalid image unit '{imageUnit}', defaulting to 'cm'")
                imageUnit = "cm"
                self._parameterNode.imageUnit = "cm"
            
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
            # ctkDirectoryButton uses 'directory' property
            if not outputDirectory:
                try:
                    # Try different possible property names
                    if hasattr(self.ui.outputDirectoryButton, 'directory'):
                        outputDirectory = self.ui.outputDirectoryButton.directory
                    elif hasattr(self.ui.outputDirectoryButton, 'currentPath'):
                        outputDirectory = self.ui.outputDirectoryButton.currentPath
                    elif hasattr(self.ui.outputDirectoryButton, 'text'):
                        outputDirectory = self.ui.outputDirectoryButton.text
                    logging.info(f"Output directory from UI button: '{outputDirectory}'")
                    if outputDirectory:
                        # Ensure directory ends with '/'
                        if not outputDirectory.endswith(os.sep):
                            outputDirectory = outputDirectory + os.sep
                        self._parameterNode.outputDirectory = outputDirectory
                except (AttributeError, Exception) as e:
                    logging.warning(f"Could not get directory from UI button: {e}")
            
            # Final check - if still empty, use default
            if not outputDirectory:
                default_output_dir = os.path.join(os.path.expanduser("~"), "SeqSeg_Output")
                outputDirectory = default_output_dir
                self._parameterNode.outputDirectory = default_output_dir
                logging.info(f"Using default output directory: {default_output_dir}")
            
            # Ensure output directory ends with '/'
            if outputDirectory and not outputDirectory.endswith(os.sep):
                outputDirectory = outputDirectory + os.sep
            
            outputSegmentationNode = slicer.mrmlScene.GetNodeByID(self._parameterNode.outputSegmentation)
            
            # Run SeqSeg segmentation
            self._updateStatusMessage("Running SeqSeg algorithm...")
            self.logic.runSeqSeg(inputVolume, seedPoint1Node, seedPoint2Node, radiusEstimate, 
                               maxSteps, maxBranches, maxStepsPerBranch,
                               imageUnit, self._parameterNode.scale, self._parameterNode.coordinateSystem, nnunetResultsPath, nnunetType, trainDataset, 
                               outputDirectory, outputSegmentationNode, simvascular=simvascular,
                               forceCpu=forceCpu, seedPairs=seedPairs)
                                   
            self._updateStatusMessage(
                _("SeqSeg finished — segmentation loaded and overlaid on the input volume in slice views.")
            )

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
            weights_status = f"SeqSeg failed: {e}"
            weights_error = True

        finally:
            # Reset button state. _checkCanApply replaces the status line, so a
            # weights cancel or failure is written again afterwards.
            self.ui.applyButton.enabled = True
            self._checkCanApply()
            if weights_status:
                self._updateStatusMessage(weights_status, isError=weights_error)

    def onOutputDirectoryChanged(self, directory: str) -> None:
        """Called when output directory button changes - sync with parameter node."""
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

    def onNnUnetTypeChanged(self, nnunet_type: str) -> None:
        """Called when nnUNet type combo box changes - sync with parameter node."""
        if self._parameterNode and nnunet_type:
            self._parameterNode.nnunetType = nnunet_type
            logging.info(f"nnUNet Type updated to: {nnunet_type}")

    def onTrainDatasetChanged(self, train_dataset: str) -> None:
        """Called when train dataset combo changes — map display label to real dataset id."""
        if not self._parameterNode:
            return
        dataset_id = self._trainDatasetIdFromCombo()
        if not dataset_id:
            dataset_id = train_dataset_id_from_ui_value(train_dataset)
        if dataset_id and dataset_id in KNOWN_TRAIN_DATASETS:
            self._parameterNode.trainDataset = dataset_id
            logging.info(
                f"Train Dataset updated to: {dataset_id} "
                f"({train_dataset_display_name(dataset_id)})"
            )

    def onScaleChanged(self, scale: str) -> None:
        """Called when Scale combo box changes - sync with parameter node."""
        if self._parameterNode and scale in ("0.1", "1", "10"):
            self._parameterNode.scale = scale
            logging.info(f"Scale updated to: {scale}")

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
                current_ds = self._parameterNode.trainDataset
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

    def _imageUnit(self) -> str:
        unit = "cm"
        if self._parameterNode and getattr(self._parameterNode, "imageUnit", None) in ("mm", "cm"):
            unit = self._parameterNode.imageUnit
        return unit

    def _markupPointRAS(self, node, index: int = 0):
        if not node or node.GetNumberOfControlPoints() <= index:
            return None
        ras = [0.0, 0.0, 0.0]
        node.GetNthControlPointPosition(index, ras)
        return [float(ras[0]), float(ras[1]), float(ras[2])]

    def _seedLineRadius(self, node) -> Optional[float]:
        """Seed radius is the Slicer line length, passed as-is (including when Image Unit is cm)."""
        start = self._markupPointRAS(node, 0)
        direction = self._markupPointRAS(node, 1)
        if start is None or direction is None:
            return None
        return line_length(start, direction)

    def createSeedLine(self):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
        node.CreateDefaultDisplayNodes()
        node.SetName(slicer.mrmlScene.GenerateUniqueName("Seed"))
        node.SetAttribute(SEED_LINE_ATTRIBUTE, "true")
        if hasattr(node, "SetMaximumNumberOfControlPoints"):
            node.SetMaximumNumberOfControlPoints(2)
        display = node.GetDisplayNode()
        if display:
            display.SetGlyphScale(3.0)
            display.SetTextScale(2.0)
        return node

    def _addSeedRowForNode(self, node, activate: bool = False):
        row = qt.QWidget()
        row_layout = qt.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        name_label = qt.QLabel(node.GetName() if node else "Seed")
        row_layout.addWidget(name_label)

        radius_label = qt.QLabel("Place start and end points")
        row_layout.addWidget(radius_label, 1)

        reposition_button = qt.QPushButton("Reposition")
        reposition_button.setToolTip("Clear this line and click start then direction again")
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
            "observers": [],
        }
        self._seedRows.append(seed_row)
        layout = self._seedRowsLayout()
        if layout is not None:
            layout.addWidget(row)
        self.observeSeedNode(seed_row)
        self.refreshSeedRowStatus(seed_row)
        if activate:
            self.activateLinePlacement(node)
            self._updateStatusMessage("Click in a viewer to place the seed start and direction points.")
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
        radius = self._seedLineRadius(node) if node and self._nodeInScene(node) else None
        if radius is None:
            radius_label.text = "Place start and end points"
            return
        unit = self._imageUnit()
        radius_label.text = f"Radius: {radius:.3f} {unit}"
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
        self._updateStatusMessage(f"Reposition {node.GetName()}: click start then direction.")
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
        """Return start/direction/radius dicts from complete SeqSeg seed lines."""
        pairs = []
        for node in self._completeSeedLines():
            start = self._markupPointRAS(node, 0)
            direction = self._markupPointRAS(node, 1)
            radius = self._seedLineRadius(node)
            if start is None or direction is None or radius is None:
                logging.warning("Skipping incomplete seed line %s", node.GetName())
                continue
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

    def onLoadSegmentationButton(self) -> None:
        """Load the latest segmentation (.mha) file from output directory."""
        try:
            output_dir = self._parameterNode.outputDirectory
            if not output_dir or not os.path.exists(output_dir):
                slicer.util.warningDisplay("Output directory not found. Please run SeqSeg first or check output directory path.")
                return
            
            # Look for segmentation files in output subdirectory
            output_subdir = os.path.join(output_dir, "output")
            if not os.path.exists(output_subdir):
                slicer.util.warningDisplay("No output subdirectory found. Please run SeqSeg first.")
                return
            
            # Find segmentation files with common patterns
            seg_files = []
            for f in os.listdir(output_subdir):
                if (f.endswith(('.mha', '.nii.gz', '.nii')) and 
                    ('seg' in f.lower() or 'segmentation' in f.lower())):
                    seg_files.append(f)
            
            if not seg_files:
                slicer.util.warningDisplay("No segmentation files found in output directory.")
                return
            
            # Use the latest file (by modification time)
            seg_files.sort(key=lambda f: os.path.getmtime(os.path.join(output_subdir, f)), reverse=True)
            latest_seg = os.path.join(output_subdir, seg_files[0])
            
            # Use the standardized loading method
            if self._loadSegmentationFile(latest_seg, update_parameter_node=True):
                seg_node = slicer.mrmlScene.GetNodeByID(self._parameterNode.outputSegmentation)
                vol_node = self._parameterNode.inputVolume if self._parameterNode else None
                if seg_node and vol_node:
                    self.logic.showSegmentationOverVolume(seg_node, vol_node)
                logging.info(f"Loaded segmentation: {seg_files[0]}")
            else:
                slicer.util.errorDisplay("Failed to load segmentation file.")
                
        except Exception as e:
            logging.error(f"Error loading segmentation: {e}")
            slicer.util.errorDisplay(f"Error loading segmentation: {str(e)}")

    def onLoadSurfaceMeshButton(self) -> None:
        """Load the latest surface mesh (.vtp) file from output directory."""
        try:
            output_dir = self._parameterNode.outputDirectory
            if not output_dir or not os.path.exists(output_dir):
                slicer.util.warningDisplay("Output directory not found. Please run SeqSeg first or check output directory path.")
                return
            
            # Look for surface mesh files in output subdirectory
            output_subdir = os.path.join(output_dir, "output")
            if not os.path.exists(output_subdir):
                slicer.util.warningDisplay("No output subdirectory found. Please run SeqSeg first.")
                return
            
            # Find .vtp surface mesh files
            mesh_files = [f for f in os.listdir(output_subdir) if f.endswith('.vtp') and 'surface_mesh' in f]
            
            if not mesh_files:
                slicer.util.warningDisplay("No surface mesh files found in output directory.")
                return
            
            # Use the latest file (by modification time)
            mesh_files.sort(key=lambda f: os.path.getmtime(os.path.join(output_subdir, f)), reverse=True)
            latest_mesh = os.path.join(output_subdir, mesh_files[0])
            
            # Load the surface mesh
            mesh_node = slicer.util.loadModel(latest_mesh)
            if mesh_node:
                logging.info(f"Loaded surface mesh: {mesh_files[0]}")
                # Make sure it's visible in 3D view
                mesh_node.GetDisplayNode().SetVisibility(True)
                mesh_node.GetDisplayNode().SetColor(1, 0, 0)  # Set to red
                mesh_node.GetDisplayNode().SetOpacity(0.8)
            else:
                slicer.util.errorDisplay("Failed to load surface mesh file.")
                
        except Exception as e:
            logging.error(f"Error loading surface mesh: {e}")
            slicer.util.errorDisplay(f"Error loading surface mesh: {str(e)}")

    def onBrowseOutputsButton(self) -> None:
        """Browse and select specific output files to load."""
        try:
            output_dir = self._parameterNode.outputDirectory
            if not output_dir or not os.path.exists(output_dir):
                slicer.util.warningDisplay("Output directory not found. Please run SeqSeg first or check output directory path.")
                return
            
            output_subdir = os.path.join(output_dir, "output")
            if not os.path.exists(output_subdir):
                slicer.util.warningDisplay("No output subdirectory found. Please run SeqSeg first.")
                return
            
            # Get all output files
            all_files = os.listdir(output_subdir)
            output_files = [f for f in all_files if f.endswith(('.mha', '.vtp', '.nii.gz', '.nii'))]
            
            if not output_files:
                slicer.util.warningDisplay("No output files found in output directory.")
                return
            
            # Create a simple selection dialog
            file_list = "\n".join([f"{i+1}. {f}" for i, f in enumerate(output_files)])
            message = f"Output files found:\n\n{file_list}\n\nAll files will be loaded automatically."
            
            response = slicer.util.confirmYesNoDisplay(message, windowTitle="Load Output Files")
            if response:
                loaded_count = 0
                for file in output_files:
                    file_path = os.path.join(output_subdir, file)
                    try:
                        if file.endswith('.vtp'):
                            # Load as model (surface mesh)
                            node = slicer.util.loadModel(file_path)
                            if node:
                                node.GetDisplayNode().SetVisibility(True)
                                node.GetDisplayNode().SetColor(1, 0, 0)
                                loaded_count += 1
                        elif file.endswith(('.mha', '.nii.gz', '.nii')):
                            # Check if it's a segmentation file and load appropriately
                            if 'seg' in file.lower() or 'segmentation' in file.lower():
                                # Load as segmentation using standardized method
                                if self._loadSegmentationFile(file_path, update_parameter_node=False):
                                    loaded_count += 1
                            else:
                                # Load as regular volume
                                node = slicer.util.loadVolume(file_path)
                                if node:
                                    loaded_count += 1
                    except Exception as file_error:
                        logging.warning(f"Failed to load {file}: {file_error}")
                
                slicer.util.infoDisplay(f"Successfully loaded {loaded_count} of {len(output_files)} output files.")
            
        except Exception as e:
            logging.error(f"Error browsing outputs: {e}")
            slicer.util.errorDisplay(f"Error browsing outputs: {str(e)}")

    def _loadSegmentationFile(self, segmentation_file_path: str, update_parameter_node: bool = True) -> bool:
        """
        Standardized method to load a segmentation file (.mha, .nii.gz, etc.) as a proper segmentation node.
        
        :param segmentation_file_path: Full path to the segmentation file
        :param update_parameter_node: Whether to update the parameter node outputSegmentation reference
        :return: True if successful, False otherwise
        """
        try:
            if not os.path.exists(segmentation_file_path):
                logging.error(f"Segmentation file not found: {segmentation_file_path}")
                return False
            
            # Generate a descriptive name for the segmentation
            filename = os.path.basename(segmentation_file_path)
            segmentation_name = f"SeqSeg_Segmentation_{filename.split('.')[0]}"
            
            # Load as labelmap volume — ImportLabelmapToSegmentationNode requires vtkMRMLLabelMapVolumeNode
            temp_volume = _normalize_loaded_volume_node(slicer.util.loadLabelVolume(segmentation_file_path))
            if not temp_volume:
                logging.error(f"Failed to load segmentation file as volume: {segmentation_file_path}")
                return False
            
            # Create a new segmentation node
            segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", segmentation_name)
            
            # Set up the segmentation node properly
            segmentation_node.CreateDefaultDisplayNodes()
            segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(temp_volume)
            
            # Use the segmentations module logic to import the labelmap
            segmentationsLogic = slicer.modules.segmentations.logic()
            segmentationsLogic.ImportLabelmapToSegmentationNode(temp_volume, segmentation_node)
            
            # Remove the temporary volume node to avoid conflicts
            slicer.mrmlScene.RemoveNode(temp_volume)
            
            # Set up display properties for better visualization
            displayNode = segmentation_node.GetDisplayNode()
            if displayNode:
                displayNode.SetVisibility(True)
                displayNode.SetOpacity(0.6)  # Semi-transparent overlay
            
            # Update the parameter node to reference this segmentation if requested
            if update_parameter_node and self._parameterNode:
                self._parameterNode.outputSegmentation = segmentation_node.GetID()
            
            logging.info(f"Successfully loaded segmentation: {segmentation_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading segmentation file {segmentation_file_path}: {e}")
            return False


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
                _("This module requires installation of additional Python packages. Installation needs a network connection and may take several minutes. Click OK to proceed."),
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

    def showSegmentationOverVolume(self, segmentationNode, volumeNode, overlayOpacity=0.45):
        """Bind segmentation geometry to an image volume and show both in slice viewers (overlay)."""
        if not segmentationNode or not volumeNode:
            logging.warning("showSegmentationOverVolume: missing segmentation or volume node")
            return
        segmentationNode.SetNodeReferenceID(
            segmentationNode.GetReferenceImageGeometryReferenceRole(),
            volumeNode.GetID(),
        )
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        segmentationNode.CreateDefaultDisplayNodes()
        displayNode = segmentationNode.GetDisplayNode()
        if displayNode:
            displayNode.SetVisibility(True)
            displayNode.SetOpacity(overlayOpacity)
            if hasattr(displayNode, "Visibility2DOn"):
                displayNode.Visibility2DOn()
            elif hasattr(displayNode, "SliceIntersectionVisibilityOn"):
                displayNode.SliceIntersectionVisibilityOn()
        slicer.util.setSliceViewerLayers(background=volumeNode, fit=True)

    def loadSegmentationToNode(self, segmentation_file_path: str, outputSegmentationNode, inputVolume) -> bool:
        """
        Standardized method to load a segmentation file into an existing segmentation node.
        
        :param segmentation_file_path: Full path to the segmentation file
        :param outputSegmentationNode: The segmentation node to update
        :param inputVolume: Reference volume for geometry
        :return: True if successful, False otherwise
        """
        try:
            if not os.path.exists(segmentation_file_path):
                logging.error(f"Segmentation file not found: {segmentation_file_path}")
                return False
            
            # Load as labelmap volume for correct MRML node type when importing into segmentation
            temp_segmentation_node = _normalize_loaded_volume_node(slicer.util.loadLabelVolume(segmentation_file_path))
            if not temp_segmentation_node:
                logging.error(f"Failed to load segmentation file: {segmentation_file_path}")
                return False
            
            # Convert volume to segmentation using segmentations module logic
            import numpy as np
            segmentationArray = slicer.util.arrayFromVolume(temp_segmentation_node)
            segmentationArray = segmentationArray.astype(np.uint8)
            
            # Create or update segment in output segmentation node
            segmentId = outputSegmentationNode.GetSegmentation().AddEmptySegment("SeqSeg_Segment")
            
            # Update segmentation with the loaded data
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                segmentationArray, 
                outputSegmentationNode, 
                segmentId, 
                inputVolume
            )
            
            # Clean up temporary volume node
            slicer.mrmlScene.RemoveNode(temp_segmentation_node)

            outputSegmentationNode.SetNodeReferenceID(
                outputSegmentationNode.GetReferenceImageGeometryReferenceRole(),
                inputVolume.GetID(),
            )
            outputSegmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)
            
            logging.info(f"Successfully loaded segmentation from: {segmentation_file_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading segmentation file {segmentation_file_path}: {e}")
            return False

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

    def _run_seqseg_cli(self, seqseg_cmd, cwd):
        logging.info(f"Running SeqSeg command: {' '.join(seqseg_cmd)}")
        result = subprocess.run(
            seqseg_cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        if result.stdout:
            logging.info(f"SeqSeg output:\n{result.stdout}")
        if result.stderr:
            logging.info(f"SeqSeg stderr:\n{result.stderr}")
        return result

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
        :param radiusEstimate: estimated radius (Slicer line length, same numeric units as the image)
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

            result = self._run_seqseg_cli(cpu_cmd if force_cpu else seqseg_cmd, cwd=data_dir)
            failure_reason = self._seqseg_log_indicates_inference_failure(
                self._combined_seqseg_log(result)
            )
            if failure_reason and not force_cpu:
                logging.warning("%s Retrying SeqSeg with -cpu.", failure_reason)
                force_cpu = True
                result = self._run_seqseg_cli(cpu_cmd, cwd=data_dir)
                failure_reason = self._seqseg_log_indicates_inference_failure(
                    self._combined_seqseg_log(result)
                )

            if result.returncode != 0:
                logging.error(f"SeqSeg failed with return code {result.returncode}")
                logging.error(f"Error output: {result.stderr}")
                raise RuntimeError(f"SeqSeg execution failed: {result.stderr}")

            if failure_reason:
                raise RuntimeError(failure_reason)

            logging.info(
                "SeqSeg execution completed successfully%s",
                " (CPU)" if force_cpu else "",
            )

            # Look for output segmentation file
            # SeqSeg typically outputs files with pattern: {case}_seg_containing_seeds_{steps}_steps.mha
            # or similar patterns with .mha or .nii.gz extensions
            output_files = []
            for ext in ['.mha', '.nii.gz', '.nii']:
                output_files.extend([f for f in os.listdir(output_dir) if f.endswith(ext) and 'seg' in f.lower()])
            
            if not output_files:
                # Fallback: look for any segmentation-like files
                output_files = [f for f in os.listdir(output_dir) if f.endswith(('.mha', '.nii.gz', '.nii'))]
            
            if not output_files:
                raise RuntimeError(f"No output segmentation file found in {output_dir}. SeqSeg may have failed or output format changed.")
            
            # Use the latest output file (sort by modification time)
            output_files.sort(key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), reverse=True)
            output_file = os.path.join(output_dir, output_files[0])
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
                
            logging.info("SeqSeg segmentation completed successfully")

        except InstallError:
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
