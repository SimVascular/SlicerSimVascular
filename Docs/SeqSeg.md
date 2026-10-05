<p align="center">
  <img src="../SeqSeg/Resources/Icons/seqseg_logo.png" alt="SeqSeg Vessel Segmentation logo" width="300">
</p>

# SeqSeg Vessel Segmentation (seqseg)

![Example coronary segmentation in 3D Slicer](../SeqSeg/Resources/screenshot/coronary.png)

**SeqSeg Vessel Segmentation** is a [3D Slicer](https://www.slicer.org/) module (internal module id: `seqseg`), listed in the module menu under both SimVascular and Segmentation. It is aimed at clinicians and researchers who want to segment tubular structures (for example coronary or other vessels) from CT or MR without drawing full contours. You place two seed points and a radius hint; the module runs the [SeqSeg](https://github.com/numisveinsson/SeqSeg) Python workflow with nnUNet models and loads the result back into Slicer. **Pretrained model weights are not something you train yourself:** in the module’s **nnUNet Configuration** section you fetch them with **built-in download buttons** (Slicer may ask for a save folder once); no command-line nnUNet setup is required for typical use.

The module selects a volume, two fiducial markups, optional **one-click** nnUNet weight downloads and paths, runs the SeqSeg CLI, then loads segmentation (.mha) and surface (.vtp) outputs.

## Publication

If you use the SeqSeg method, Python package, or this extension in research, please cite:

Sveinsson Cepero, N., Shadden, S.C. SeqSeg: Learning Local Segments for Automatic Vascular Model Construction. *Ann Biomed Eng* **53**, 158–179 (2025). [https://doi.org/10.1007/s10439-024-03611-z](https://doi.org/10.1007/s10439-024-03611-z)

## Prerequisites

### Slicer extensions (required before Python inference works)

Install these from **View → Extension Manager** (search by name and install):

- **[PyTorch](https://slicer.readthedocs.io/en/latest/user_guide/extensions_manager.html)** (catalog name usually **PyTorch**): supplies **`PyTorchUtils`**, used to install a compatible **`torch`** / **`torchvision`** build into Slicer’s Python.
- **NNUNet** (catalog name usually **NNUNet**; Python module **`SlicerNNUNetLib`**): supplies **`nnunetv2`** installation and version checks aligned with other nnUNet-based Slicer extensions.

Install **PyTorch** and **NNUNet** manually from the Extension Manager before running SeqSeg. If Slicer prompts for a restart after installing either extension, restart Slicer before opening **SeqSeg Vessel Segmentation**.

Without both extensions loaded, **Run SeqSeg** stops with an error until they are installed (then restart Slicer if prompted).

The SimVascular extension does **not** declare **PyTorch** and **NNUNet** in `EXTENSION_DEPENDS`. SeqSeg is the only module here that needs them, and declaring them would make the Extension Manager install PyTorch for every SimVascular user, including those who only use the meshing modules. The scripted module does not list them as hard Qt module dependencies either (same idea as Total Segmentator): otherwise Slicer would refuse to open SeqSeg until PyTorch loads, and you would not get the in-module dependency checks.

### First-time Python setup

The **SeqSeg Vessel Segmentation** module does **not** ship PyTorch or nnUNet wheels inside its own installer. Before running SeqSeg:

1. Install **PyTorch** from Slicer Extensions and use it to provide **`torch` 2.2.2** / **`torchvision` 0.17.2** in Slicer’s Python. **`torch` must be pinned to `==2.2.2`** (with the matching **`torchvision==0.17.2`**)—other versions are not supported. In the **PyTorch Util** module set the requested version to **`2.2.2`** before installing, or install manually with **`torch==2.2.2`**.
2. Install **NNUNet** from Slicer Extensions and use it to provide **`nnunetv2` 2.5.1** in Slicer’s Python.
3. On first **Run SeqSeg**, the module may ask you to confirm installation of extra Python packages (network required). It installs the **`seqseg`** PyPI package (**`seqseg>=2.1.1`**, the latest release on PyPI) using **`slicer.util.pip_install`**, first with **`--no-deps`**, then installs declared dependencies selectively—**skipping** packages that must stay under Slicer’s control (**SimpleITK**, **torch**, **torchvision**, **nnunetv2**, **requests**, **rt_utils**), same pattern as extensions such as **Total Segmentator**. Later runs check PyPI and upgrade **`seqseg`** if a newer version is published.

You may need to **restart Slicer** once after PyTorch or other packages are installed; follow any prompt the module shows.

### Neural network weights (pretrained models)

Use the **Download Aorta Weights (CT/MR)** and **Download Coronary CT Weights** buttons inside the module’s nnUNet section. You choose a parent folder when asked; the extension downloads from Zenodo, unpacks, and can set **nnUNet Results Path** automatically. You do **not** need to train networks or copy files by hand unless you use your own custom-trained models.

### Manual / advanced installation

Use **Slicer’s** Python interpreter, not your system `python`. Example (adjust path to your Slicer install):

```bash
/path/to/Slicer-X.Y.Z/bin/PythonSlicer -m pip install --upgrade "seqseg>=2.1.1"
```

You still need the **PyTorch** and **Slicer NNUNet** extensions and their installers to provide **`torch`** and **`nnunetv2`** consistently.

## Tutorial sample data

You can practice the workflow on **CTA-cardio**, a cardiac CT angiography–style volume from the [Slicer testing data](https://github.com/Slicer/SlicerTestingData) collection:

- **File**: `CTA-cardio.nrrd`
- **SHA256**: `3b0d4eb1a7d8ebb0c5a89cc0504640f76a030b4e869e33ff34c564c3d3b88ad2`
- **Download**: [https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/3b0d4eb1a7d8ebb0c5a89cc0504640f76a030b4e869e33ff34c564c3d3b88ad2](https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/3b0d4eb1a7d8ebb0c5a89cc0504640f76a030b4e869e33ff34c564c3d3b88ad2)

Browsers often save that URL as a file named only with the SHA256 hash and **no extension**. Before loading in Slicer, **rename the file** on disk to end with **`.nrrd`** (for example `CTA-cardio.nrrd`) so Slicer recognizes it as NRRD.

In 3D Slicer, use **File → Add Data**, paste the download URL, or choose the renamed file; confirm the checksum if prompted.

**For this tutorial**, use the **CT aorta** pipeline: in **nnUNet Configuration**, click the **Download Aorta Weights (CT/MR)** button; after the download finishes, **nnUNet Results Path** is usually filled in for you (otherwise pick the folder that contains `nnUNet_results`). When the module asks, choose **CT** so **Train Dataset** is set to `Dataset006_SEQAORTANDFEMOCT` (you can also pick that value directly from the **Train Dataset** dropdown). Do not use the separate coronary CTA button for this walkthrough (see Usage below).

### Placing seed points on CTA-cardio (CT aorta)

After the volume is in Slicer, open **SeqSeg Vessel Segmentation**, click **Auto-Create Seeds**, then **toggle on Planes visible in 3D** for the seed markups (so slice intersections show in the 3D view). Zoom into the anatomy and **drag the seed points** until they sit where you want (inside the vessel).

Whether you used **Auto-Create Seeds** or the **Markups** module, you need two fiducials in the vessel. Place **Seed point 1** in the **ascending aorta** near the **aortic root / proximal ascending** segment, and **Seed point 2** **higher** along the same lumen toward the **arch**—both inside the contrast-filled vessel. Seed 1 is the start; seed 2 sets the initial tracking direction along the aorta.

**Seed point 1** (red markup **SeqSeg Seed Point 1**), centered in the ascending aorta on axial / coronal / sagittal slices:

![CTA-cardio tutorial: seed point 1 in the ascending aorta](../SeqSeg/Resources/screenshot/seed_1.png)

**Seed point 2** (green markup **SeqSeg Seed Point 2**), placed superiorly along the ascending aorta relative to seed 1 (example slice positions in the 2D viewers):

![CTA-cardio tutorial: seed point 2 higher in the ascending aorta](../SeqSeg/Resources/screenshot/seed_2.png)

### Tutorial module settings (CTA-cardio, CT aorta)

Before **Run SeqSeg**, align **Inputs**, **Segmentation Limits**, and **nnUNet Configuration** with the tutorial setup below. Example from the module UI:

![CTA-cardio tutorial: recommended Inputs, Segmentation Limits, and nnUNet settings](../SeqSeg/Resources/screenshot/settings_screenshot.png)

- **Image Unit** = **mm** and **Scale** = **0.1**: the bundled deep-learning weights for this walkthrough were trained with **centimeter**–style image spacing; typical Slicer CT volumes (including CTA-cardio) use **millimeter** voxel spacing. Choosing **mm** for the image and **0.1** for **Scale** maps the volume to the physical scale the model expects. Wrong combinations here are a common cause of poor or misaligned results (the in-module Scale explanation under **nnUNet Configuration** describes the same rule in general form).
- **nnUNet Type** = **2d**: use the **2D** model for **faster inference** while following the tutorial (especially on a laptop or for a first run). **3d_fullres** is available if you prefer the full-resolution 3D configuration and can accept longer run times.

Other fields in the screenshot (e.g. **Train Dataset** `Dataset006_SEQAORTANDFEMOCT`, radius **10** (mm, same as volume unit), **Nr. of Steps** **2**) are reasonable starting points for this dataset; adjust if your machine or case needs it.

## Usage

1. **Load your volume**: Use the Data module or File menu to load your medical imaging volume, or the [tutorial sample data](#tutorial-sample-data) above (`CTA-cardio.nrrd`).

2. **Create and refine seed points** (typical workflow after loading):
   - Open **SeqSeg Vessel Segmentation** and select your volume when you get to that step, then click **Auto-Create Seeds** so the module creates the two fiducial markups for you.
   - **Toggle on Planes visible in 3D** for those seed markups (in the markups display / 3D view options) so you can see how the seeds line up with the volume in 3D.
   - Zoom the 3D view (and use the slice viewers as needed) and **drag the seed points** to move them onto the vessel centerline where you want them.
   - **Alternative (manual):** In the **Markups** module, create two fiducial points with **Add** and place them in the slice viewers; you can still enable **Planes visible in 3D** and drag points the same way.
   - For **CTA-cardio** and **CT aorta**, follow [Placing seed points on CTA-cardio (CT aorta)](#placing-seed-points-on-cta-cardio-ct-aorta) for where to put each seed.

3. **Finish settings and run SeqSeg** (in **SeqSeg Vessel Segmentation**, which you already opened for seeds):
   - Ensure **PyTorch** and **Slicer NNUNet** are installed from the Extension Manager ([Prerequisites](#prerequisites)).
   - Confirm **Input Volume** and both **Seed Point** markups match what you set up above (open the module from the module menu under SimVascular or Segmentation if you closed it—you can search for *vessel*, *segmentation*, or *SeqSeg*).
   - Set the radius estimate (approximate size of the structure you want to segment)
   - For **CTA-cardio** + **CT aorta**, set **Image Unit** to **mm**, **Scale** to **0.1**, and consider **nnUNet Type** **2d** as in [Tutorial module settings (CTA-cardio, CT aorta)](#tutorial-module-settings-cta-cardio-ct-aorta)
   - Configure **nnUNet** (expand **nnUNet Configuration** if needed—it opens expanded by default):
     - **Getting pretrained weights (easy path):** you **do not** train models or use the terminal. Press **Download Aorta Weights (CT/MR)** or **Download Coronary CT Weights**; Slicer asks for a parent directory, then the extension downloads from [Zenodo](https://zenodo.org/records/15020477) (aorta, ~236 MB) or [Zenodo](https://zenodo.org/records/19547894) (coronary CT, ~2.7 MB), unpacks, and typically sets **nnUNet Results Path** for you (`nnUNet_results` or `nnUNet_results_coronary` under the folder you chose; default parent is often `~/SeqSeg_Weights`). After the aorta download, answer the MR vs CT prompt so **Train Dataset** matches your images (the **Train Dataset** dropdown is updated accordingly); the coronary button selects **Dataset010_SEQCOROASOCACT** when used.
     - **Advanced:** if you already have nnUNet results on disk, set **nnUNet Results Path** with the folder control instead of the buttons.
     - Choose nnUNet type (3d_fullres or 2d)
     - **Train Dataset** (dropdown): choose the dataset id that matches your weights—`Dataset005_SEQAORTANDFEMOMR` (aorta MR), `Dataset006_SEQAORTANDFEMOCT` (aorta CT), or `Dataset010_SEQCOROASOCACT` (coronary CT lumen). The aorta models (005 and 006) were trained with **cm-scale** voxel spacing; the coronary model (010) was trained with **mm-scale** spacing—align **Image Unit** (under **Inputs**) and **Scale** (**nnUNet Configuration**) with your volume and that training convention.
   - Create or select an output segmentation node
   - Click "Run SeqSeg"

## Module Interface

### Inputs
- **Input Volume**: The medical imaging volume to segment (NIfTI format recommended)
- **Image Unit**: Physical unit of the volume spacing as interpreted by the module (**cm** or **mm**); must be consistent with **Scale** in **nnUNet Configuration** for the selected **Train Dataset**
- **Seed Point 1**: First markups fiducial node (must contain at least one point)
- **Seed Point 2**: Second markups fiducial node (must contain at least one point)
- **Radius Estimate**: Estimated radius of the structure to segment in millimeters (range: 0.1 - 100 mm)

### Segmentation Limits
- **Nr. of Steps**: Maximum number of tracking steps (1 - 10000, default: 10)
- **Max Branches**: Maximum number of vessel branches to follow (1 - 100, default: 1). Raised automatically to at least the number of complete seeds so extra seed lines are used as starts.
- **Max Steps per Branch**: Maximum steps per vessel branch (1 - 1000, default: 5)

### nnUNet Configuration
This section opens **expanded** by default. **Pretrained weights are meant to be installed with the two download buttons** (not by hunting files on the web unless you prefer to): click, pick a save location when prompted, wait for unpack; the module wires **nnUNet Results Path** when it can.
- **nnUNet Results Path**: Folder that contains the `nnUNet_results` (or `nnUNet_results_coronary`) tree. Filled automatically after a successful button download, or set manually if you maintain your own models.
  - **Download Aorta Weights (CT/MR)** (button): One-click fetch of aorta weights (CT/MR), 236.3 MB, from [Zenodo](https://zenodo.org/records/15020477). Default parent folder is often `~/SeqSeg_Weights` with extraction to `nnUNet_results`. After a new download, the module asks whether **Train Dataset** should be MR (`Dataset005_SEQAORTANDFEMOMR`) or CT (`Dataset006_SEQAORTANDFEMOCT`), and updates the **Train Dataset** dropdown to match.
  - **Download Coronary CT Weights** (button): One-click fetch of coronary CTA lumen weights (~2.7 MB) from [Zenodo](https://zenodo.org/records/19547894). Extracts beside your chosen parent as `nnUNet_results_coronary`. **Train Dataset** is set to `Dataset010_SEQCOROASOCACT` when you use this download (including when reusing an existing folder).
- **nnUNet Type**: Type of nnUNet model (3d_fullres or 2d, default: 3d_fullres)
- **Train Dataset** (dropdown): Fixed choices `Dataset005_SEQAORTANDFEMOMR` (aorta MR), `Dataset006_SEQAORTANDFEMOCT` (aorta CT), and `Dataset010_SEQCOROASOCACT` (coronary CT lumen). Must match the model tree under **nnUNet Results Path**. **Training units:** 005 and 006 were trained on **cm-scale** data; 010 was trained on **mm-scale** data (the module shows this hint under the dropdown).
- **Device** → **Force CPU (no GPU)** (checkbox, default off): Runs nnUNet inference on the CPU by adding **`-cpu`** to the SeqSeg command, skipping the GPU entirely. Leave it off to use the GPU when one is available. Turn it on when the machine has no usable GPU, or when GPU runs fail with a CUDA error such as *no kernel image is available for execution on the device*. CPU inference is considerably slower, so prefer **nnUNet Type** **2d** and a small **Nr. of Steps** when testing.

### Output
- **Output Segmentation**: Segmentation node where the result will be stored
- **Output Directory**: Directory where SeqSeg saves all results including segmentation (.mha), surface mesh (.vtp), and intermediate files

### Visualization Features
After running SeqSeg, use the visualization buttons to load results:
- **Load Segmentation**: Loads the most recent .mha segmentation file as a segmentation overlay
- **Load Surface Mesh**: Loads the most recent .vtp surface mesh as a 3D model
- **Browse All Output Files**: Opens a dialog to view and selectively load output files (segmentations, meshes, etc.)

Output files are organized in the structure:
```
output_directory/
├── images/           # Input volume (converted to .nii.gz)
├── seeds.json       # Seed points in SeqSeg format
└── output/          # SeqSeg results
    ├── *_seg_containing_seeds_*_steps.mha  # Segmentation
    └── *_surface_mesh_*_steps.vtp          # Surface mesh
```

## Technical Details

The extension follows the [SeqSeg CLI interface](https://github.com/numisveinsson/SeqSeg/tree/main):

1. **Creates data directory structure**: Sets up a temporary workspace with an `images/` subdirectory
2. **Saves input volume**: Exports the input volume as a NIfTI file in the `images/` subdirectory
3. **Creates seeds.json**: Generates a `seeds.json` file in the data directory with the format:
   ```json
   [
     {
       "name": "case_001",
       "seeds": [
         [[x1, y1, z1], [x2, y2, z2], radius_estimate]
       ]
     }
   ]
   ```
   Where `name` matches the image filename (without extension), the first point is the start seed, the second point is the direction seed, and the third value is the radius estimate in millimeters.
4. **Executes SeqSeg**: Runs the SeqSeg command-line tool with parameters:
   - `-data_dir`: Path to the temporary data directory
   - `-nnunet_results_path`: Path to trained nnUNet models
   - `-nnunet_type`: Model architecture (3d_fullres or 2d)
   - `-train_dataset`: Dataset name used for training
   - `-img_ext`: Image file extension (.nii.gz)
   - `-config_name`: Configuration name (default: global)
   - `-outdir`: Output directory for results
   - `-unit`: Image coordinate units (mm or cm)
   - `-max_n_steps`: Maximum tracking steps (same quantity as **Nr. of Steps** in the module UI)
   - `-max_n_branches`: Maximum number of branches
   - `-max_n_steps_per_branch`: Maximum steps per branch
   - `-cpu`: Added when **Force CPU (no GPU)** is checked, or automatically when the GPU cannot run PyTorch kernels, to keep nnU-Net inference on the CPU (requires `seqseg` 2.1.1 or newer)
5. **Loads results**: Imports the resulting segmentation file back into 3D Slicer
6. **Converts to segmentation**: Converts the volume to a Slicer segmentation node for visualization

## Troubleshooting

- **PyTorch / Slicer NNUNet missing**: Install both from **Extension Manager**, restart Slicer if prompted, then open **SeqSeg Vessel Segmentation** again.
- **Dependency installation cancelled or failed**: Check the **Python Interactor** log for `pip` output. Confirm network access. Use the **PyTorch Util** module to fix **`torch`** versions if the error mentions an incompatible PyTorch build; use the **nnUNet** module from **Slicer NNUNet** if **`nnunetv2`** is missing or too old.
- **Restart requested after install**: Complete any dependency dialog; restart Slicer when the module asks so newly installed packages load cleanly.
- **"SeqSeg dependency installation failed"** or **`seqseg`** still missing**: Install **`seqseg>=2.1.1`** with **`PythonSlicer -m pip`** ([Manual / advanced installation](#manual--advanced-installation)) after **`torch`** and **`nnunetv2`** are working via the extensions.
- **"Seed point is not defined"**: Make sure both markups nodes contain at least one fiducial point.
- **"SeqSeg execution failed"**: If you have not downloaded weights yet, use **Download Aorta Weights (CT/MR)** or **Download Coronary CT Weights** in the module first; otherwise confirm **nnUNet Results Path** points at the folder that contains the unpacked `nnUNet_results` tree and that the **Train Dataset** dropdown matches the model you installed.
- **"No output segmentation file found"**: SeqSeg may have failed silently—check the Slicer console for detailed error messages.
- **"no kernel image is available for execution on the device"**: The installed **`torch`** build has no CUDA kernels for your GPU architecture, so nnU-Net inference cannot run there. nnU-Net reports this as a memory problem, but it is not. The module detects it, reruns SeqSeg with **`-cpu`**, and fails the run if tracking still segments nothing (for example a `_0_steps` output file). Check **Force CPU (no GPU)** under **nnUNet Configuration** to skip the failing GPU attempt altogether, or install a PyTorch build matching your GPU in the **PyTorch Util** module to use the GPU.
- **Segmentation appears in wrong location**: Check that your volume and seed points are in the same coordinate system.
- **Permission errors during install**: Try running Slicer with appropriate permissions for writing under its Python environment, or install packages manually with **`PythonSlicer -m pip`** after diagnosing the error.

## Prerequisites for SeqSeg

**Using this extension:** install **PyTorch** and **Slicer NNUNet** from the Extension Manager ([Prerequisites](#prerequisites)) so **`torch` 2.2.2** and **`nnunetv2` 2.5.1** are available in Slicer’s Python. Then install **pretrained** nnUNet weights with the **download buttons** in the module—no manual command-line nnUNet training for typical use.

The underlying SeqSeg / nnUNet stack ultimately needs:
1. **PyTorch** and **`nnunetv2`** managed through those Slicer extensions (not ad hoc system-wide pip alone).
2. The **`seqseg`** Python package (**`seqseg>=2.1.1`**, upgraded to the latest PyPI release when a newer one is available) plus its remaining dependencies, installed via **`slicer.util.pip_install`** with selective skipping of Slicer-managed packages (see [First-time Python setup](#first-time-python-setup)).
3. **Trained model files** on disk—the extension supplies these via **Download Aorta…** / **Download Coronary CT…** unless you point **nnUNet Results Path** at your own trained outputs (e.g., `Dataset005_SEQAORTANDFEMOMR`, `Dataset006_SEQAORTANDFEMOCT`, or `Dataset010_SEQCOROASOCACT` for coronary CT).
4. For **custom** nnUNet layouts, the environment variables your setup expects; using the module’s **download buttons** covers the common case without manual variable tuning.
5. Compatible image formats (NIfTI recommended).

For more information about SeqSeg, see the [official SeqSeg repository](https://github.com/numisveinsson/SeqSeg).

## Development

This module began as the standalone [SlicerSeqSeg](https://github.com/numisveinsson/SlicerSeqSeg) extension and is now developed here, as part of the SimVascular extension. For issues and contributions, use this repository.

## License

Unlike the rest of this repository, `SeqSeg/` is licensed under the **Apache License, Version 2.0**. See [SeqSeg/LICENSE.txt](../SeqSeg/LICENSE.txt).

The [SeqSeg](https://github.com/numisveinsson/SeqSeg) Python package is a separate project; see that repository for its license and citation information.
