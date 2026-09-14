"""Name the faces of a volume mesh and write the folder an svMultiPhysics case reads.

A mesher hands back one grid holding the volume elements and the boundary cells they stand
on, every cell carrying a face id. svMultiPhysics reads a folder: the volume elements alone
under `GlobalNodeID`/`GlobalElementID`, the exterior under `ModelFaceID`, and one file per
named face, which is what its boundary conditions bind to. No mesher writes any of that.

Between the two sits naming. A face id is a number; a boundary condition is per vessel.
Somebody has to say which cap is the azygous vein -- and the names they give become the
`mesh-surfaces/` file names, and through them the `Add_face` and `Add_BC` names in
`solver.xml`, so they are what every result comes back labelled with. That is what this
panel is for: the faces measured and listed, the selected one shown in the 3D view, a name
against it.

## Where the names come from

Usually not from typing. A mesh clipped in Clip Vessel arrives with the names already on it:
the vessel names are the labels on the clip points, Clip Vessel records which control point
named each face, and CFD Mesh Generator carries that record onto the volume mesh. This panel
follows it, so a Fontan case with twenty-six faces opens named rather than empty. The chain
is the node reference and attributes named in the constants below; a mesh from anywhere else
carries none of them, and then every name is typed here, as it always was.

What is typed is an *override*, not the name: the panel keeps the overrides and works the
inherited names out again on every load, so there is no second copy of a label to go stale.
Renaming a clip point renames the face; typing over it makes it stick; clearing the cell
takes the inherited name back. Inherited names are drawn dimmed and italic, because the risk
of a name you did not choose is that it looks like one you did -- `cap_Outlet_1` off Clip
Vessel's positional default reads exactly like a decision, and noticing that is the point of
having a panel at all.

## Where the geometry is

All of it is in `svmeshcomplete`, the package beside this file, which imports nothing from
Slicer and is pip-installable on its own. This file is an MRML adapter: it reads the
selected node's grid, calls the package, and puts what it says into a table.

The split is what keeps a case prepared here and a case prepared from a terminal identical
-- the workflow scripts that package Fontan cases call the same functions -- and it is why
the checks below are the package's rather than this panel's. All three of them catch inputs
that otherwise reach the solver and are misread in silence: a volume mesh of more than one
element type, a face table that does not describe the mesh, and a boundary cell standing
against no volume element.

## Flatness

The one measurement worth explaining. It is the largest distance of any point of a face
from that face's own best-fit plane, so a cap cut normal to its vessel reads 0 and one that
was not cut cleanly does not. The flow crossing a cap that is not planar is not what its
boundary condition says, and nothing downstream will notice.
"""

import json
import logging
import os

import numpy as np
import qt
import slicer
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

from svmeshcomplete import cells, face_table, faces, mesh_complete

# Columns of the table, and which of them the operator edits.
COLUMNS = ("Face", "Name", "Cells", "Area", "Diameter", "Flatness")
NAME_COLUMN = COLUMNS.index("Name")

# Where the panel's state is kept so that it is saved with the scene, which is what makes
# the naming survive closing Slicer. The names are the expensive part -- twenty-odd caps
# matched to vessels by eye -- and a scene that came back without them would be a scene
# whose mesh had to be named again.
NAMES_PARAMETER = "FaceNames"
FACE_ID_ARRAY_PARAMETER = "FaceIdArrayNames"
OUTPUT_DIRECTORY_PARAMETER = "OutputDirectory"
INPUT_MESH_REFERENCE = "InputMesh"

# Where Clip Vessel recorded which of its clip points named each face, carried onto the volume
# mesh by CFD Mesh Generator. The node reference names the markups node the names are control
# point labels of; the attribute says which control point each face id came from, keyed by
# control point ID so that deleting a clip point cannot move a name onto its neighbour's face.
#
# Strings rather than an import: Clip Vessel is in another extension, which this one does not
# depend on and which may not be installed. A mesh from anywhere else simply carries none of
# these, and then every name is typed here, as it was before.
CLIP_POINTS_NODE_REFERENCE = "ClipPoints"
FACE_ID_TO_CLIP_POINT_ID_ATTRIBUTE = "ClipVessel.FaceIdToClipPointID"
WALL_FACE_ID_ATTRIBUTE = "ClipVessel.WallFaceID"

# How an inherited name is drawn: the same text a typed name would be, in italic and dimmed.
# Worth the trouble because the risk of inheriting names is that they look like chosen ones --
# `cap_Outlet_1` off a positional default reads exactly like a name somebody decided on, and
# noticing that is what this panel is for.
INHERITED_NAME_COLOR = qt.QColor(128, 128, 128)

# The highlight's own node, kept out of the way of anything the operator has, and the
# colour it is drawn in: yellow against the grey the mesh sits at.
HIGHLIGHT_NODE_NAME = "Mesh Prep face highlight"
HIGHLIGHT_COLOR = (1.0, 1.0, 0.0)

# What the mesh is left as when face colouring is turned off: the same neutral grey Clip
# Vessel gives its output, rather than whatever colour the node happened to be created
# with, and see-through enough to find a cap behind it without reaching for a button.
SOLID_COLOR = (0.75, 0.75, 0.75)
SOLID_OPACITY = 0.8

# What the transparency button drops to, and comes back from.
TRANSPARENT_OPACITY = 0.4

# Priority the 3D view's mouse events are observed at, which has to be above zero. The
# view's own camera widget observes them at zero and aborts the left button press once it
# has taken it to start a rotate, so an observer added at zero -- after that one, which is
# where equal priority puts it -- is never told a button went down at all. Above zero is
# ahead of it. Nothing here aborts anything, so the camera still rotates and zooms as it
# did; this only buys the right to watch.
VIEW_EVENT_PRIORITY = 1.0


class SimVascularMeshPrep(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SimVascular Mesh Prep")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "SimVascular")]
        self.parent.dependencies = []
        self.parent.contributors = ["Aaron Brown (Stanford University)"]
        self.parent.helpText = _(
            "Name the faces of a volume mesh and write the mesh-complete folder an "
            "svMultiPhysics case reads. The mesh is one carrying a face id per cell, "
            "typically out of CFD Mesh Generator; the output is the volume elements, the "
            "exterior surface and one file per named face, under the GlobalNodeID the "
            "solver binds its boundary conditions through."
        )
        self.parent.acknowledgementText = _(
            "Developed in the Cardiovascular Biomechanics Computation Lab at Stanford "
            "University. The mesh packaging is the svmeshcomplete package beside this "
            "module, which runs outside Slicer as well."
        )


class SimVascularMeshPrepWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._measured = []
        # What was typed, which is not the same as what the faces are called. The names shown
        # are `override or inherited`: see effectiveNames(). Only the overrides are saved --
        # the inherited ones are worked out again on every load, so that they cannot go stale
        # against the clip points they came from.
        self._overrides = {}
        self._inherited = {}
        self._inheritedNotes = ()
        self._wallFaceId = None
        self._clipPointsNode = None
        self._updating = False
        self._lookup = None
        self._hovered = None
        self._pressedAt = None
        self._cameraDrag = False
        self._viewObservers = []
        self._pickingFailed = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SimVascularMeshPrep.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.logic = SimVascularMeshPrepLogic()

        table = self.ui.facesTable
        table.setColumnCount(len(COLUMNS))
        table.setHorizontalHeaderLabels([_(name) for name in COLUMNS])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(False)
        table.horizontalHeader().setSectionResizeMode(qt.QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(NAME_COLUMN, qt.QHeaderView.Stretch)

        # The three buttons beside the selector, none of them checkable: a mark would be
        # saying what the display node holds, and nothing tells this when that changes
        # elsewhere, so it would sooner or later contradict the scene. Each reads the state
        # at the moment it is pressed and turns it around.
        rowHeight = self.ui.inputMeshSelector.sizeHint.height()
        buttons = (
            (self.ui.meshEdgesButton, self.resourcePath("Icons/ToggleEdges.svg"),
             self.onToggleEdges),
            (self.ui.meshColorsButton, self.resourcePath("Icons/ToggleFaceColors.svg"),
             self.onToggleFaceColors),
            (self.ui.meshTransparencyButton, self.resourcePath("Icons/ToggleTransparency.svg"),
             self.onToggleTransparency),
            (self.ui.meshVisibilityButton, ":/Icons/Medium/SlicerVisibleInvisible.png",
             self.onToggleVisibility),
        )
        for button, icon, handler in buttons:
            button.setIcon(qt.QIcon(icon))
            button.setAutoRaise(True)
            button.setFixedHeight(rowHeight)
            button.setIconSize(qt.QSize(rowHeight - 8, rowHeight - 8))
            button.connect("clicked()", handler)

        self.ui.inputMeshSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onMeshChanged)
        self.ui.faceIdArrayLineEdit.connect("editingFinished()", self.onMeshChanged)
        self.ui.facesTable.connect("itemSelectionChanged()", self.onSelectionChanged)
        self.ui.facesTable.connect("cellChanged(int,int)", self.onNameEdited)
        self.ui.exportButton.connect("clicked(bool)", self.onExport)
        self.ui.outputDirectoryPathLineEdit.connect(
            "currentPathChanged(QString)", self.onOutputDirectoryChanged
        )

        # A scene being loaded replaces the parameter node, so the panel has to read itself
        # back out of the new one rather than trust what it is showing.
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneEndImport
        )
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose
        )

        self.restoreFromParameterNode()
        self.updateButtons()

    def enter(self):
        """Coming back to the panel, possibly after a scene was loaded elsewhere."""
        self.restoreFromParameterNode()
        self.observeThreeDViews()

    def exit(self):
        """Leaving it. The views are let go of: moving the mouse over a 3D view while some
        other module is open has nothing to do with this panel's table.

        The highlight goes back to the selected row on the way out. Whatever the cursor
        happened to be over as the panel closed is not something anything still on screen
        says, and leaving it there is a view disagreeing with a table nobody can see.
        """
        self.stopObservingThreeDViews()
        self._hovered = None
        self._pressedAt = None
        self._cameraDrag = False
        self.onSelectionChanged()

    def onSceneEndImport(self, caller=None, event=None):
        self.restoreFromParameterNode()

    def onSceneEndClose(self, caller=None, event=None):
        self._measured = []
        self._overrides = {}
        self.forgetInheritedNames()
        self.populateTable()

    # -- state that is saved with the scene ---------------------------------
    def parameterNode(self):
        return self.logic.getParameterNode()

    def restoreFromParameterNode(self):
        """Put the panel back the way the scene left it."""
        node = self.parameterNode()
        if node is None:
            return
        self._updating = True
        try:
            # What a saved scene holds is read back as overrides, which is what it is: before
            # there was anything to inherit every name was typed, so an old scene is a scene
            # of nothing but overrides and loads correctly under the new rule.
            self._overrides = self.logic.readNamesParameter(node.GetParameter(NAMES_PARAMETER))
            arrayNames = node.GetParameter(FACE_ID_ARRAY_PARAMETER)
            if arrayNames:
                self.ui.faceIdArrayLineEdit.text = arrayNames
            directory = node.GetParameter(OUTPUT_DIRECTORY_PARAMETER)
            if directory:
                self.ui.outputDirectoryPathLineEdit.currentPath = directory
            mesh = node.GetNodeReference(INPUT_MESH_REFERENCE)
            if mesh is not None:
                self.ui.inputMeshSelector.setCurrentNode(mesh)
        finally:
            self._updating = False
        self.onMeshChanged()

    def saveToParameterNode(self):
        node = self.parameterNode()
        if node is None:
            return
        was_modifying = node.StartModify()
        node.SetParameter(NAMES_PARAMETER, self.logic.writeNamesParameter(self._overrides))
        node.SetParameter(FACE_ID_ARRAY_PARAMETER, self.ui.faceIdArrayLineEdit.text)
        node.SetParameter(
            OUTPUT_DIRECTORY_PARAMETER, self.ui.outputDirectoryPathLineEdit.currentPath or ""
        )
        node.SetNodeReferenceID(
            INPUT_MESH_REFERENCE,
            self.ui.inputMeshSelector.currentNodeID or None,
        )
        node.EndModify(was_modifying)

    def onOutputDirectoryChanged(self, *_args):
        if not self._updating:
            self.saveToParameterNode()
            self.updateButtons()

    def cleanup(self):
        self.clearHighlight()
        self.stopObservingThreeDViews()
        self.removeObservers()

    # -- picking a face in the 3D view --------------------------------------
    def observeThreeDViews(self):
        """Watch the 3D views for the cursor, so a face can be picked by looking at it.

        Observers on the interactor rather than on the crosshair node: the crosshair
        reports where the cursor is and this needs to know what is under it, which is a
        pick. Nothing aborts the event, so the camera still rotates and zooms as it did.
        """
        self.stopObservingThreeDViews()
        self._pressedAt = None
        self._cameraDrag = False
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return
        for index in range(layoutManager.threeDViewCount):
            view = layoutManager.threeDWidget(index).threeDView()
            interactor = view.interactor()
            for event, handler in (
                (vtk.vtkCommand.MouseMoveEvent, self.onViewMouseMove),
                (vtk.vtkCommand.LeftButtonPressEvent, self.onViewButtonPress),
                (vtk.vtkCommand.LeftButtonReleaseEvent, self.onViewButtonRelease),
                (vtk.vtkCommand.MiddleButtonPressEvent, self.onViewCameraPress),
                (vtk.vtkCommand.MiddleButtonReleaseEvent, self.onViewCameraRelease),
                (vtk.vtkCommand.RightButtonPressEvent, self.onViewCameraPress),
                (vtk.vtkCommand.RightButtonReleaseEvent, self.onViewCameraRelease),
            ):
                tag = interactor.AddObserver(
                    event,
                    lambda caller, e, v=view, h=handler: self.callHandler(h, v),
                    VIEW_EVENT_PRIORITY,
                )
                self._viewObservers.append((interactor, tag))

    def callHandler(self, handler, view):
        """Run a view handler, reporting the first exception rather than losing it.

        VTK discards what an observer callback raises, so a mistake in one of these is
        invisible from the outside: the cursor moves, nothing happens, nothing is said.
        Reported once because a mouse move fires often enough to fill a log in seconds.
        """
        try:
            handler(view)
        except Exception:
            if not self._pickingFailed:
                self._pickingFailed = True
                logging.exception("Mesh Prep could not pick a face in the 3D view")
                self.setStatus(
                    _("Picking faces in the 3D view failed; see the error log. The table "
                      "still works."),
                    warning=True,
                )

    def stopObservingThreeDViews(self):
        for interactor, tag in self._viewObservers:
            interactor.RemoveObserver(tag)
        self._viewObservers = []

    def faceUnderCursor(self, view):
        """The face id under the cursor in this view, or None.

        The mesh's own boundary is intersected with the ray the cursor looks down, rather
        than the 3D view being asked what it drew there. Two reasons, both learned from
        vtkMRMLModelDisplayableManager::Pick. It is a software ray cast over every cell
        drawn, with no locator behind it, so on a mesh this size it costs a third of a
        second -- and this runs once per mouse move, which a camera cannot be dragged
        through. And it answers in a position, which then has to be matched back to a face
        by proximity; the boundary's own locator answers in the cell it hit, which carries
        the face id already.

        The cost of asking the mesh rather than the view is that another model in front of
        the mesh no longer hides it. Nothing is normally in front of it but this module's
        own highlight, which is the same surface.
        """
        if self._lookup is None:
            return None
        renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
        if renderer is None:
            return None
        x, y = view.interactor().GetEventPosition()
        return self.logic.faceAlongRay(
            self._lookup,
            self.displayToWorld(renderer, x, y, 0.0),
            self.displayToWorld(renderer, x, y, 1.0),
        )

    @staticmethod
    def displayToWorld(renderer, x, y, z):
        """A point on the ray under (x, y): z of 0 is the near plane, 1 the far one.

        The interactor's event position is measured from the bottom left of the view, which
        is the renderer's own convention, so it goes in as it comes out. Worth saying
        because the displayable manager's Pick is the exception -- it takes a y measured
        from the top -- and handing that one an event position mirrors every pick about the
        middle of the view, which is a wrong answer rather than no answer.
        """
        renderer.SetDisplayPoint(float(x), float(y), float(z))
        renderer.DisplayToWorld()
        point = list(renderer.GetWorldPoint())
        if not point[3]:
            return point[:3]
        return [coordinate / point[3] for coordinate in point[:3]]

    def onViewMouseMove(self, view):
        """Show whichever face the cursor is over, without disturbing the selection.

        Nothing happens while a mouse button is held. A button down in a 3D view means the
        camera is being moved -- turned, panned or zoomed -- and then it is the model that
        travels past a cursor holding still, rather than somebody pointing at one face
        after another. Following it would flicker the highlight through every face the
        anatomy happens to sweep through on the way round.
        """
        if self._pressedAt is not None or self._cameraDrag:
            return
        faceId = self.faceUnderCursor(view)
        if faceId == self._hovered:
            return
        self._hovered = faceId
        if faceId is None:
            # Back to the row that is selected, which is what the highlight otherwise says.
            self.onSelectionChanged()
            return
        self.showFace(faceId)

    def showFace(self, faceId):
        """Draw one face of the selected mesh over it, edged to match the mesh."""
        node = self.ui.inputMeshSelector.currentNode()
        if node is None or node.GetMesh() is None:
            return
        display = node.GetDisplayNode()
        self.logic.highlight(
            node.GetMesh(),
            faceId,
            self.ui.faceIdArrayLineEdit.text,
            showEdges=bool(display and display.GetEdgeVisibility()),
            boundary=self._lookup[0] if self._lookup else None,
        )

    def onViewButtonPress(self, view):
        self._pressedAt = view.interactor().GetEventPosition()

    def onViewButtonRelease(self, view):
        """Select the face that was clicked -- but not at the end of a camera drag."""
        pressedAt, self._pressedAt = self._pressedAt, None
        released = view.interactor().GetEventPosition()
        if pressedAt is None:
            return
        if abs(released[0] - pressedAt[0]) > 2 or abs(released[1] - pressedAt[1]) > 2:
            # The camera was turned rather than a face chosen. Hovering was held off for
            # the whole drag, so the highlight is still on whatever the cursor was over
            # when it started; it is caught up here rather than left wrong until the mouse
            # next moves, which may be a while if the operator stops to look.
            self.onViewMouseMove(view)
            return
        faceId = self.faceUnderCursor(view)
        if faceId is not None:
            self.selectFace(faceId)

    def onViewCameraPress(self, view):
        """A middle or right button going down: the start of a pan or a zoom.

        Only that it happened is recorded, not where. Neither gesture can be meant as a
        click on a face -- the left button is the one that picks -- so there is nothing to
        tell a drag from a click for, and any release ends it.
        """
        self._cameraDrag = True

    def onViewCameraRelease(self, view):
        self._cameraDrag = False
        self.onViewMouseMove(view)

    def selectFace(self, faceId):
        """Select a face's row and open its name for typing, scrolling to it if need be.

        Clicking a cap in the 3D view is somebody saying "this one is the azygous vein",
        so it leaves the cursor in the name cell ready for them to say it. The row is
        reached through the name cell rather than selected separately: the table selects
        whole rows, so setting the current cell selects the row as well, and doing it once
        means the highlight is rebuilt once rather than twice.
        """
        table = self.ui.facesTable
        for row, face in enumerate(self._measured):
            if face.face_id == faceId:
                item = table.item(row, NAME_COLUMN)
                table.setCurrentCell(row, NAME_COLUMN)
                if item is not None:
                    table.scrollToItem(item)
                    # Typing a name replaces whatever is there, which is what renaming a
                    # cap_3 to the vessel it is wants. Qt selects the text for us.
                    #
                    # Not opened twice. Clicking the same cap again -- looking at it once
                    # more before naming it, say -- leaves the editor already up, and Qt
                    # answers a second request to open it with "editing failed" in the
                    # log. Despite the name, isPersistentEditorOpen reports any editor on
                    # the cell, which is the question being asked.
                    if not table.isPersistentEditorOpen(item):
                        table.editItem(item)
                return True
        return False

    # -- which name a face has ---------------------------------------------
    def effectiveNames(self):
        """What the faces are called: what was typed, and the clip's own name where nothing was.

        The one rule the rest of the panel reads names through. An override wins, an empty
        override is not an override - clearing a cell puts the inherited name back rather than
        blanking a face that has a perfectly good name upstream - and a face with neither is
        left empty, which is what the table shows as still to name.
        """
        return {face.face_id: (self._overrides.get(face.face_id)
                               or self._inherited.get(face.face_id, ""))
                for face in self._measured}

    def isInherited(self, faceId):
        """Whether the name shown for a face is the clip's rather than one somebody typed."""
        return not self._overrides.get(faceId) and bool(self._inherited.get(faceId))

    def forgetInheritedNames(self):
        """Drop what was inherited, and stop watching the clip points it came from."""
        if self._clipPointsNode is not None:
            self.removeObserver(self._clipPointsNode,
                                slicer.vtkMRMLMarkupsNode.PointModifiedEvent,
                                self.onClipPointsChanged)
            self.removeObserver(self._clipPointsNode,
                                slicer.vtkMRMLMarkupsNode.PointRemovedEvent,
                                self.onClipPointsChanged)
            self._clipPointsNode = None
        self._inherited = {}
        self._inheritedNotes = ()
        self._wallFaceId = None

    def readInheritedNames(self, node):
        """Work out what this mesh's faces are already called, and watch for that changing.

        Observed rather than read once: renaming a clip point in Clip Vessel has to show up
        here, and a deleted clip point has to take its face's name away. Those are the two
        events a label can move under - a label change fires PointModified and nothing else, a
        deletion fires PointRemoved - and without them the panel would be showing names that
        the scene no longer agrees with.
        """
        self.forgetInheritedNames()
        if node is None:
            return
        derived = self.logic.inheritedNames(node)
        self._inherited = dict(derived.names)
        self._inheritedNotes = derived.notes
        self._wallFaceId = self.logic.wallFaceId(node)
        self._clipPointsNode = self.logic.clipPointsNode(node)
        if self._clipPointsNode is not None:
            self.addObserver(self._clipPointsNode,
                             slicer.vtkMRMLMarkupsNode.PointModifiedEvent,
                             self.onClipPointsChanged)
            self.addObserver(self._clipPointsNode,
                             slicer.vtkMRMLMarkupsNode.PointRemovedEvent,
                             self.onClipPointsChanged)

    def onClipPointsChanged(self, caller=None, event=None):
        """A clip point was renamed or removed upstream, so the inherited names have moved.

        Only the names are recomputed; the mesh has not changed, so nothing is re-measured.
        """
        node = self.ui.inputMeshSelector.currentNode()
        if node is None or not self._measured:
            return
        derived = self.logic.inheritedNames(node)
        self._inherited = dict(derived.names)
        self._inheritedNotes = derived.notes
        self.populateTable()
        self.updateStatus()

    # -- reading the mesh --------------------------------------------------
    def onMeshChanged(self, _node=None):
        """Measure the selected mesh's faces, keeping any names already typed."""
        self._measured = []
        self._lookup = None
        self._hovered = None
        self.forgetInheritedNames()
        node = self.ui.inputMeshSelector.currentNode()
        mesh = node.GetMesh() if node else None
        if mesh is None:
            self.setStatus(
                _("Select a volume mesh. A model node holding a surface is not one: this "
                  "wants the grid a mesher filled it with.")
                if node else ""
            )
            self.populateTable()
            return

        try:
            self._measured = self.logic.measure(mesh, self.ui.faceIdArrayLineEdit.text)
        except ValueError as error:
            self.setStatus(str(error), warning=True)
            self.populateTable()
            return

        arrayName = self.logic.faceIdArrayName(mesh, self.ui.faceIdArrayLineEdit.text)
        self._lookup = self.logic.faceLookup(mesh, arrayName) if arrayName else None
        # Before the table, which draws inherited names differently from typed ones.
        self.readInheritedNames(node)
        self.populateTable()
        self.saveToParameterNode()
        self.updateStatus()
        if not self.ui.outputDirectoryPathLineEdit.currentPath:
            self.ui.outputDirectoryPathLineEdit.currentPath = self.logic.suggestedOutputDirectory()

    def wallFaceId(self):
        """Which face is the wall: what the clip recorded, or the face with the most cells.

        The guess is right almost always -- a wall has far more cells than any cap -- but it is
        a guess, and it is wrong exactly where it matters: a wall split into several faces, or
        a short vessel with one very large cap. Where the mesh came through Clip Vessel the
        answer was written down, so it is not guessed at.
        """
        if self._wallFaceId is not None:
            return self._wallFaceId
        wall = max(self._measured, key=lambda face: face.cell_count, default=None)
        return wall.face_id if wall is not None else None

    def populateTable(self):
        table = self.ui.facesTable
        wallFaceId = self.wallFaceId()
        names = self.effectiveNames()
        labels = self.logic.inheritedLabels(self.ui.inputMeshSelector.currentNode())
        clipPointsNode = self._clipPointsNode
        self._updating = True
        try:
            table.setRowCount(len(self._measured))
            for row, face in enumerate(self._measured):
                values = (
                    str(face.face_id),
                    names.get(face.face_id, ""),
                    f"{face.cell_count:,}",
                    f"{face.area:.2f}",
                    f"{face.effective_diameter:.2f}",
                    f"{face.flatness:.4f}",
                )
                for column, value in enumerate(values):
                    # Reused where there is one already, and only handed to the table when
                    # it is new: setItem on an item the table already owns is refused, with
                    # a Qt warning per cell that buries everything else in the log.
                    item = table.item(row, column)
                    if item is None:
                        item = qt.QTableWidgetItem()
                        table.setItem(row, column, item)
                    item.setText(value)
                    if column == NAME_COLUMN:
                        item.setFlags(item.flags() | qt.Qt.ItemIsEditable)
                        self.styleNameCell(item, face.face_id, labels, clipPointsNode)
                    else:
                        item.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
                    if column == COLUMNS.index("Flatness") and self.logic.looksUnplanar(
                        face, face.face_id == wallFaceId
                    ):
                        item.setToolTip(
                            _("Not planar, so this was not cut cleanly. Whatever this face "
                              "is named, the flow crossing it is not what a boundary "
                              "condition on it would say.")
                        )
                        item.setForeground(qt.QBrush(qt.QColor(200, 120, 0)))
        finally:
            self._updating = False
        self.updateButtons()

    def styleNameCell(self, item, faceId, labels, clipPointsNode):
        """Draw an inherited name as what it is: not typed here, and traceable to where it was.

        Dimmed and italic, with a tooltip naming the clip point it came from. The styling is
        the point of the whole panel applied to itself -- an operator has to be able to see at
        a glance which of twenty-six names nobody has actually checked, and `cap_Outlet_1` off
        a positional default is indistinguishable from a real name otherwise.
        """
        font = item.font()
        inherited = self.isInherited(faceId)
        font.setItalic(inherited)
        item.setFont(font)
        item.setForeground(qt.QBrush(INHERITED_NAME_COLOR) if inherited else qt.QBrush())
        if not inherited:
            item.setToolTip(_("Type a name for this face. It becomes the file name under "
                              "mesh-surfaces/ and the solver's boundary condition name."))
            return
        label = labels.get(faceId) or ""
        source = clipPointsNode.GetName() if clipPointsNode is not None else ""
        item.setToolTip(
            _("From the clip point “{label}” in “{node}”, not typed here. "
              "Type over it to override; clear the cell to take this name back.").format(
                label=label, node=source)
        )

    # -- naming ------------------------------------------------------------
    def onNameEdited(self, row, column):
        """Take what was typed as an override, or drop the override if the cell was cleared.

        An emptied cell is not an empty name: it means "whatever the clip calls this", which is
        the inherited name if there is one and still-to-name if there is not. Dropping the key
        rather than storing "" keeps the saved scene to the names somebody actually chose.
        """
        if self._updating or column != NAME_COLUMN:
            return
        faceId = self._measured[row].face_id
        typed = (self.ui.facesTable.item(row, column).text() or "").strip()
        if typed and typed != self._inherited.get(faceId):
            self._overrides[faceId] = typed
        else:
            # Typing the inherited name back is not an override either: it is what the face is
            # already called, and storing it would freeze it against a later rename upstream.
            self._overrides.pop(faceId, None)
        self.saveToParameterNode()
        # The cell has to be redrawn: it may have gone from typed to inherited or back, and the
        # styling and the tooltip say which.
        self.populateTable()
        self.updateStatus()
        self.updateButtons()

    def selectedFace(self):
        rows = {index.row() for index in self.ui.facesTable.selectedIndexes()}
        if len(rows) != 1:
            return None
        row = rows.pop()
        return self._measured[row] if 0 <= row < len(self._measured) else None

    def onSelectionChanged(self, *_args):
        """Show whichever face is selected. Always: a row on its own says very little."""
        face = self.selectedFace()
        node = self.ui.inputMeshSelector.currentNode()
        if face is None or node is None or node.GetMesh() is None:
            self.clearHighlight()
            return
        self.showFace(face.face_id)

    def clearHighlight(self):
        self.logic.clearHighlight()

    # -- looking at the mesh -----------------------------------------------
    def meshDisplayNode(self):
        node = self.ui.inputMeshSelector.currentNode()
        if node is None:
            return None
        if node.GetDisplayNode() is None:
            node.CreateDefaultDisplayNodes()
        return node.GetDisplayNode()

    def onToggleVisibility(self):
        display = self.meshDisplayNode()
        if display is not None:
            display.SetVisibility(not display.GetVisibility())

    def onToggleEdges(self):
        """On the mesh and on the highlighted face together: they are one picture."""
        display = self.meshDisplayNode()
        if display is None:
            return
        visible = not display.GetEdgeVisibility()
        display.SetEdgeVisibility(visible)
        self.logic.setHighlightEdgeVisibility(visible)

    def onToggleFaceColors(self):
        display = self.meshDisplayNode()
        node = self.ui.inputMeshSelector.currentNode()
        if display is None or node is None or node.GetMesh() is None:
            return
        if display.GetScalarVisibility():
            self.logic.showSolidColor(display)
            return
        arrayName = self.logic.faceIdArrayName(
            node.GetMesh(), self.ui.faceIdArrayLineEdit.text
        )
        if arrayName is None:
            self.setStatus(
                _("Nothing to colour by: this mesh carries no face ids."), warning=True
            )
            return
        self.logic.colourByFaceIds(display, arrayName)

    def onToggleTransparency(self):
        """Between see-through and the mesh's own opacity, not between that and opaque.

        The grey the mesh sits at is already a little transparent, so testing against 1.0
        would make this button turn transparency *off* the first time it is pressed.
        """
        display = self.meshDisplayNode()
        if display is None:
            return
        transparent = display.GetOpacity() <= TRANSPARENT_OPACITY
        display.SetOpacity(SOLID_OPACITY if transparent else TRANSPARENT_OPACITY)

    # -- export ------------------------------------------------------------
    def onExport(self):
        node = self.ui.inputMeshSelector.currentNode()
        directory = self.ui.outputDirectoryPathLineEdit.currentPath
        if node is None or node.GetMesh() is None or not directory:
            return
        with slicer.util.tryWithErrorDisplay(_("Failed to export the mesh."), waitCursor=True):
            result = self.logic.export(
                node.GetMesh(),
                self._measured,
                self.effectiveNames(),
                directory,
                self.ui.faceIdArrayLineEdit.text,
            )
        self.setStatus(
            _("Wrote {directory}: {elements:,} {kind} elements over {nodes:,} nodes, "
              "{faces} faces.").format(
                directory=directory,
                elements=result.number_of_elements,
                kind=result.element_type,
                nodes=result.number_of_nodes,
                faces=len(result.face_surfaces),
            )
        )

    # -- panel state -------------------------------------------------------
    def updateButtons(self):
        names = self.effectiveNames()
        named = self._measured and all(
            names.get(face.face_id) for face in self._measured
        )
        self.ui.exportButton.enabled = bool(
            named and self.ui.outputDirectoryPathLineEdit.currentPath
        )
        self.ui.exportButton.toolTip = (
            _("Write the mesh-complete folder.")
            if named
            else _("Every face has to be named first: the names are the solver's boundary "
                   "condition names, so a face without one has nothing to bind to.")
        )

    def updateStatus(self):
        """What is left to do, which changes with every name typed.

        Not folded into updateButtons: that also runs when the output folder changes, and
        it would wipe out what an export had just reported.
        """
        self.setStatus(self.logic.statusText(
            self._measured, self._overrides, self._inherited, self._inheritedNotes))

    def setStatus(self, text, warning=False):
        self.ui.statusLabel.text = text
        self.ui.statusLabel.styleSheet = "QLabel { color: #d08000; }" if warning else ""


class SimVascularMeshPrepLogic(ScriptedLoadableModuleLogic):
    """The MRML-free work, which is `svmeshcomplete` plus the highlight node."""

    # Beyond this, in the mesh's own units, a face is not flat enough to be a clean cut.
    # A cap cut normal to its vessel comes out two orders of magnitude below it.
    FLATNESS_TOLERANCE = 0.5

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

    @staticmethod
    def measure(mesh, faceIdArrayNames):
        """Every labelled face of the mesh, measured."""
        names = [name.strip() for name in (faceIdArrayNames or "").split(",") if name.strip()]
        arrayName = None
        for candidate in names:
            if mesh.GetCellData().GetArray(candidate) is not None:
                arrayName = candidate
                break
        if arrayName is None and names:
            raise ValueError(
                _("The mesh carries none of {names} as face ids. Without them there is no "
                  "telling a cap from the wall; in CFD Mesh Generator the array is named "
                  "by its own 'Face ids array' field.").format(names=names)
            )
        return faces.measure_faces(mesh, arrayName)

    # -- names inherited from upstream -------------------------------------
    @staticmethod
    def clipPointsNode(meshNode):
        """The markups node whose control point labels name this mesh's faces, or None."""
        if meshNode is None:
            return None
        return meshNode.GetNodeReference(CLIP_POINTS_NODE_REFERENCE)

    @staticmethod
    def wallFaceId(meshNode):
        """The wall's face id as Clip Vessel recorded it, or None if nothing recorded it.

        Worth having for its own sake: the wall is the one face flatness says nothing about,
        and without this it has to be guessed at as the face with the most cells.
        """
        if meshNode is None:
            return None
        recorded = meshNode.GetAttribute(WALL_FACE_ID_ATTRIBUTE)
        try:
            return int(recorded)
        except (TypeError, ValueError):
            return None

    @classmethod
    def inheritedLabels(cls, meshNode):
        """`{faceId: clip point label}` for the faces Clip Vessel recorded a name source for.

        The labels are read from the markups node every time rather than from anything stored
        on the mesh, which is what makes a rename upstream arrive here: there is no second copy
        of a label to go stale.

        A face whose control point has been deleted is kept, with an empty label. Dropping it
        would be the same as never having recorded it, and the two are worth telling apart -
        that face had a name and no longer has one, which is a thing to tell the operator.
        """
        markups = cls.clipPointsNode(meshNode)
        recorded = meshNode.GetAttribute(FACE_ID_TO_CLIP_POINT_ID_ATTRIBUTE) if meshNode else None
        if markups is None or not recorded:
            return {}
        try:
            stored = json.loads(recorded)
        except ValueError:
            logging.warning("Mesh Prep could not read which clip point named each face of %s.",
                            meshNode.GetName())
            return {}
        labels = {}
        for faceId, controlPointId in stored.items():
            try:
                faceId = int(faceId)
            except (TypeError, ValueError):
                continue
            index = markups.GetNthControlPointIndexByID(str(controlPointId))
            labels[faceId] = markups.GetNthControlPointLabel(index) if index >= 0 else ""
        return labels

    @classmethod
    def inheritedNames(cls, meshNode):
        """The names this mesh's faces already have, from the clip they came out of.

        The whole of the chain from a clip point label to a face name, read at this end: walk
        the node reference, look each control point up by ID, and put the labels through the
        package's sanitizer, which is also what the headless workflow scripts use - a case
        packaged here and a case packaged from a terminal have to name their files the same.

        :return: a `face_table.DerivedNames`, whose `notes` are worth showing: a duplicated
          label upstream comes back numbered apart rather than as an error, and that is
          something to say out loud rather than to resolve silently.
        """
        return face_table.names_from_labels(cls.inheritedLabels(meshNode),
                                            wall_face_id=cls.wallFaceId(meshNode))

    @classmethod
    def looksUnplanar(cls, face, isWall):
        """Whether a face is too far from its own plane to be a clean cut.

        Never asked of the wall, which is not meant to be planar and would read large on
        any anatomy at all.
        """
        return not isWall and face.flatness > cls.FLATNESS_TOLERANCE

    @staticmethod
    def suggestedOutputDirectory():
        """`mesh` beside the scene file, once the scene has one.

        Nothing is suggested for a scene that has never been saved. `GetRootDirectory`
        answers even then -- with Slicer's default, the user's Documents folder -- and a
        mesh-complete folder written there is not where anyone wanted it, least of all
        with Export enabled and looking ready.
        """
        url = slicer.mrmlScene.GetURL()
        directory = os.path.dirname(url) if url else ""
        return os.path.join(directory, "mesh") if os.path.isdir(directory) else ""

    @staticmethod
    def writeNamesParameter(names):
        """The names as one string, for a parameter node to carry into a saved scene."""
        return json.dumps(
            {str(face_id): name for face_id, name in sorted(names.items())},
            separators=(",", ":"),
        )

    @staticmethod
    def readNamesParameter(text):
        """The names back out, and nothing at all rather than an exception from a bad one."""
        if not text:
            return {}
        try:
            stored = json.loads(text)
        except ValueError:
            logging.warning("Mesh Prep could not read the face names saved with this scene.")
            return {}
        names = {}
        for face_id, name in stored.items():
            try:
                names[int(face_id)] = str(name)
            except (TypeError, ValueError):
                continue
        return names

    def export(self, mesh, measured, names, directory, faceIdArrayNames):
        """Write the mesh-complete folder, through the package's own checks.

        The names are not written beside it. They are saved with the scene, which is
        where anything needing them reads them from -- including the workflow scripts that
        package cases from a terminal. A copy in the folder would be a second answer to
        the same question, and the one that goes stale is the one on disk.
        """
        table = face_table.FaceTable([
            face_table.Face(face.face_id, names[face.face_id]) for face in measured
        ])
        arrayName = self.faceIdArrayName(mesh, faceIdArrayNames)
        return mesh_complete.write_mesh_complete(
            mesh, table, directory, face_id_array_name=arrayName
        )

    @staticmethod
    def faceLookup(mesh, arrayName):
        """What is needed to answer "which face is under this ray": the boundary, its face
        ids, and a locator over it.

        The boundary is rebuilt here rather than the cell id a view pick returns being
        used, because that id indexes the polydata the display pipeline built to draw the
        node -- for an unstructured grid, not the grid's own cells -- and the mapping back
        is not ours to rely on. Built once per mesh and held, because it is what every
        hover and every highlight reads.
        """
        boundary = faces.boundary_of(mesh)
        ids = vtk_to_numpy(boundary.GetCellData().GetArray(arrayName)).astype(np.int64)
        locator = vtk.vtkStaticCellLocator()
        locator.SetDataSet(boundary)
        locator.BuildLocator()
        return boundary, ids, locator

    @staticmethod
    def faceAlongRay(lookup, start, end):
        """The face id of the first boundary cell the segment start -> end crosses, or None.

        No tolerance: the ray either goes through a cell of the boundary or it does not,
        and a cursor that is off the mesh has to report nothing rather than the nearest
        face to a line that missed.
        """
        _boundary, ids, locator = lookup
        crossing = [0.0, 0.0, 0.0]
        parametric = [0.0, 0.0, 0.0]
        along = vtk.reference(0.0)
        subId = vtk.reference(0)
        cellId = vtk.reference(0)
        hit = locator.IntersectWithLine(
            list(start), list(end), 0.0, along, crossing, parametric, subId, cellId
        )
        if not hit or int(cellId) < 0:
            return None
        return int(ids[int(cellId)])

    @staticmethod
    def statusText(measured, overrides=None, inherited=None, notes=()):
        """How many faces there are, how many came named from the clip, and what is left to do.

        The count of inherited names is said rather than left to the styling in the table: on a
        Fontan case there are twenty-six faces and the operator's question on opening the panel
        is how much of the work is already done.
        """
        if not measured:
            return ""
        overrides = overrides or {}
        inherited = inherited or {}
        fromClip = sum(1 for face in measured
                       if not overrides.get(face.face_id) and inherited.get(face.face_id))
        unnamed = sum(1 for face in measured
                      if not (overrides.get(face.face_id) or inherited.get(face.face_id)))

        typedHere = len(measured) - fromClip - unnamed

        parts = [_("{count} faces").format(count=len(measured))]
        if not fromClip:
            parts.append(_("{count} still to name").format(count=unnamed) if unnamed
                         else _("all named"))
        elif not unnamed and not typedHere:
            parts.append(_("all named from Clip Vessel"))
        else:
            parts.append(_("{count} named from Clip Vessel").format(count=fromClip))
            if unnamed:
                parts.append(_("{count} still to name").format(count=unnamed))
            else:
                parts.append(_("{count} named here").format(count=typedHere))
        return " ".join([", ".join(parts) + "."] + list(notes)).strip()

    @staticmethod
    def faceIdArrayName(mesh, faceIdArrayNames):
        """The first of the offered names the mesh carries, or None."""
        for candidate in (name.strip() for name in (faceIdArrayNames or "").split(",")):
            if candidate and mesh.GetCellData().GetArray(candidate) is not None:
                return candidate
        return None

    @staticmethod
    def showSolidColor(display):
        """Turn face colouring off, leaving the mesh a slightly see-through neutral grey."""
        display.SetScalarVisibility(False)
        display.SetColor(*SOLID_COLOR)
        display.SetOpacity(SOLID_OPACITY)

    @staticmethod
    def colourByFaceIds(display, arrayName):
        """Colour a model by its face ids, each face its own flat colour.

        The ids are labels rather than a measurement, so the colour table has to be one
        whose neighbouring entries differ -- a continuous one would give twenty pulmonary
        branches twenty shades of the same colour.
        """
        display.SetActiveScalarName(arrayName)
        display.SetActiveAttributeLocation(vtk.vtkAssignAttribute.CELL_DATA)
        for colourNodeId in ("vtkMRMLColorTableNodeRandom", "vtkMRMLColorTableNodeLabels"):
            if slicer.mrmlScene.GetNodeByID(colourNodeId) is not None:
                display.SetAndObserveColorNodeID(colourNodeId)
                break
        display.SetScalarRangeFlag(display.UseDataScalarRange)
        display.SetScalarVisibility(True)

    # -- the highlight -----------------------------------------------------
    def highlight(self, mesh, faceId, faceIdArrayNames, showEdges=False, boundary=None):
        """Put one face of the mesh in a node of its own, so it can be seen.

        The node is not saved with the scene and is hidden from the editors, so it stays
        out of the subject hierarchy and out of every node selector: it is a way of
        looking at the mesh, not a thing the case has. Hidden before it is added, because
        the subject hierarchy takes its item from a node as it arrives.

        `boundary` is the mesh's boundary if the caller already has it. Worth passing: it
        is a quarter of a second to extract on a mesh of a million cells, against the
        millisecond the one face then costs, and hovering asks for a face per mouse move.
        """
        arrayName = self.faceIdArrayName(mesh, faceIdArrayNames)
        if arrayName is None:
            return None
        surface = faces.boundary_of(mesh) if boundary is None else boundary
        ids = vtk_to_numpy(surface.GetCellData().GetArray(arrayName)).astype(np.int64)
        selected = np.flatnonzero(ids == faceId)
        face = cells.as_polydata(cells.extract(surface, selected))

        node = self.highlightNode()
        if node is None:
            node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLModelNode")
            node.SetName(HIGHLIGHT_NODE_NAME)
            node.SetSaveWithScene(False)
            node.SetHideFromEditors(True)
            node = slicer.mrmlScene.AddNode(node)
            node.CreateDefaultDisplayNodes()
            node.GetDisplayNode().SetSaveWithScene(False)
            node.GetDisplayNode().SetHideFromEditors(True)
        node.SetAndObserveMesh(face)

        # Set every time rather than only on the node's first use, so that a highlight left
        # over from an earlier run of the module cannot keep an earlier appearance.
        display = node.GetDisplayNode()
        display.SetColor(*HIGHLIGHT_COLOR)
        display.SetLineWidth(2)
        # One colour whichever way a cell faces. Slicer shifts a backface's hue so that
        # inside can be told from outside -- by (-0.05, -0.1, 0), which turns this yellow
        # into RGB (1, 0.73, 0.1), an orange. On a cap that is the strips the boundary
        # layer sweeps out at the vessel end, wound the other way from the cap's own
        # triangles, and a highlight saying "this is the face" has nothing to say about
        # which way round its cells are.
        display.SetBackfaceColorHSVOffset(0.0, 0.0, 0.0)
        display.SetEdgeVisibility(showEdges)
        display.SetVisibility(True)
        return node

    @staticmethod
    def highlightNode():
        """The highlight's node, which being hidden is not found by name lookups."""
        for index in range(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLModelNode")):
            node = slicer.mrmlScene.GetNthNodeByClass(index, "vtkMRMLModelNode")
            if node.GetName() == HIGHLIGHT_NODE_NAME:
                return node
        return None

    @classmethod
    def setHighlightEdgeVisibility(cls, visible):
        node = cls.highlightNode()
        if node is not None and node.GetDisplayNode() is not None:
            node.GetDisplayNode().SetEdgeVisibility(visible)

    @classmethod
    def clearHighlight(cls):
        node = cls.highlightNode()
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)


class SimVascularMeshPrepTest(ScriptedLoadableModuleTest):
    """Runs under Slicer; the package's own tests run without it."""

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_measureNamesAndExports()
        self.setUp()
        self.test_namesSurviveASavedScene()
        self.setUp()
        self.test_namesAreInheritedFromTheClip()
        self.setUp()
        self.test_overridesBeatInheritedNamesAndSurviveAScene()
        self.setUp()
        self.test_thePanelInheritsFollowsAndOverrides()

    # -- inherited names ---------------------------------------------------
    def recordedMesh(self, labels, faceIdsByIndex=(2, 3), wallFaceId=1):
        """A volume mesh carrying Clip Vessel's record, as one out of CFD Mesh Generator does.

        Built by hand rather than by running a clip: the clip is another extension's, and what
        this module has to be held to is reading the record, not producing it.
        """
        from svmeshcomplete import testing

        mesh = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "volume mesh")
        mesh.SetAndObserveMesh(testing.cube_mesh())
        clipPoints = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Clip points")
        controlPointIds = []
        for label in labels:
            index = clipPoints.AddControlPoint([0.0, 0.0, 0.0])
            clipPoints.SetNthControlPointLabel(index, label)
            controlPointIds.append(clipPoints.GetNthControlPointID(index))
        mesh.SetNodeReferenceID(CLIP_POINTS_NODE_REFERENCE, clipPoints.GetID())
        mesh.SetAttribute(FACE_ID_TO_CLIP_POINT_ID_ATTRIBUTE, json.dumps(
            {str(faceId): controlPointId
             for faceId, controlPointId in zip(faceIdsByIndex, controlPointIds)}))
        mesh.SetAttribute(WALL_FACE_ID_ATTRIBUTE, str(wallFaceId))
        return mesh, clipPoints

    def test_namesAreInheritedFromTheClip(self):
        """A mesh out of Clip Vessel arrives named, which is the whole point of the chain."""
        logic = SimVascularMeshPrepLogic()
        mesh, clipPoints = self.recordedMesh(("Inlet", "Outlet 1"))

        self.assertEqual(logic.wallFaceId(mesh), 1)
        self.assertEqual(logic.clipPointsNode(mesh), clipPoints)
        self.assertEqual(logic.inheritedNames(mesh).names,
                         {1: "wall", 2: "cap_Inlet", 3: "cap_Outlet_1"})

        # The labels are read from the markups node every time, so a rename upstream arrives.
        clipPoints.SetNthControlPointLabel(0, "RSVC")
        self.assertEqual(logic.inheritedNames(mesh).names[2], "cap_RSVC")

        # A deleted clip point takes its own face's name and no other. Keying the record by
        # control point ID is what buys this: by index, face 3's name would move onto face 2.
        clipPoints.RemoveNthControlPoint(0)
        derived = logic.inheritedNames(mesh)
        self.assertNotIn(2, derived.names)
        self.assertEqual(derived.names[3], "cap_Outlet_1")
        self.assertTrue(derived.notes)

        # A mesh from anywhere else carries no record, and then nothing is inherited.
        from svmeshcomplete import testing
        plain = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "imported mesh")
        plain.SetAndObserveMesh(testing.cube_mesh())
        self.assertIsNone(logic.wallFaceId(plain))
        self.assertEqual(logic.inheritedNames(plain).names, {})

        # And a record that cannot be read is not a crash: the panel falls back to typing.
        mesh.SetAttribute(FACE_ID_TO_CLIP_POINT_ID_ATTRIBUTE, "not json")
        self.assertEqual(logic.inheritedLabels(mesh), {})

        self.delayDisplay("Names inherited, followed, and dropped with their clip point")

    def test_overridesBeatInheritedNamesAndSurviveAScene(self):
        """Only what was typed is saved; the rest is worked out again from the clip.

        Which is what keeps the two from disagreeing. A saved copy of an inherited name would
        be a second answer to the same question, and the one on disk is the one that goes stale.
        """
        logic = SimVascularMeshPrepLogic()
        mesh, clipPoints = self.recordedMesh(("Inlet", "Outlet 1"))
        parameters = logic.getParameterNode()
        parameters.SetParameter(FACE_ID_ARRAY_PARAMETER, "CellEntityIds, ModelFaceID")
        parameters.SetNodeReferenceID(INPUT_MESH_REFERENCE, mesh.GetID())
        # One typed name against two inherited ones.
        parameters.SetParameter(NAMES_PARAMETER, logic.writeNamesParameter({3: "cap_azygous"}))

        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            bundle = os.path.join(directory, "scene.mrb")
            self.assertTrue(slicer.util.saveScene(bundle))
            # The parameter node is a singleton and Clear() keeps the singletons, so it is
            # taken out by hand - otherwise what comes back below is the node still in memory
            # and this asserts nothing about the scene on disk. Not Clear(1), which takes the
            # layout node with it and segfaults a running main window.
            slicer.mrmlScene.Clear()
            slicer.mrmlScene.RemoveNode(logic.getParameterNode())
            self.assertEqual(
                logic.readNamesParameter(logic.getParameterNode().GetParameter(NAMES_PARAMETER)),
                {}, "the parameter node should be empty before the scene is read back")

            self.assertTrue(slicer.util.loadScene(bundle))

            logic = SimVascularMeshPrepLogic()
            restored = logic.getParameterNode()
            overrides = logic.readNamesParameter(restored.GetParameter(NAMES_PARAMETER))
            self.assertEqual(overrides, {3: "cap_azygous"})

            reloadedMesh = restored.GetNodeReference(INPUT_MESH_REFERENCE)
            inherited = logic.inheritedNames(reloadedMesh).names
            self.assertEqual(inherited, {1: "wall", 2: "cap_Inlet", 3: "cap_Outlet_1"})
            # What the panel shows: the override where there is one, the clip's name otherwise.
            effective = {faceId: overrides.get(faceId) or inherited.get(faceId, "")
                         for faceId in (1, 2, 3)}
            self.assertEqual(effective, {1: "wall", 2: "cap_Inlet", 3: "cap_azygous"})

        self.delayDisplay("Overrides saved, inherited names rebuilt")

    def test_thePanelInheritsFollowsAndOverrides(self):
        """The three behaviours the split between overrides and inherited names buys.

        Driven through the widget rather than the logic, because what is being checked is the
        panel: that a mesh out of Clip Vessel opens named with Export already enabled, that a
        rename upstream arrives without anything being reselected, and that typing and clearing
        move a face between the two states. The styling is checked too - an inherited name has
        to be visibly not a chosen one, and that is the only thing standing between an operator
        and a positional default they never looked at.
        """
        import tempfile

        widget = slicer.util.getModuleWidget("SimVascularMeshPrep")
        mesh, clipPoints = self.recordedMesh(("Inlet", "Outlet 1"))
        widget.ui.faceIdArrayLineEdit.text = "CellEntityIds, ModelFaceID"
        # Export needs somewhere to write as well as every face named, and what is being
        # checked below is the naming half of that.
        widget.ui.outputDirectoryPathLineEdit.currentPath = tempfile.mkdtemp()
        widget.ui.inputMeshSelector.setCurrentNode(mesh)

        table = widget.ui.facesTable
        self.assertEqual(table.rowCount, 3)

        def shown():
            return {int(table.item(row, 0).text()): table.item(row, NAME_COLUMN).text()
                    for row in range(table.rowCount)}

        def rowOf(faceId):
            return [row for row in range(table.rowCount)
                    if int(table.item(row, 0).text()) == faceId][0]

        # It opens named, with nothing typed, and ready to export.
        self.assertEqual(shown(), {1: "wall", 2: "cap_Inlet", 3: "cap_Outlet_1"})
        self.assertEqual(widget._overrides, {})
        self.assertTrue(widget.ui.exportButton.enabled,
                        "a mesh out of Clip Vessel should open ready to export")
        self.assertTrue(table.item(rowOf(2), NAME_COLUMN).font().italic(),
                        "an inherited name has to be drawn as one")

        # 1. Renamed upstream: the face name follows, with nothing reselected here.
        clipPoints.SetNthControlPointLabel(0, "RSVC")
        self.assertEqual(shown()[2], "cap_RSVC")

        # 2. Typed over: it sticks, is no longer drawn as inherited, and stops following.
        row = rowOf(2)
        table.item(row, NAME_COLUMN).setText("cap_superior_vena_cava")
        widget.onNameEdited(row, NAME_COLUMN)
        self.assertEqual(widget._overrides, {2: "cap_superior_vena_cava"})
        self.assertFalse(table.item(rowOf(2), NAME_COLUMN).font().italic())
        clipPoints.SetNthControlPointLabel(0, "something else")
        self.assertEqual(shown()[2], "cap_superior_vena_cava")

        # 3. Cleared: the inherited name comes back rather than the face going blank.
        row = rowOf(2)
        table.item(row, NAME_COLUMN).setText("")
        widget.onNameEdited(row, NAME_COLUMN)
        self.assertEqual(widget._overrides, {})
        self.assertEqual(shown()[2], "cap_something_else")
        self.assertTrue(table.item(rowOf(2), NAME_COLUMN).font().italic())

        # A deleted clip point leaves its face to be named, and blocks the export until it is.
        clipPoints.RemoveNthControlPoint(0)
        self.assertEqual(shown()[2], "")
        self.assertEqual(shown()[3], "cap_Outlet_1")
        self.assertFalse(widget.ui.exportButton.enabled)

        # The wall is known rather than guessed, which is what flatness is judged against.
        self.assertEqual(widget.wallFaceId(), 1)

        # Selecting away and back is not something the panel may be left broken by: the
        # observers on the clip points have to come off the old mesh and onto the new one.
        widget.ui.inputMeshSelector.setCurrentNode(None)
        self.assertEqual(widget._inherited, {})
        self.assertIsNone(widget._clipPointsNode)
        widget.ui.inputMeshSelector.setCurrentNode(mesh)
        self.assertEqual(widget._clipPointsNode, clipPoints)

        self.delayDisplay("The panel inherits, follows a rename, and takes an override")

    def test_measureNamesAndExports(self):
        import tempfile

        from svmeshcomplete import testing

        logic = SimVascularMeshPrepLogic()
        mesh = testing.cube_mesh()
        measured = logic.measure(mesh, "CellEntityIds, ModelFaceID")
        self.assertEqual(len(measured), 3)

        wall = max(measured, key=lambda face: face.cell_count)
        names = {
            face.face_id: "wall" if face is wall else f"cap_{face.face_id}"
            for face in measured
        }

        directory = tempfile.mkdtemp()
        result = logic.export(mesh, measured, names, directory, "CellEntityIds")
        self.assertEqual(result.element_type, "Tetra")
        self.assertEqual(len(result.face_surfaces), 3)
        self.assertTrue(os.path.isfile(os.path.join(directory, mesh_complete.VOLUME_MESH_NAME)))

        # The names are the scene's, not the folder's.
        self.assertFalse(os.path.isfile(os.path.join(directory, "face_table.csv")))

        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "mesh")
        node.SetAndObserveMesh(mesh)
        highlighted = logic.highlight(mesh, wall.face_id, "CellEntityIds", showEdges=True)
        self.assertEqual(highlighted.GetMesh().GetNumberOfCells(), wall.cell_count)
        self.assertTrue(highlighted.GetDisplayNode().GetEdgeVisibility())
        self.assertEqual(
            tuple(round(c, 2) for c in highlighted.GetDisplayNode().GetColor()),
            HIGHLIGHT_COLOR,
        )
        # A backface has to be the same colour: Slicer shifts its hue by default, which
        # turns this yellow orange on the strips a boundary layer leaves at a vessel end.
        self.assertEqual(
            tuple(highlighted.GetDisplayNode().GetBackfaceColorHSVOffset()), (0.0, 0.0, 0.0)
        )
        self.assertTrue(highlighted.GetHideFromEditors())
        self.assertFalse(highlighted.GetSaveWithScene())

        # The edges toggle reaches it.
        logic.setHighlightEdgeVisibility(False)
        self.assertFalse(highlighted.GetDisplayNode().GetEdgeVisibility())

        logic.clearHighlight()
        self.assertIsNone(logic.highlightNode())

        # Picking a face along a ray, which is what hovering over one comes down to: the
        # cursor is a line through the scene, and the face is the first one it crosses.
        # The fixture is the unit cube, face 2 its bottom and face 3 its top.
        lookup = logic.faceLookup(mesh, "CellEntityIds")
        self.assertEqual(logic.faceAlongRay(lookup, (0.5, 0.5, 2.0), (0.5, 0.5, -1.0)), 3)
        self.assertEqual(logic.faceAlongRay(lookup, (0.5, 0.5, -1.0), (0.5, 0.5, 2.0)), 2)
        self.assertEqual(logic.faceAlongRay(lookup, (-1.0, 0.5, 0.5), (2.0, 0.5, 0.5)), 1)
        # The near face, not whichever the locator reaches first: looking down on the cube
        # from above has to name its top, though the same line leaves through its bottom.
        self.assertEqual(logic.faceAlongRay(lookup, (0.5, 0.5, 9.0), (0.5, 0.5, -9.0)), 3)
        # A cursor off the mesh is over nothing, rather than over the nearest face to a
        # line that missed. A ray beside the cube and one stopping short of it both miss.
        self.assertIsNone(logic.faceAlongRay(lookup, (2.0, 2.0, -1.0), (2.0, 2.0, 2.0)))
        self.assertIsNone(logic.faceAlongRay(lookup, (0.5, 0.5, 9.0), (0.5, 0.5, 4.0)))

        # A scene with nowhere to write to suggests nowhere, rather than Documents.
        self.assertEqual(logic.suggestedOutputDirectory(), "")

        # The count has to follow the naming: it is read after every name typed.
        self.assertEqual(logic.statusText([], {}), "")
        self.assertIn("3 still to name", logic.statusText(measured, {}))
        partly = {wall.face_id: "wall"}
        self.assertIn("2 still to name", logic.statusText(measured, partly))
        # Not dict(names, **{...}): the keys are face ids, and keyword expansion needs strings.
        blank = {**names, wall.face_id: ""}
        self.assertIn("1 still to name", logic.statusText(measured, blank))
        self.assertIn("all named", logic.statusText(measured, names))

        # The three display toggles, each reading the state it is turning around.
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        self.assertEqual(logic.faceIdArrayName(mesh, "CellEntityIds, ModelFaceID"), "CellEntityIds")
        logic.colourByFaceIds(display, "CellEntityIds")
        self.assertTrue(display.GetScalarVisibility())
        self.assertEqual(display.GetActiveScalarName(), "CellEntityIds")
        logic.showSolidColor(display)
        self.assertFalse(display.GetScalarVisibility())
        self.assertEqual(tuple(round(c, 2) for c in display.GetColor()), SOLID_COLOR)
        self.assertAlmostEqual(display.GetOpacity(), SOLID_OPACITY)

        display.SetEdgeVisibility(not display.GetEdgeVisibility())
        self.assertTrue(display.GetEdgeVisibility())

        self.delayDisplay("Measured, named, exported, highlighted and coloured")

    def test_namesSurviveASavedScene(self):
        """The naming is the expensive part, so it has to come back with the scene."""
        import tempfile

        from svmeshcomplete import testing

        logic = SimVascularMeshPrepLogic()
        names = {1: "wall", 2: "cap_RSVC", 3: "cap_lpa_a"}

        mesh = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "volume mesh")
        mesh.SetAndObserveMesh(testing.cube_mesh())

        parameters = logic.getParameterNode()
        parameters.SetParameter(NAMES_PARAMETER, logic.writeNamesParameter(names))
        parameters.SetParameter(FACE_ID_ARRAY_PARAMETER, "CellEntityIds, ModelFaceID")
        parameters.SetNodeReferenceID(INPUT_MESH_REFERENCE, mesh.GetID())

        scene = os.path.join(tempfile.mkdtemp(), "scene.mrb")
        self.assertTrue(slicer.util.saveScene(scene))

        # The parameter node is what must not survive, or this would be testing that a value
        # is still in memory rather than that it came back off disk. It is a singleton, so it
        # is taken out by hand: Clear(1) removes every singleton, and taking the layout node
        # out from under a running main window segfaults the layout manager ("The layout to be
        # removed is not the same as the stored one") - which made this test crash Slicer
        # whenever it was run from the Reload and Test button rather than headless.
        slicer.mrmlScene.Clear()
        slicer.mrmlScene.RemoveNode(logic.getParameterNode())
        self.assertEqual(
            logic.readNamesParameter(logic.getParameterNode().GetParameter(NAMES_PARAMETER)), {}
        )

        self.assertTrue(slicer.util.loadScene(scene))
        restored = logic.getParameterNode()
        self.assertEqual(
            logic.readNamesParameter(restored.GetParameter(NAMES_PARAMETER)), names
        )
        self.assertEqual(
            restored.GetParameter(FACE_ID_ARRAY_PARAMETER), "CellEntityIds, ModelFaceID"
        )
        self.assertIsNotNone(restored.GetNodeReference(INPUT_MESH_REFERENCE))

        # A parameter that is not readable loses the names rather than the panel.
        self.assertEqual(logic.readNamesParameter("not json"), {})

        self.delayDisplay("Names survived a saved scene")
