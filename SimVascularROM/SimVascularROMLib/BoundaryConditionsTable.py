"""What the SimVascular simulation panels share: a mesh's named faces, and the conditions on them.

Two panels set boundary conditions on the caps of a mesh SimVascular Mesh Prep has named --
SimVascular ROM Simulation for the 0D model, SimVascular MultiPhysics for the 3D one -- and they
set the *same* ones: one set per mesh, kept on the mesh node, so that a 0D model tuned in one
panel is the 3D case written in the other, with nothing copied between them to go stale. The
table they are set in is therefore here, once, rather than in each panel: two copies of it would
be two tables that drift, and the first sign would be a condition that reads one way in one
panel and another in the other.

What differs between the panels stays in them. The 0D panel has a source -- the cap its
centerlines start at -- and shows it in bold here through `emphasised`; the 3D panel has none.
Each says what a change means to it through the `changed` and `selectionChanged` callbacks.
"""

import dataclasses
import json
import logging

import ctk
import qt
import slicer
import vtk
from slicer.i18n import tr as _

from svmeshcomplete import faces
from svmeshcomplete.face_table import Face
from svromsetup import boundary_conditions as bcs

# The face names are SimVascular Mesh Prep's, kept on the mesh as `{"<face id>": "<name>"}`.
# Read here, never written: a name typed in two panels is two names, and the one in Mesh Prep is
# the one the mesh-complete folder and every case of this mesh are written under.
FACE_NAMES_ATTRIBUTE = "SimVascularMeshPrep.FaceNames"

# Where the conditions are kept: on the mesh, for the reason Mesh Prep keeps the names there. A
# scene can hold a pre-op and a post-op anatomy, each with a face 4, and conditions kept per scene
# would move between them when a selector changed. Named for the panel that first kept them, and
# kept under that name so that scenes saved before the 3D panel existed still open with theirs.
BOUNDARY_CONDITIONS_ATTRIBUTE = "SimVascularROM.BoundaryConditions"

# Where the last waveform was loaded from, shared by both panels: a case's waveforms are kept
# together, whichever solver they are being given to.
WAVEFORM_DIRECTORY_SETTING = "SimVascularROM/WaveformDirectory"

# A .flow file's table, as SimVascular ROM Simulation's reader makes it: marked with the file it
# was read from, so that a node selector can offer waveforms alone, and anything reading the scene
# can tell a waveform's table from any other, and where it came from; a time and a flow column.
# Here rather than beside the reader, so that the table can filter by it in either panel.
FLOW_FILE_ATTRIBUTE = "SimVascularROM.FlowFile"
FLOW_TIME_COLUMN, FLOW_VALUE_COLUMN = "time", "flow"
# Which flow table a cap's inflow is read from, as a reference on the mesh, one role per cap: a
# reference rather than an ID in the conditions' text, so that Slicer keeps it right when the
# scene is saved, opened, or imported into another whose IDs it has to renumber.
INFLOW_TABLE_ROLE = "SimVascularROMInflowTable"

COLUMNS = ("Name", "Type", "Values", "Flow table", "Invert")
NAME_COLUMN = COLUMNS.index("Name")
TYPE_COLUMN = COLUMNS.index("Type")
VALUES_COLUMN = COLUMNS.index("Values")
# The flow table an inflow's waveform is read from, chosen in the scene.
TABLE_COLUMN = COLUMNS.index("Flow table")
# A box on an inflow read from a table or a file: ticked, its flows are those multiplied by -1.
INVERT_COLUMN = COLUMNS.index("Invert")

# The sign each panel's solver takes an inflow in, which is the sign the panel shows it in:
# positive into the model for the 0D solvers, as svromsetup keeps every condition; negative for
# svMultiPhysics, which measures a face's flow along its outward normal, as its own .flow files
# have it. Shown in its solver's sign so that a waveform drawn is the waveform written.
INTO_THE_MODEL_POSITIVE, INTO_THE_MODEL_NEGATIVE = 1, -1


def inSign(condition, sign):
    """A condition with its inflow in `sign`'s terms: the same condition, shown or kept, for an
    inflow in the other sign multiplied by -1. Its own inverse, so it shows a kept condition and
    keeps a shown one alike. Not `Inflow.flipped`, which turns a waveform round and says so: this
    is the one inflow, written down the other way."""
    if sign > 0 or not isinstance(condition, bcs.Inflow):
        return condition
    return dataclasses.replace(condition, flow=tuple(-flow for flow in condition.flow))


def flowTableWaveform(tableNode):
    """A flow table's waveform, time and flow, its first two columns when they are not named so;
    raised as a condition error when it is not one, as a file that is not is."""
    table = tableNode.GetTable() if tableNode is not None else None
    if table is None or table.GetNumberOfColumns() < 2:
        raise bcs.BoundaryConditionError(_("A flow table needs a time and a flow column."))
    columns = [table.GetColumnByName(FLOW_TIME_COLUMN), table.GetColumnByName(FLOW_VALUE_COLUMN)]
    if None in columns:
        columns = [table.GetColumn(0), table.GetColumn(1)]
    values = []
    for column in columns:
        try:
            values.append(tuple(float(column.GetValue(row)) for row in range(column.GetNumberOfTuples())))
        except (TypeError, ValueError):
            raise bcs.BoundaryConditionError(
                _("{name}'s columns are not numbers.").format(name=tableNode.GetName())) from None
    return bcs.Inflow(*values)

# What the type column offers, the first meaning none: a cap with no condition is shown as one,
# rather than given a default that looks like a decision.
KIND_CHOICES = (None,) + bcs.KINDS
NO_CONDITION_TEXT = "-"

INVALID_VALUES_COLOR = qt.QColor(200, 60, 40)
# A value's box in the Values column is named for the value, so that it can be found by it.
VALUE_BOX_PREFIX = "valueBox_"
# What each value is, in its units, for its label's and box's tooltip: the label is its name alone,
# as SimVascular names it, to keep four of them in the column.
VALUE_TIPS = {
    "Q": "Q: a steady flow, in mL/s.",
    "Rp": "Rp: the proximal resistance, in dyn·s/cm⁵.",
    "C": "C: the compliance, in cm⁵/dyn.",
    "Rd": "Rd: the distal resistance, in dyn·s/cm⁵.",
    "Pd": "Pd: the distal pressure, in dyn/cm² (1 mmHg = 1333.22). Left empty, it is 0.",
    "R": "R: the resistance, in dyn·s/cm⁵, draining to zero pressure.",
}
WAVEFORM_COLOR = (31, 94, 160)
WARNING_COLOR = "#d08000"


class WaveformPlot:
    """The selected inflow's waveform, drawn under the table: every point marked, a line at zero to
    read its sign against, and its points, period, mean and range under it.

    Under the table rather than in a window opened as a file is loaded (the user's choice): a
    window showed the curve once, as it came in, and nothing after; here it is the selected cap's
    as it is used now, inverted or not, whenever that cap is selected -- which is when its shape is
    in question. Whether it is upside down is the Invert column's to say, beside the cap's other
    values.

    The tables and plots are kept on the instance: the chart holds them only weakly from Python's
    side, and one collected while shown takes the chart down with it.
    """

    def __init__(self):
        self.widget = qt.QWidget()
        layout = qt.QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.chartView = ctk.ctkVTKChartView()
        self.chartView.minimumHeight = 220
        layout.addWidget(self.chartView)
        self.summaryLabel = qt.QLabel()
        self.summaryLabel.wordWrap = True
        layout.addWidget(self.summaryLabel)
        self.curve, self.zero = vtk.vtkTable(), vtk.vtkTable()
        for table in (self.curve, self.zero):
            for name in ("time (s)", "flow (mL/s)"):
                column = vtk.vtkDoubleArray()
                column.SetName(name)
                table.AddColumn(column)
        self.zeroPlot = vtk.vtkPlotLine()
        self.zeroPlot.SetInputData(self.zero, 0, 1)
        self.zeroPlot.SetColor(150, 150, 150, 255)
        self.zeroPlot.SetWidth(1.0)
        self.linePlot = vtk.vtkPlotLine()
        self.linePlot.SetInputData(self.curve, 0, 1)
        self.linePlot.SetColor(*WAVEFORM_COLOR, 255)
        self.linePlot.SetWidth(1.5)
        self.linePlot.SetMarkerStyle(vtk.vtkPlotPoints.CIRCLE)
        chart = self.chartView.chart()
        chart.AddPlot(self.zeroPlot)
        chart.AddPlot(self.linePlot)
        chart.GetAxis(vtk.vtkAxis.BOTTOM).SetTitle(_("time (s)"))
        chart.GetAxis(vtk.vtkAxis.LEFT).SetTitle(_("flow (mL/s)"))
        chart.SetShowLegend(False)
        self.name = None
        self.widget.visible = False

    def show(self, name, inflow, sign=INTO_THE_MODEL_POSITIVE, inverted=False):
        """Draw a cap's inflow as the panel's solver takes it, in `sign`, and say what it comes to:
        which way is in, and, when its mean goes the other way, that it takes flow out."""
        self.name = name
        self.linePlot.SetMarkerSize(4.0 if len(inflow.time) <= 400 else 2.0)
        for table, times, flows in ((self.curve, inflow.time, inflow.flow),
                                    (self.zero, (0.0, inflow.period), (0.0, 0.0))):
            table.SetNumberOfRows(len(times))
            for row, (time, flow) in enumerate(zip(times, flows)):
                table.SetValue(row, 0, vtk.vtkVariant(float(time)))
                table.SetValue(row, 1, vtk.vtkVariant(float(flow)))
            table.Modified()
        text = _("{name}{inverted}: {points} points over {period:g} s, mean {mean:.4g} mL/s, from "
                 "{low:.4g} to {high:.4g} mL/s.").format(
            name=name, inverted=_(", inverted") if inverted else "", points=len(inflow.time),
            period=inflow.period, mean=inflow.mean, low=min(inflow.flow), high=max(inflow.flow))
        text += " " + intoTheModel(sign)
        backwards = inflow.mean * sign < 0
        if backwards:
            text += " " + backwardsNote(sign)
        self.summaryLabel.text = text
        self.summaryLabel.styleSheet = f"QLabel {{ color: {WARNING_COLOR}; }}" if backwards else ""
        self.widget.visible = True
        self.chartView.chart().RecalculateBounds()
        self.chartView.scene().SetDirty(True)
        self.chartView.renderWindow().Render()

    def hide(self):
        self.name = None
        self.widget.visible = False


def intoTheModel(sign):
    """Which way is in, said wherever an inflow is shown: nothing in a file says which it is."""
    if sign < 0:
        return _("Into the model is negative, as svMultiPhysics takes it: flow is measured along the "
                 "cap's outward normal.")
    return _("Into the model is positive, as the 0D solvers take it.")


def warningIcon():
    """The style's own warning sign, as Slicer's dialogs show it."""
    return qt.QApplication.style().standardIcon(qt.QStyle.SP_MessageBoxWarning)


def backwardsNote(sign):
    """What an inflow whose mean goes out of the model means, and what to do about it."""
    if sign < 0:
        return _("The mean is positive: as it is, this cap takes flow out of the model. Tick Invert "
                 "if the curve is upside down, as a waveform for a 0D solver, or from the ROM "
                 "panel, is.")
    return _("The mean is negative: as it is, this cap takes flow out of the model. Tick Invert if "
             "the curve is upside down, as a file written for svMultiPhysics is.")


def faceNames(meshNode):
    """`{face id: name}` as SimVascular Mesh Prep saved them on the mesh, or {}."""
    text = meshNode.GetAttribute(FACE_NAMES_ATTRIBUTE) if meshNode is not None else None
    if not text:
        return {}
    try:
        stored = json.loads(text)
    except ValueError:
        logging.warning("Could not read the face names on %s.", meshNode.GetName())
        return {}
    names = {}
    for faceId, name in stored.items():
        try:
            names[int(faceId)] = str(name)
        except (TypeError, ValueError):
            continue
    return names


class NamedMeshFaces:
    """A mesh's faces as Mesh Prep named them, measured once for as long as the mesh is the same.

    Measured once because coming back to a panel reads the names again, which is cheap, and would
    measure the faces again with them, which on a clinical volume mesh of a million cells is not.
    The boundary is kept too: it is what the highlight is cut from.
    """

    def __init__(self):
        self._key = None
        self._measured = []
        self.boundary = None
        self.arrayName = None
        self.names = {}
        self.caps = []
        self.unnamed = []
        self.walls = 0

    def read(self, node):
        """Measure the node's faces and read their names; raises ValueError if it has no faces.

        :return: whether the faces are named at all. Unnamed faces are left out of `caps` and
          listed in `unnamed`, so a panel can say which ones Mesh Prep has still to name.
        """
        key = (node.GetID(), node.GetMesh().GetMTime())
        if self._key != key:
            self.arrayName = faces.find_face_id_array(node.GetMesh())
            self._measured = faces.measure_faces(node.GetMesh(), self.arrayName)
            self.boundary = faces.boundary_of(node.GetMesh())
            self._key = key
        self.names = faceNames(node)
        self.unnamed = [face.face_id for face in self._measured if not self.names.get(face.face_id)]
        self.caps = [(face.face_id, self.names[face.face_id], face) for face in self._measured
                     if self.names.get(face.face_id)
                     and Face(face.face_id, self.names[face.face_id]).is_cap]
        self.walls = len(self._measured) - len(self.caps) - len(self.unnamed)
        return bool(self.names)

    @property
    def count(self):
        return len(self._measured)

    def describe(self):
        """What the mesh panel says about these faces."""
        if not self.names:
            return _("{count} faces, none of them named. The names are the boundary conditions' "
                     "and the results', and they come from SimVascular Mesh Prep.").format(
                count=self.count)
        text = _("{caps} caps and {walls} wall face(s), named in SimVascular Mesh Prep.").format(
            caps=len(self.caps), walls=self.walls)
        if self.unnamed:
            text += " " + _("Face(s) {faces} have no name and are left out; name them in Mesh "
                            "Prep.").format(faces=", ".join(str(faceId) for faceId in self.unnamed))
        return text


def highlight(mesh, faceId, arrayName, boundary=None):
    """Show one face, with Mesh Prep's own highlight, so that every panel shows a face the same way."""
    from SimVascularMeshPrep import SimVascularMeshPrepLogic
    SimVascularMeshPrepLogic().highlight(mesh, faceId, arrayName or "", boundary=boundary)


def clearHighlight():
    try:
        from SimVascularMeshPrep import SimVascularMeshPrepLogic
    except ImportError:
        return
    SimVascularMeshPrepLogic.clearHighlight()


class BoundaryConditionsTable:
    """One row per cap: its name, the type of its condition, and the condition's values.

    The values are typed one to a box, each labelled with its name, so that their order is on
    screen rather than in the hint. The values of an inflow are a waveform, which nobody types: it
    is chosen in the Flow table column, from the tables the `.flow` reader makes of files dropped
    on Slicer, and a steady flow is typed in its Q box instead. A waveform has a box in the Invert column,
    unticked as chosen, and the selected one is drawn under the table, so that one upside down --
    as a file written for svMultiPhysics is -- is seen and turned round in place. An inflow follows
    its table: edited in Slicer's table view, the cap's waveform is read from it again. A value
    that cannot be read stays in its box in red, with the reason as its tooltip, rather than
    vanishing on a typo; the condition is not one until every value it needs reads.

    An inflow is shown, drawn and typed in the sign the panel's solver takes it in, `inflowSign`,
    and kept positive into the model whatever that is, so that both panels share one condition: a
    file written for svMultiPhysics, negative, is drawn as it is in the 3D panel, and turned round
    by its Invert box in the 0D one; a 0D waveform the other way about. The box says whether the
    curve shown is its table's multiplied by -1, so the same cap can be ticked in one panel and not
    in the other -- and nothing has to be turned round in moving a case between them.

    The plot is put in the layout that holds the hint, after it, so that both panels have it
    without either's UI file knowing of it.
    """

    def __init__(self, table, hintLabel, setStatus, changed=None, selectionChanged=None,
                 inflowSign=INTO_THE_MODEL_POSITIVE):
        self.table = table
        self.hintLabel = hintLabel
        self.setStatus = setStatus
        self.changed = changed or (lambda: None)
        self.selectionChanged = selectionChanged or (lambda faceIds: None)
        # The sign the panel's solver takes an inflow in, and so the sign it is shown and typed in
        # here. The conditions are kept positive into the model whatever it is, as svromsetup keeps
        # them, so that both panels share one: a cap given 5 mL/s in one is -5 in the other.
        self.inflowSign = inflowSign
        self.mesh = None
        self.faces = None
        self.caps = []
        self.conditions = {}
        self.invalid = {}
        # The cap shown in bold, and what its tooltip says about why: the 0D panel's source.
        self.emphasised = None
        self.emphasisNote = ""
        self.updating = False
        # The flow tables chosen, by cap, and the observation that reads one again when it changes.
        self.flowTables = {}
        self.tableObservations = {}

        table.setColumnCount(len(COLUMNS))
        table.setHorizontalHeaderLabels([_(name) for name in COLUMNS])
        table.verticalHeader().setVisible(False)
        # Every column as wide as what is in it, the values' boxes too: stretched to what was left,
        # the values' column was too narrow in a module panel to read four boxes in, and the flow
        # table's, stretched instead, too narrow to read a table's name. A panel narrower than
        # the table scrolls it.
        table.horizontalHeader().setSectionResizeMode(qt.QHeaderView.ResizeToContents)
        # Nothing is typed into a cell: the values have boxes of their own, and a name is Mesh
        # Prep's. A cell opened for typing would open under the boxes, over the values they show.
        table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        table.connect("cellChanged(int,int)", self.onValuesEdited)
        table.connect("itemSelectionChanged()", self.onSelectionChanged)

        self.plot = WaveformPlot()
        holder = hintLabel.parentWidget() if hintLabel is not None else None
        layout = holder.layout() if holder is not None else None
        if layout is not None and hasattr(layout, "insertWidget"):
            layout.insertWidget(layout.indexOf(hintLabel) + 1, self.plot.widget)

    # -- which mesh ---------------------------------------------------------------------
    def setMesh(self, mesh, namedFaces):
        """Show `mesh`'s caps, with the conditions saved on it, or nothing for None."""
        self.mesh = mesh
        self.faces = namedFaces
        self.caps = list(namedFaces.caps) if (mesh is not None and namedFaces is not None) else []
        self.invalid = {}
        capIds = {faceId for faceId, _name, _geometry in self.caps}
        self.conditions = ({faceId: condition for faceId, condition in bcs.from_json(
            mesh.GetAttribute(BOUNDARY_CONDITIONS_ATTRIBUTE)).items() if faceId in capIds}
            if self.caps else {})
        # Each inflow's table, as saved on the mesh, watched again. Not read again here: what was
        # read from it is kept on the mesh at full precision, and a table saved with a scene comes
        # back at the six figures Slicer writes a table's numbers to.
        self.unobserveTables()
        self.flowTables = {}
        for faceId in capIds:
            tableNode = mesh.GetNodeReference(f"{INFLOW_TABLE_ROLE}{faceId}") if self.caps else None
            if tableNode is not None:
                self.flowTables[faceId] = tableNode
                self.observeTable(faceId, tableNode)
        clearHighlight()
        self.populate()

    def save(self):
        """Write the conditions onto the mesh, as they are edited."""
        if self.mesh is None or not self.caps:
            return
        capIds = {faceId for faceId, _name, _geometry in self.caps}
        self.mesh.SetAttribute(BOUNDARY_CONDITIONS_ATTRIBUTE, bcs.to_json(
            {faceId: condition for faceId, condition in self.conditions.items()
             if faceId in capIds}))

    # -- what is in it ----------------------------------------------------------------------
    def capName(self, faceId):
        return next((name for capId, name, _geometry in self.caps if capId == faceId), str(faceId))

    def capNames(self):
        return {faceId: name for faceId, name, _geometry in self.caps}

    def conditionsByName(self):
        names = self.capNames()
        return {names[faceId]: condition for faceId, condition in self.conditions.items()
                if faceId in names}

    def period(self):
        return bcs.cycle_period(self.conditions.values())

    def kindOf(self, faceId):
        """The kind shown for a face: its condition's, or the one chosen for what failed to read."""
        if faceId in self.invalid:
            return self.invalid[faceId][0]
        condition = self.conditions.get(faceId)
        return condition.kind if condition is not None else None

    def rowOf(self, faceId):
        return next((row for row, (capId, _name, _geometry) in enumerate(self.caps)
                     if capId == faceId), None)

    def selectedFaceIds(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        return [self.caps[row][0] for row in rows if row < len(self.caps)]

    # -- drawing it ------------------------------------------------------------------------
    def populate(self):
        self.updating = True
        try:
            table = self.table
            table.setRowCount(len(self.caps))
            for row, (faceId, name, _geometry) in enumerate(self.caps):
                nameItem = table.item(row, NAME_COLUMN)
                if nameItem is None:
                    nameItem = qt.QTableWidgetItem()
                    table.setItem(row, NAME_COLUMN, nameItem)
                nameItem.setText(name)
                nameItem.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
                font = nameItem.font()
                font.setBold(faceId == self.emphasised)
                nameItem.setFont(font)
                nameItem.setToolTip(self.emphasisNote if faceId == self.emphasised else "")

                combo = table.cellWidget(row, TYPE_COLUMN)
                if combo is None:
                    combo = qt.QComboBox()
                    for kind in KIND_CHOICES:
                        combo.addItem(_(kind) if kind else NO_CONDITION_TEXT)
                    combo.connect("currentIndexChanged(int)",
                                  lambda index, combo=combo: self.onKindChanged(combo, index))
                    table.setCellWidget(row, TYPE_COLUMN, combo)
                combo.setProperty("faceId", faceId)
                combo.currentIndex = KIND_CHOICES.index(self.kindOf(faceId))
                self.populateValues(row, faceId)
                self.populateFlowTable(row, faceId)
                self.populateInvert(row, faceId)
        finally:
            self.updating = False
        self.updateHint()
        self.updatePlot()

    def populateFlowTable(self, row, faceId):
        """The Flow table selector, on an inflow alone: the scene's flow tables, and none."""
        selector = self.table.cellWidget(row, TABLE_COLUMN)
        if self.kindOf(faceId) != bcs.INFLOW:
            if selector is not None:
                self.table.removeCellWidget(row, TABLE_COLUMN)
            return
        if selector is None:
            selector = slicer.qMRMLNodeComboBox()
            selector.nodeTypes = ["vtkMRMLTableNode"]
            # The .flow reader's tables alone: any other table is not a waveform to offer.
            selector.addAttribute("vtkMRMLTableNode", FLOW_FILE_ATTRIBUTE)
            selector.noneEnabled = True
            selector.addEnabled = False
            selector.removeEnabled = False
            selector.renameEnabled = False
            selector.showChildNodeTypes = False
            selector.setMRMLScene(slicer.mrmlScene)
            selector.toolTip = _("The flow table this inflow's waveform is read from: drop a .flow "
                                 "file on Slicer, or use Add Data, to make one.")
            # Activated, not changed: a selector whose node is deleted moves on to another of the
            # scene's tables by itself, and that is nobody's choice of waveform. The cap is read
            # when it fires, as the Type box's is: a row is another cap's once another mesh is shown.
            selector.connect("nodeActivated(vtkMRMLNode*)", lambda node, selector=selector:
                             self.onFlowTableChosen(int(selector.property("faceId")), node))
            # Put right once the change is over, by a timer of the selector's own, which goes with
            # it: one of the module's would fire into a table already destroyed, as Slicer closes.
            putRight = qt.QTimer(selector)
            putRight.setSingleShot(True)
            putRight.setInterval(0)
            putRight.connect("timeout()", lambda selector=selector:
                             self.onFlowTableShown(int(selector.property("faceId"))))
            selector.connect("currentNodeChanged(vtkMRMLNode*)", putRight, "start()")
            self.table.setCellWidget(row, TABLE_COLUMN, selector)
        selector.setProperty("faceId", faceId)
        wasBlocked = selector.blockSignals(True)
        selector.setCurrentNode(self.flowTables.get(faceId))
        selector.blockSignals(wasBlocked)

    def shown(self, condition):
        """A condition as this panel shows it: an inflow in its solver's sign."""
        return inSign(condition, self.inflowSign)

    def kept(self, condition):
        """A condition shown here as it is kept: an inflow positive into the model."""
        return inSign(condition, self.inflowSign)

    def isInverted(self, condition):
        """Whether the curve this panel shows is its table's multiplied by -1: what the box says.

        The condition's own `negated` says so of the condition as kept, positive into the model;
        where the solver takes inflow negative, the curve shown is that one multiplied by -1 again.
        """
        return bool(condition.negated) != (self.inflowSign < 0)

    def hasWaveform(self, faceId):
        """Whether a cap's inflow was read rather than typed: from a table or a file, or a waveform
        kept after its table was deleted. A steady flow typed is typed with its sign."""
        condition = self.conditions.get(faceId)
        return isinstance(condition, bcs.Inflow) and bool(
            faceId in self.flowTables or condition.source or not condition.is_steady)

    def populateInvert(self, row, faceId):
        """The Invert box, on an inflow read rather than typed alone: anything else has no flow to
        turn round, or was typed the right way round.

        With a warning sign beside it while the inflow takes flow out of the model on average, as
        one upside down does: under the curve alone, the warning is the selected cap's, and a cap
        with a file the wrong way round would show nothing until it was selected. Judged on the
        condition as kept, positive into the model, so that it is the same warning in either
        panel's sign, and the plot's.
        """
        item = qt.QTableWidgetItem()
        condition = self.conditions.get(faceId)
        if self.hasWaveform(faceId):
            item.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable)
            item.setCheckState(qt.Qt.Checked if self.isInverted(condition) else qt.Qt.Unchecked)
            tip = _("Multiply every flow by -1. Tick it when the curve under the table is upside "
                    "down. {direction}").format(direction=intoTheModel(self.inflowSign))
            if self.takesFlowOut(condition):
                item.setIcon(warningIcon())
                tip = f"{backwardsNote(self.inflowSign)}\n{tip}"
            item.setToolTip(tip)
        else:
            item.setFlags(qt.Qt.ItemIsSelectable)
        self.table.setItem(row, INVERT_COLUMN, item)

    @staticmethod
    def takesFlowOut(condition):
        """Whether an inflow, as kept -- positive into the model -- takes flow out on average: what
        a waveform upside down for its panel's solver does."""
        return isinstance(condition, bcs.Inflow) and condition.mean < 0

    def populateValues(self, row, faceId):
        """A box for each of the cap's values, labelled, filled with what it has or was typed."""
        # An item under the boxes, for the row to be selected by and the hint to show on.
        item = self.table.item(row, VALUES_COLUMN)
        if item is None:
            item = qt.QTableWidgetItem()
            self.table.setItem(row, VALUES_COLUMN, item)
        item.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)
        kind = self.kindOf(faceId)
        item.setToolTip(self.hintFor(kind))
        editor = self.table.cellWidget(row, VALUES_COLUMN)
        if kind is None:
            if editor is not None:
                self.table.removeCellWidget(row, VALUES_COLUMN)
                self.table.setRowHeight(row, self.table.verticalHeader().defaultSectionSize)
            return
        if editor is None or editor.property("kind") != kind:
            # Another kind's boxes are another kind's values: nothing carries over between them.
            editor = self.makeValuesEditor(kind)
            self.table.setCellWidget(row, VALUES_COLUMN, editor)
            # As tall as its boxes, which for an RCR are two lines of them, and no taller once
            # the cap is given another kind.
            self.table.setRowHeight(row, max(self.table.verticalHeader().defaultSectionSize,
                                             editor.sizeHint.height()))
        editor.setProperty("faceId", faceId)
        condition = self.conditions.get(faceId)
        if faceId in self.invalid:
            _kind, fields, reason, wrong = self.invalid[faceId]
        else:
            fields, reason, wrong = bcs.fields_of(self.shown(condition)), "", None
        # A waveform has no value to show in a box: it is named there, greyed, described in the
        # box's tooltip and under the plot, and a flow typed over it is a steady one in its place.
        waveform = bcs.to_text(self.shown(condition)) if self.hasWaveform(faceId) else ""
        for box, name, text in zip(self.boxesOf(editor), bcs.VALUE_ORDER[kind],
                                   list(fields) + [""] * len(bcs.VALUE_ORDER[kind])):
            if box.text != text:
                box.text = text
            box.placeholderText = (_("waveform") if name == "Q" and waveform
                                   else "0" if name == "Pd" else "")
            tip = _(VALUE_TIPS[name])
            if name == wrong:
                box.styleSheet = f"QLineEdit {{ color: {INVALID_VALUES_COLOR.name()}; }}"
                box.toolTip = f"{reason}\n{tip}"
            else:
                box.styleSheet = ""
                box.toolTip = f"{waveform}\n{tip}" if name == "Q" and waveform else tip

    def makeValuesEditor(self, kind):
        """A label and a box for each of a kind's values, in `bcs.VALUE_ORDER`'s order, two to a
        line: an RCR's four in one line would make the table too wide for a module panel."""
        editor = qt.QWidget()
        editor.setProperty("kind", kind)
        layout = qt.QGridLayout(editor)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(1)
        for index, name in enumerate(bcs.VALUE_ORDER[kind]):
            label = qt.QLabel(name)
            label.toolTip = _(VALUE_TIPS[name])
            box = qt.QLineEdit()
            box.objectName = f"{VALUE_BOX_PREFIX}{name}"
            # About a dozen characters, which `1.5e-4` and `0.00015` fit in: a line edit asks for
            # seventeen, and two to a line of those leave no room for the flow table's name.
            box.minimumWidth = 56
            box.maximumWidth = 96
            label.setBuddy(box)
            # Read when it is left or Enter is pressed, rather than at every key: a value half typed
            # is not yet wrong.
            box.connect("editingFinished()", lambda editor=editor: self.onValueBoxEdited(editor))
            # A click in a box is a click on its cap: its row selected, so that the hint and the
            # highlight are that cap's. Not while the table fills it in, which moves the cursor too.
            box.connect("cursorPositionChanged(int,int)",
                        lambda _old, _new, editor=editor: self.onValueBoxEntered(editor))
            line, column = divmod(index, 2)
            layout.addWidget(label, line, 2 * column)
            layout.addWidget(box, line, 2 * column + 1)
        # What the column has to spare goes after the boxes, so that each stays beside its label.
        layout.setColumnStretch(4, 1)
        return editor

    @staticmethod
    def boxesOf(editor):
        return [editor.findChild("QLineEdit", f"{VALUE_BOX_PREFIX}{name}")
                for name in bcs.VALUE_ORDER[editor.property("kind")]]

    def valueBoxes(self, faceId):
        """A cap's value boxes, in `bcs.VALUE_ORDER`'s order, or none while it has no kind."""
        row = self.rowOf(faceId)
        editor = self.table.cellWidget(row, VALUES_COLUMN) if row is not None else None
        return self.boxesOf(editor) if editor is not None else []

    def redrawValues(self, faceId):
        row = self.rowOf(faceId)
        if row is None:
            return
        self.updating = True
        try:
            self.populateValues(row, faceId)
            self.populateFlowTable(row, faceId)
            self.populateInvert(row, faceId)
        finally:
            self.updating = False
        self.updatePlot()

    def hintFor(self, kind):
        if kind == bcs.INFLOW:
            return _("Inflow: choose its waveform's table in the Flow table column -- drop a .flow "
                     "file on Slicer to make one -- or type a steady flow in Q, in mL/s. "
                     "{direction} A waveform is drawn below; tick Invert if it is upside "
                     "down.").format(direction=intoTheModel(self.inflowSign))
        if kind == bcs.RCR_KIND:
            return _("RCR: Rp, C, Rd, Pd -- proximal resistance, compliance, distal resistance, "
                     "distal pressure (may be left out, and is then 0).")
        if kind == bcs.RESISTANCE_KIND:
            return _("Resistance: R -- the resistance, draining to zero pressure. An outlet "
                     "that needs a pressure to drain to is an RCR.")
        return _("Choose a kind of condition for the cap first.")

    def updateHint(self):
        selected = self.selectedFaceIds()
        self.hintLabel.text = self.hintFor(self.kindOf(selected[0])) if selected else ""

    # -- editing it ------------------------------------------------------------------------
    def onKindChanged(self, combo, index):
        """A cap given another kind of condition, its boxes empty for the new kind's values.

        Nothing carries over: the boxes are named, and another kind's values are other values --
        an RCR's Rp is not a resistance's R, nor a flow.
        """
        if self.updating:
            return
        faceId = int(combo.property("faceId"))
        kind = KIND_CHOICES[index]
        previous = self.conditions.get(faceId)
        self.invalid.pop(faceId, None)
        if kind != bcs.INFLOW:
            self.unlinkTable(faceId)
        if kind is None:
            self.conditions.pop(faceId, None)
        elif previous is None or previous.kind != kind:
            self.setFromFields(faceId, kind, [])
        self.save()
        self.redrawValues(faceId)
        self.updateHint()
        self.changed()

    def setFromFields(self, faceId, kind, fields):
        """A cap's condition from what was typed in its boxes, in this panel's sign.

        Kept as typed when it does not read, with the reason and the box it is about, so that the
        box can be shown wrong; and when a value it needs is still empty, so that a choice of kind
        with nothing in it yet stays chosen.
        """
        fields = list(fields)
        try:
            condition = bcs.from_fields(kind, fields, self.period())
        except bcs.FieldError as error:
            self.conditions.pop(faceId, None)
            self.invalid[faceId] = (kind, fields, str(error), error.name)
            return
        if condition is None:
            self.conditions.pop(faceId, None)
            missing = bcs.missing_fields(kind, fields)
            reason = (_("Fill in {names}.").format(names=", ".join(missing))
                      if len(missing) < len(bcs.VALUE_ORDER[kind]) else self.hintFor(kind))
            self.invalid[faceId] = (kind, fields, reason, None)
            return
        self.invalid.pop(faceId, None)
        self.conditions[faceId] = self.kept(condition)

    def onValuesEdited(self, row, column):
        """A cell changed: of the cells, only Invert's box is the user's to change."""
        if self.updating or row >= len(self.caps):
            return
        if column == INVERT_COLUMN:
            self.onInvertToggled(row)

    def onValueBoxEdited(self, editor):
        """A value typed in its box, and left or entered: the cap's condition read from its boxes."""
        if self.updating:
            return
        faceId = int(editor.property("faceId"))
        kind = editor.property("kind")
        if self.kindOf(faceId) != kind:
            return
        fields = [box.text for box in self.boxesOf(editor)]
        previous = self.conditions.get(faceId)
        if kind == bcs.INFLOW and not "".join(fields).strip() and self.hasWaveform(faceId):
            # Q left empty over a waveform: the waveform, as it was.
            return
        self.setFromFields(faceId, kind, fields)
        # A flow typed over a table's waveform is the cap's from now on, not the table's. Compared,
        # not taken for the same object: what was shown comes back as a copy, put back in its sign.
        if self.conditions.get(faceId) != previous:
            self.unlinkTable(faceId)
        self.save()
        self.redrawValues(faceId)
        self.changed()

    def onValueBoxEntered(self, editor):
        if self.updating:
            return
        row = self.rowOf(int(editor.property("faceId")))
        if row is not None and self.selectedFaceIds() != [self.caps[row][0]]:
            self.table.selectRow(row)

    def onSelectionChanged(self):
        selected = self.selectedFaceIds()
        if selected and self.mesh is not None and self.mesh.GetMesh() is not None and self.faces:
            highlight(self.mesh.GetMesh(), selected[0], self.faces.arrayName, self.faces.boundary)
        else:
            clearHighlight()
        self.updateHint()
        self.updatePlot()
        self.selectionChanged(selected)

    def onInvertToggled(self, row):
        """An inflow's Invert box ticked or unticked: its flows turned round to match it."""
        faceId = self.caps[row][0]
        condition = self.conditions.get(faceId)
        item = self.table.item(row, INVERT_COLUMN)
        if not self.hasWaveform(faceId) or item is None:
            return
        wanted = item.checkState() == qt.Qt.Checked
        if wanted == self.isInverted(condition):
            return
        self.conditions[faceId] = condition.flipped()
        self.save()
        self.redrawValues(faceId)
        self.changed()
        self.setStatus(self.describeWaveform(faceId))

    def updatePlot(self):
        """The selected cap's waveform under the table, when it has one; nothing otherwise."""
        selected = self.selectedFaceIds()
        if selected and self.hasWaveform(selected[0]):
            condition = self.conditions[selected[0]]
            self.plot.show(self.capName(selected[0]), self.shown(condition), self.inflowSign,
                           self.isInverted(condition))
        else:
            self.plot.hide()

    def describeWaveform(self, faceId):
        """A cap's waveform in a line for the status, as this panel shows it: its points, period
        and mean, whether it is inverted, and what a mean that goes out of the model means."""
        waveform = self.shown(self.conditions[faceId])
        note = _("{name}: {points} points over {period:g} s, mean {mean:.4g} mL/s.").format(
            name=self.capName(faceId), points=len(waveform.time), period=waveform.period,
            mean=waveform.mean)
        if self.isInverted(self.conditions[faceId]):
            note += " " + _("Inverted: every flow multiplied by -1.")
        if self.takesFlowOut(self.conditions[faceId]):
            note += " " + backwardsNote(self.inflowSign)
        return note

    # -- a waveform, from a flow table ------------------------------------------------------
    def onFlowTableChosen(self, faceId, tableNode):
        """A flow table chosen for a cap: its waveform read in, or, for none, the inflow cleared."""
        if self.updating:
            return
        if tableNode is not None and tableNode is self.flowTables.get(faceId):
            # Chosen again: read again, the box left as it was.
            self.readFlowTable(faceId, always=True)
            return
        self.unlinkTable(faceId)
        if tableNode is None:
            self.conditions.pop(faceId, None)
            self.invalid[faceId] = (bcs.INFLOW, [], self.hintFor(bcs.INFLOW), None)
            self.save()
            self.redrawValues(faceId)
            self.changed()
            return
        self.flowTables[faceId] = tableNode
        if self.mesh is not None:
            self.mesh.SetNodeReferenceID(f"{INFLOW_TABLE_ROLE}{faceId}", tableNode.GetID())
        self.observeTable(faceId, tableNode)
        # Unticked as chosen, whatever the table before it was: a box ticked for that one says
        # nothing about this one. Unticked is the table as it is, in this panel's sign, which kept
        # positive into the model is the table multiplied by -1 where the solver takes inflow
        # negative.
        self.readFlowTable(faceId, invert=self.inflowSign < 0, always=True)
        row = self.rowOf(faceId)
        if row is not None:
            # Selected, so that its curve is drawn under the table, where Invert is judged.
            self.table.selectRow(row)

    def onFlowTableShown(self, faceId):
        """A selector showing a table that is not its cap's: its table deleted, the selector moved on.

        Put back to what the cap is read from, which, its table gone, is nothing. The waveform is
        kept, as one from a file is when the file is deleted: it is on the mesh, and a scene tidied
        of its tables would otherwise lose every inflow set up from them. Run once the deletion is
        over, when the node is out of the scene, rather than while the selector is moving on.
        """
        tableNode = self.flowTables.get(faceId)
        scene = tableNode.GetScene() if tableNode is not None else None
        if tableNode is not None and not (scene and scene.IsNodePresent(tableNode)):
            self.unlinkTable(faceId)
            self.setStatus(_("{name}: its flow table was deleted; the waveform read from it is kept.")
                           .format(name=self.capName(faceId)))
        row = self.rowOf(faceId)
        selector = self.table.cellWidget(row, TABLE_COLUMN) if row is not None else None
        if selector is not None and selector.currentNode() is not self.flowTables.get(faceId):
            wasBlocked = selector.blockSignals(True)
            selector.setCurrentNode(self.flowTables.get(faceId))
            selector.blockSignals(wasBlocked)

    def onFlowTableModified(self, faceId, tableNode):
        """A flow table edited, in Slicer's table view or by a script: its cap's inflow read again.

        Into the conditions as the mesh has them now, not as this panel last saw them: both panels
        watch the tables their mesh's inflows are read from, the one not shown among them, and its
        conditions are as they were when it was last shown -- written back whole, they would undo
        whatever the other panel has changed since.
        """
        if self.mesh is None or self.mesh.GetNodeReference(f"{INFLOW_TABLE_ROLE}{faceId}") is not tableNode:
            # Unlinked in the other panel since: no longer this cap's to follow.
            self.unlinkTable(faceId)
            return
        capIds = {capId for capId, _name, _geometry in self.caps}
        self.conditions = {capId: condition for capId, condition in bcs.from_json(
            self.mesh.GetAttribute(BOUNDARY_CONDITIONS_ATTRIBUTE)).items() if capId in capIds}
        self.readFlowTable(faceId)

    def readFlowTable(self, faceId, invert=None, always=False):
        """A cap's inflow read from its table again: inverted as it was, unless `invert` says --
        in the condition's own terms, whether it is kept as the table multiplied by -1.

        Nothing is done when the numbers are the ones already there, unless `always`: a table is
        modified by being renamed, or given an attribute, too, and each would otherwise rewrite the
        status over whatever it said.
        """
        tableNode = self.flowTables.get(faceId)
        if tableNode is None:
            return
        previous = self.conditions.get(faceId)
        if invert is None:
            invert = isinstance(previous, bcs.Inflow) and previous.negated
        try:
            waveform = flowTableWaveform(tableNode)
        except bcs.BoundaryConditionError as error:
            self.conditions.pop(faceId, None)
            self.invalid[faceId] = (bcs.INFLOW, [], str(error), None)
            status = f"{self.capName(faceId)}: {error}"
        else:
            waveform = waveform.flipped() if invert else waveform
            if not always and faceId not in self.invalid and waveform == previous:
                return
            self.conditions[faceId] = waveform
            self.invalid.pop(faceId, None)
            status = self.describeWaveform(faceId)
        self.save()
        self.redrawValues(faceId)
        self.changed()
        # After the panel has had its say about the conditions as a whole, as Invert's is.
        self.setStatus(status)

    def observeTable(self, faceId, tableNode):
        tag = tableNode.AddObserver(
            vtk.vtkCommand.ModifiedEvent,
            lambda _caller, _event, faceId=faceId, tableNode=tableNode: self.onFlowTableModified(faceId, tableNode))
        self.tableObservations[faceId] = (tableNode, tag)

    def unlinkTable(self, faceId):
        """A cap's inflow no longer read from a table: not watched, and not referenced.

        The reference is removed only when it is still to that table: the other panel may have
        given the cap another since.
        """
        observed = self.tableObservations.pop(faceId, None)
        if observed is not None:
            observed[0].RemoveObserver(observed[1])
        tableNode = self.flowTables.pop(faceId, None)
        role = f"{INFLOW_TABLE_ROLE}{faceId}"
        if tableNode is not None and self.mesh is not None and self.mesh.GetNodeReference(role) is tableNode:
            self.mesh.RemoveNodeReferenceIDs(role)

    def unobserveTables(self):
        for tableNode, tag in self.tableObservations.values():
            tableNode.RemoveObserver(tag)
        self.tableObservations = {}
