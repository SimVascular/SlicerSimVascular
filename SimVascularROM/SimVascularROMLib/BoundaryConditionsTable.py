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

import json
import logging
import os

import qt
import slicer
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

COLUMNS = ("Name", "Type", "Values")
NAME_COLUMN = COLUMNS.index("Name")
TYPE_COLUMN = COLUMNS.index("Type")
VALUES_COLUMN = COLUMNS.index("Values")

# What the type column offers, the first meaning none: a cap with no condition is shown as one,
# rather than given a default that looks like a decision.
KIND_CHOICES = (None,) + bcs.KINDS
NO_CONDITION_TEXT = "-"

INVALID_VALUES_COLOR = qt.QColor(200, 60, 40)


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

    The values of an inflow are a waveform, which nobody types: double-clicking them asks for the
    file it is in, and a steady flow is typed by selecting the cell and typing. Anything that
    cannot be read stays on screen in red, with the reason as its tooltip, rather than vanishing
    on a typo; it is not a condition until it reads.
    """

    def __init__(self, table, hintLabel, setStatus, changed=None, selectionChanged=None,
                 startDirectory=lambda: ""):
        self.table = table
        self.hintLabel = hintLabel
        self.setStatus = setStatus
        self.changed = changed or (lambda: None)
        self.selectionChanged = selectionChanged or (lambda faceIds: None)
        self.startDirectory = startDirectory
        self.mesh = None
        self.faces = None
        self.caps = []
        self.conditions = {}
        self.invalid = {}
        # The cap shown in bold, and what its tooltip says about why: the 0D panel's source.
        self.emphasised = None
        self.emphasisNote = ""
        self.updating = False

        table.setColumnCount(len(COLUMNS))
        table.setHorizontalHeaderLabels([_(name) for name in COLUMNS])
        table.verticalHeader().setVisible(False)
        # A waveform shows as its file's path, and a path cut short at its end loses the one part
        # of it that says which file it is. Without word wrap, which the table has on by default
        # and which, a path being full of places to break, elides the end whatever the mode says.
        table.setWordWrap(False)
        table.setTextElideMode(qt.Qt.ElideMiddle)
        table.horizontalHeader().setSectionResizeMode(qt.QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(VALUES_COLUMN, qt.QHeaderView.Stretch)
        # A double-click is this table's to answer rather than Qt's, because it means two things:
        # on an inflow it asks for a waveform file, anywhere else it opens the cell for typing.
        # Typing over a selected cell, which is how a steady inflow is entered, still opens it.
        table.setEditTriggers(qt.QAbstractItemView.EditKeyPressed
                              | qt.QAbstractItemView.AnyKeyPressed)
        table.connect("cellChanged(int,int)", self.onValuesEdited)
        table.connect("itemSelectionChanged()", self.onSelectionChanged)
        table.connect("cellDoubleClicked(int,int)", self.onCellDoubleClicked)

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
        finally:
            self.updating = False
        self.updateHint()

    def populateValues(self, row, faceId):
        item = self.table.item(row, VALUES_COLUMN)
        if item is None:
            item = qt.QTableWidgetItem()
            self.table.setItem(row, VALUES_COLUMN, item)
        if faceId in self.invalid:
            _kind, text, reason = self.invalid[faceId]
            item.setText(text)
            item.setForeground(qt.QBrush(INVALID_VALUES_COLOR))
            item.setToolTip(reason)
        else:
            text = bcs.to_text(self.conditions.get(faceId))
            item.setText(text)
            item.setForeground(qt.QBrush())
            # The text as well as the hint: a waveform's path is often longer than the column.
            hint = self.hintFor(self.kindOf(faceId))
            item.setToolTip(f"{text}\n{hint}" if text else hint)
        editable = self.kindOf(faceId) is not None
        item.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable
                      | (qt.Qt.ItemIsEditable if editable else 0))

    def redrawValues(self, faceId):
        row = self.rowOf(faceId)
        if row is None:
            return
        self.updating = True
        try:
            self.populateValues(row, faceId)
        finally:
            self.updating = False

    @staticmethod
    def hintFor(kind):
        if kind == bcs.INFLOW:
            return _("Inflow: double-click the values to load a waveform from a .flow file, or "
                     "select them and type a steady flow Q in mL/s.")
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
        """A cap given another kind of condition: what was typed is read again as the new kind.

        So that a value that fits the new kind carries over, and one that does not is shown in
        red rather than dropped.
        """
        if self.updating:
            return
        faceId = int(combo.property("faceId"))
        kind = KIND_CHOICES[index]
        row = self.rowOf(faceId)
        previous = self.conditions.get(faceId)
        self.invalid.pop(faceId, None)
        if kind is None:
            self.conditions.pop(faceId, None)
        elif previous is None or previous.kind != kind:
            text = self.table.item(row, VALUES_COLUMN).text() if row is not None else ""
            if isinstance(previous, bcs.Inflow) and not previous.is_steady:
                text = ""
            self.setFromText(faceId, kind, text)
        self.save()
        self.redrawValues(faceId)
        self.updateHint()
        self.changed()

    def setFromText(self, faceId, kind, text):
        try:
            condition = bcs.from_text(kind, text, self.conditions.get(faceId), self.period())
        except bcs.BoundaryConditionError as error:
            self.conditions.pop(faceId, None)
            self.invalid[faceId] = (kind, text, str(error))
            return
        self.invalid.pop(faceId, None)
        if condition is None:
            self.conditions.pop(faceId, None)
            if kind is not None:
                # Chosen but not yet given values: keep the choice visible.
                self.invalid[faceId] = (kind, "", self.hintFor(kind))
        else:
            self.conditions[faceId] = condition

    def onValuesEdited(self, row, column):
        if self.updating or column != VALUES_COLUMN or row >= len(self.caps):
            return
        faceId = self.caps[row][0]
        kind = KIND_CHOICES[self.table.cellWidget(row, TYPE_COLUMN).currentIndex]
        if kind is None:
            return
        self.setFromText(faceId, kind, self.table.item(row, column).text())
        self.save()
        self.redrawValues(faceId)
        self.changed()

    def onSelectionChanged(self):
        selected = self.selectedFaceIds()
        if selected and self.mesh is not None and self.mesh.GetMesh() is not None and self.faces:
            highlight(self.mesh.GetMesh(), selected[0], self.faces.arrayName, self.faces.boundary)
        else:
            clearHighlight()
        self.updateHint()
        self.selectionChanged(selected)

    # -- a waveform, by double-clicking an inflow's values ---------------------------------
    def onCellDoubleClicked(self, row, column):
        """Ask for a waveform file on an inflow's values; open any other values for typing."""
        if column != VALUES_COLUMN or row >= len(self.caps):
            return
        faceId = self.caps[row][0]
        kind = self.kindOf(faceId)
        if kind == bcs.INFLOW:
            path = self.chooseWaveformFile(faceId)
            if not path:
                return
            with slicer.util.tryWithErrorDisplay(_("Could not read the waveform."), waitCursor=True):
                self.setStatus(self.loadWaveform(path, faceId))
        elif kind is not None:
            self.table.editItem(self.table.item(row, column))

    def chooseWaveformFile(self, faceId):
        """The file a cap's waveform is to be read from, or "" if none was chosen.

        Opened where the last waveform came from, since a case's waveforms are usually kept
        together, and a Fontan has seven of them to load one after another.
        """
        directory = qt.QSettings().value(WAVEFORM_DIRECTORY_SETTING, "") or self.startDirectory()
        path = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(), _("Inflow waveform for {name}").format(name=self.capName(faceId)),
            directory, _("Flow files (*.flow *.dat *.txt *.csv);;All files (*)"))
        if path:
            qt.QSettings().setValue(WAVEFORM_DIRECTORY_SETTING, os.path.dirname(path))
        return path

    def loadWaveform(self, path, faceId):
        """Give a cap the waveform in a file, saying so if it had to be turned round."""
        waveform, turned = bcs.as_inflow(bcs.read_flow_file(path))
        self.conditions[faceId] = waveform
        self.invalid.pop(faceId, None)
        self.save()
        self.populate()
        self.changed()
        note = _("{name}: {points} points over {period:g} s, mean {mean:.4g} mL/s.").format(
            name=self.capName(faceId), points=len(waveform.time), period=waveform.period,
            mean=waveform.mean)
        if turned:
            note += " " + _("Its flows were negative, as svMultiPhysics writes an inflow, and have "
                            "been turned round: here an inflow is positive.")
        return note
