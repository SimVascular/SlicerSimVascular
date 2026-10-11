"""The 0D network drawn as a directed graph, in a Qt view the panel docks beside Slicer's views.

A Qt graphics view rather than a web page or an application of its own: Slicer's Qt has all a
graph of a few dozen blocks needs, so nothing is installed for this, and being inside the
application is what lets a click on a block plot it in Slicer's own plot view and show it on the
anatomy, which a separate application cannot do.

The graph and its layout are `svromsetup.network`'s, where they are tested without Slicer; this
draws them, colours them as it is told, and says which block was clicked.
"""

import qt
from slicer.i18n import tr as _

from svromsetup import network

# Scene units, which the view starts at one to one with screen pixels.
ROW_SPACING = 40.0
LAYER_GAP = 22.0
PADDING_X = 8.0
MINIMUM_WIDTH = {network.VESSEL: 34.0, network.BOUNDARY: 0.0}
MINIMUM_HEIGHT = {network.VESSEL: 22.0, network.BOUNDARY: 32.0}
CORNER_RADIUS = {network.VESSEL: 11.0, network.BOUNDARY: 3.0}
# A press and release this close together, in pixels, is a click; further apart, a drag.
CLICK_TOLERANCE = 4
JUNCTION_RADIUS = 5.5
ARROW_LENGTH = 8.0
FONT_POINT_SIZE = 8.5
ZOOM_STEP = 1.15
# Fitting a tall network to a short view would shrink its labels past reading; past this, the
# view keeps the scale and the rest is scrolled to.
SMALLEST_FIT_SCALE = 0.35

NO_RESULT_COLOR = qt.QColor(226, 229, 235)
JUNCTION_COLOR = qt.QColor(90, 95, 105)
EDGE_COLOR = qt.QColor(125, 130, 140)
OUTLINE_COLOR = qt.QColor(70, 75, 85)
BACKGROUND_COLOR = qt.QColor(250, 250, 252)
# Mesh Prep's face highlight, so that a selected block and the cap it is on look selected the
# same way -- as a halo behind the block rather than its outline, which on the yellows and
# greens of the pressure colours would not be seen; the outline goes dark instead.
SELECTED_COLOR = qt.QColor(255, 255, 0)
SELECTED_OUTLINE_COLOR = qt.QColor(25, 25, 35)
HALO_MARGIN = 5.0


class ZoomingGraphicsView(qt.QGraphicsView):
    """A graphics view that zooms on the wheel, pans on a drag, and says which block is clicked.

    A Fontan's network is fifty blocks tall: without zooming it is too small to read or too big
    to see. Clicks are found here, by what is under the mouse, rather than through Qt's item
    selection, which outlines a selected item with a dashed box of its own over the one this draws.
    """

    clicked = None
    """Called with the item clicked, or None for the background."""

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if steps:
            factor = ZOOM_STEP ** steps
            self.scale(factor, factor)

    def mousePressEvent(self, event):
        self._pressedAt = event.pos()
        qt.QGraphicsView.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        qt.QGraphicsView.mouseReleaseEvent(self, event)
        pressed = getattr(self, "_pressedAt", None)
        self._pressedAt = None
        if pressed is None or event.button() != qt.Qt.LeftButton:
            return
        moved = event.pos() - pressed
        if abs(moved.x()) + abs(moved.y()) <= CLICK_TOLERANCE and self.clicked is not None:
            self.clicked(self.itemAt(event.pos()))


class NetworkView:
    """A graph of the network, a line saying what its colours are, and one about the selection.

    :param nodeSelected: called with the key of the block clicked, or None when nothing is.
    """

    def __init__(self, nodeSelected):
        self.nodeSelected = nodeSelected
        self.widget = qt.QWidget()
        layout = qt.QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)

        top = qt.QHBoxLayout()
        self.legendLabel = qt.QLabel()
        self.legendLabel.wordWrap = True
        top.addWidget(self.legendLabel, 1)
        self.fitButton = qt.QPushButton(_("Fit"))
        self.fitButton.toolTip = _("Show the whole network.")
        self.fitButton.connect("clicked(bool)", lambda _checked: self.fit())
        top.addWidget(self.fitButton)
        layout.addLayout(top)

        self.scene = qt.QGraphicsScene()
        self.view = ZoomingGraphicsView()
        self.view.setScene(self.scene)
        self.view.setRenderHint(qt.QPainter.Antialiasing, True)
        self.view.setRenderHint(qt.QPainter.TextAntialiasing, True)
        # Dragging the background pans; a click on a block still selects it.
        self.view.setDragMode(qt.QGraphicsView.ScrollHandDrag)
        self.view.setTransformationAnchor(qt.QGraphicsView.AnchorUnderMouse)
        self.view.setBackgroundBrush(qt.QBrush(BACKGROUND_COLOR))
        self.view.clicked = self.onItemClicked
        layout.addWidget(self.view, 1)

        self.infoLabel = qt.QLabel()
        self.infoLabel.wordWrap = True
        self.infoLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        layout.addWidget(self.infoLabel)

        self.items = {}
        self._selected = None
        self._halo = None

    # -- drawing ---------------------------------------------------------------------------
    def clear(self, message=""):
        self.items = {}
        self._selected = None
        self._halo = None
        self.scene.clear()
        self.legendLabel.text = message
        self.infoLabel.text = ""

    def setNetwork(self, net, colours=None, tooltips=None, legend=""):
        """Draw `net`: `colours` is {key: (r, g, b)} in 0..1, `tooltips` {key: text}."""
        colours, tooltips = colours or {}, tooltips or {}
        self.clear(legend)
        font = qt.QFont()
        font.setPointSizeF(FONT_POINT_SIZE)
        positions = network.layered_layout(net)
        texts, sizes = {}, {}
        for key, node in net.nodes.items():
            if node.kind == network.JUNCTION:
                sizes[key] = (2 * JUNCTION_RADIUS, 2 * JUNCTION_RADIUS)
                continue
            text = qt.QGraphicsSimpleTextItem(
                node.short if node.kind == network.VESSEL else f"{node.label}\n{node.detail}")
            text.setFont(font)
            bounds = text.boundingRect()
            texts[key] = text
            sizes[key] = (max(bounds.width() + 2 * PADDING_X, MINIMUM_WIDTH[node.kind]),
                          max(bounds.height() + 8.0, MINIMUM_HEIGHT[node.kind]))

        # Each layer as wide as its widest block, so that a long cap name pushes the layers
        # after it along rather than running into them.
        layerCount = 1 + max((layer for layer, _row in positions.values()), default=0)
        widths = [0.0] * layerCount
        for key, (layer, _row) in positions.items():
            widths[layer] = max(widths[layer], sizes[key][0])
        starts, cursor = [], 0.0
        for width in widths:
            starts.append(cursor)
            cursor += width + LAYER_GAP
        # A layer of conditions alone -- the caps at the ends, which the layout lines up last -- is
        # aligned on its left, so that every arrow into it ends at one edge whatever the length of
        # the cap's name; any other layer is centred, which keeps a branching symmetrical.
        conditionsOnly = [all(net.nodes[key].kind == network.BOUNDARY
                              for key, (at, _row) in positions.items() if at == layer)
                          for layer in range(layerCount)]
        centres = {}
        for key, (layer, row) in positions.items():
            left = starts[layer] if conditionsOnly[layer] else starts[layer] + (widths[layer] - sizes[key][0]) / 2.0
            centres[key] = (left + sizes[key][0] / 2.0, row * ROW_SPACING)

        for source, target in net.edges:
            self._addEdge(centres[source], sizes[source], centres[target], sizes[target])
        for key, node in net.nodes.items():
            self._addNode(key, node, centres[key], sizes[key], texts.get(key), colours.get(key),
                          tooltips.get(key, ""))
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-30, -30, 30, 30))
        self.fit()

    def _addEdge(self, sourceCentre, sourceSize, targetCentre, targetSize):
        start = qt.QPointF(sourceCentre[0] + sourceSize[0] / 2.0, sourceCentre[1])
        end = qt.QPointF(targetCentre[0] - targetSize[0] / 2.0, targetCentre[1])
        bend = (end.x() - start.x()) / 2.0
        path = qt.QPainterPath(start)
        path.cubicTo(qt.QPointF(start.x() + bend, start.y()),
                     qt.QPointF(end.x() - bend, end.y()), qt.QPointF(end.x() - ARROW_LENGTH, end.y()))
        curve = self.scene.addPath(path, qt.QPen(EDGE_COLOR, 1.2))
        arrow = self.scene.addPolygon(
            qt.QPolygonF([end, qt.QPointF(end.x() - ARROW_LENGTH, end.y() - ARROW_LENGTH / 2.0),
                          qt.QPointF(end.x() - ARROW_LENGTH, end.y() + ARROW_LENGTH / 2.0)]),
            qt.QPen(qt.Qt.NoPen), qt.QBrush(EDGE_COLOR))
        for item in (curve, arrow):
            item.setZValue(0)

    def _addNode(self, key, node, centre, size, text, colour, tooltip):
        width, height = size
        rectangle = qt.QRectF(centre[0] - width / 2.0, centre[1] - height / 2.0, width, height)
        fill = qt.QColor.fromRgbF(*colour) if colour is not None else NO_RESULT_COLOR
        if node.kind == network.JUNCTION:
            item = self.scene.addEllipse(rectangle, qt.QPen(OUTLINE_COLOR, 1.0), qt.QBrush(JUNCTION_COLOR))
        else:
            path = qt.QPainterPath()
            radius = CORNER_RADIUS[node.kind]
            path.addRoundedRect(rectangle, radius, radius)
            item = self.scene.addPath(path, self._pen(node, False), qt.QBrush(fill))
        item.setData(0, key)
        item.setToolTip(tooltip)
        item.setZValue(1)
        if text is not None:
            text.setParentItem(item)
            bounds = text.boundingRect()
            text.setPos(centre[0] - bounds.width() / 2.0, centre[1] - bounds.height() / 2.0)
            # Dark text on a light fill and light on a dark one: the colour map runs from dark
            # blue to yellow, and black on its blue end cannot be read.
            light = 0.299 * fill.redF() + 0.587 * fill.greenF() + 0.114 * fill.blueF() > 0.5
            text.setBrush(qt.QBrush(qt.QColor(20, 20, 25) if light else qt.QColor(250, 250, 250)))
            # A click on the label is a click on the block.
            text.setAcceptedMouseButtons(qt.Qt.NoButton)
        self.items[key] = item
        item.setData(1, node.kind)

    @staticmethod
    def _pen(node_or_kind, selected):
        kind = node_or_kind if isinstance(node_or_kind, str) else node_or_kind.kind
        if selected:
            return qt.QPen(SELECTED_OUTLINE_COLOR, 2.2)
        return qt.QPen(OUTLINE_COLOR, 1.6 if kind == network.BOUNDARY else 1.0)

    # -- the view and the selection --------------------------------------------------------
    def fit(self):
        """Show the whole network, unless that would make it too small to read."""
        bounds = self.scene.itemsBoundingRect().adjusted(-20, -20, 20, 20)
        viewport = self.view.viewport().rect
        if bounds.isEmpty() or viewport.width() < 10 or viewport.height() < 10:
            return
        scale = min(viewport.width() / bounds.width(), viewport.height() / bounds.height(), 1.0)
        self.view.resetTransform()
        self.view.scale(max(scale, SMALLEST_FIT_SCALE), max(scale, SMALLEST_FIT_SCALE))
        self.view.centerOn(bounds.center() if scale >= SMALLEST_FIT_SCALE
                           else qt.QPointF(bounds.center().x(), bounds.top()))

    def selectedKey(self):
        return self._selected

    def select(self, key, info=None):
        """Show `key` selected, as a click would, without saying so to whoever asked for it."""
        self._selected = key if key in self.items else None
        self._restyle()
        if self._selected is not None:
            self.view.ensureVisible(self.items[self._selected], 40, 40)
        if info is not None:
            self.infoLabel.text = info

    def click(self, key):
        """Select `key` and say so, as clicking its block does."""
        self.select(key)
        self.nodeSelected(self._selected)

    def setInfo(self, text):
        self.infoLabel.text = text or ""

    def onItemClicked(self, item):
        # The block, if what was clicked is its label.
        while item is not None and item.data(0) is None:
            item = item.parentItem()
        self.click(item.data(0) if item is not None else None)

    def _restyle(self):
        for key, item in self.items.items():
            kind, selected = item.data(1), key == self._selected
            item.setPen(qt.QPen(SELECTED_OUTLINE_COLOR if selected else OUTLINE_COLOR, 1.0)
                        if kind == network.JUNCTION else self._pen(kind, selected))
            # Above its neighbours while selected, so that its halo is not drawn under theirs.
            item.setZValue(3 if selected else 1)
        if self._halo is not None:
            self.scene.removeItem(self._halo)
            self._halo = None
        if self._selected is not None:
            bounds = self.items[self._selected].sceneBoundingRect().adjusted(
                -HALO_MARGIN, -HALO_MARGIN, HALO_MARGIN, HALO_MARGIN)
            path = qt.QPainterPath()
            path.addRoundedRect(bounds, HALO_MARGIN + 4.0, HALO_MARGIN + 4.0)
            self._halo = self.scene.addPath(path, qt.QPen(qt.Qt.NoPen), qt.QBrush(SELECTED_COLOR))
            self._halo.setZValue(2)
