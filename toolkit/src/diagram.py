# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:44:41 2019

@author: abastillasgl

Define classes for drawing (SVG) diagrams from specifications from CSV files.

Parent class show contain just the x, y, width, and height parameters.

Children classes contain connection, label, and visual attributes.
"""

import svgwrite as svg
import pandas as pd
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex
from collections import namedtuple
from string import ascii_letters as alphabet, digits as numbers
from configuration import Configuration
from itertools import filterfalse, dropwhile

config = Configuration("configuration.yaml")
SHAPES = config.shapes
STYLES = config.styles


class Shape(object):
    """
    Parent Class for all diagram objects

    Contains attributes for position and size and methods to connect to other
    shapes
    """
    TOP, RIGHT, BOTTOM, LEFT = range(4)
    COORDINATES = namedtuple('Coordinates', list('xy'))

    def __init__(self, **kwargs):

        # Identifiers
        self._uid = self.__name()
        self._name = self.__name(kwargs.get('name', self._uid))
        self._type = kwargs.get('type', None)

        # Locational Properties
        self._x = max(kwargs.get("x", 0), 0)  # X coordinate from top left
        self._y = max(kwargs.get("y", 0), 0)  # Y coordinate from top left
        self._w = max(kwargs.get("w", 0), 0)  # Width of shape
        self._h = max(kwargs.get("h", 0), 0)  # Height of shape
        self._r = max(kwargs.get("r", 0), 0)  # Radius of shape

        if kwargs.get("insert", None):
            self._x, self._y = kwargs.get('insert', (0, 0))

        if kwargs.get("size", None):
            self._w, self._h = kwargs.get('size', (1, 1))

        # Normalize Measurements not in User Units (pixels(?))
        measurements = [self._x, self._y, self._w, self._h, self._r]

        for i, measurement in enumerate(measurements):
            measurement_string = str(measurement).lower()

            if 'mm' in measurement_string:
                mm = measurement.replace("mm", "").strip()
                mm = self.mm_to_pixels(float(mm))
                measurements[i] = mm
                continue

            elif 'cm' in measurement_string:
                cm = measurement.replace("cm", "").strip()
                cm = self.cm_to_pixels(float(cm))
                measurements[i] = cm
                continue

            elif 'in' in measurement_string:
                in_ = measurement.replace("in", "").strip()
                in_ = self.cm_to_pixels(float(in_))
                measurements[i] = in_
                continue

            else:
                pass

        self._x, self._y, self._w, self._h, self._r = measurements

        # Geometric Properties
        self._padding = kwargs.get("padding", 5)
        self._shape = kwargs.get("svg_shape", "rect")  # Name of SVG shape

        # Visual Properties
        self._stroke = kwargs.get("stroke", "#000000")
        self._stroke_width = kwargs.get("stroke-width", 1)
        self._fill = kwargs.get("fill", "#ffffff")
        self._alpha = kwargs.get("alpha", 1)

        # Font Properties
        self._font_family = kwargs.get("font_family", "arial")
        self._font_size = kwargs.get("font_size", 12)
        self._font_color = kwargs.get("font_color", "black")
        self._font_weight = kwargs.get("font_weight")
        self._text_anchor = kwargs.get("text_anchor", "middle")
        self._alignment_baseline = kwargs.get("alignment_baseline", "central")

        # Layer Properties
        self._layer = 0
        self._origin = kwargs.get("origin", None)

        # Top, Right, Bottom, Left xy-tuples for connections
        self._connections = kwargs.get("connections", dict())

        self._top, self._right, self._bottom, self._left = [None] * 4
        self._top_left, self._top_right = [None] * 2
        self._bottom_right, self._bottom_left = [None] * 2

        self._sides = None
        self._corners = None

        self.refresh_shape()

    def __repr__(self):
        s = (f"Shape(name='{self._name}', x={self.x}, y={self.y}, "
             f"width={self.w}, height={self.h})")
        return s

    def __name(self, name="Shape", n=17):
        defaultname = 'Shape'
        if name == defaultname:
            a = np.array(list(alphabet + numbers))
            indices = np.random.randint(0, a.size - 1, n)
            name = f"{''.join(a[indices])}"
        return name

    @property
    def uid(self):
        return self._uid

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def w(self):
        return self._w

    @property
    def h(self):
        return self._h

    @x.setter
    def x(self, x):
        self._x = x
        self.refresh_shape()

    @y.setter
    def y(self, y):
        self._y = y
        self.refresh_shape()

    @w.setter
    def w(self, w):
        self._w = w
        self.refresh_shape()

    @h.setter
    def h(self, h):
        self._h = h
        self.refresh_shape()

    @property
    def r(self):
        return self._r

    @property
    def shape(self):
        return self._x, self._y, self._w, self._h

    @property
    def connections(self):
        return self._connections

    @connections.setter
    def connections(self, value):
        self._connections = value

    @property
    def padding(self):
        return self._padding

    @property
    def top(self):
        return self._top

    @property
    def right(self):
        return self._right

    @property
    def bottom(self):
        return self._bottom

    @property
    def left(self):
        return self._left

    @property
    def top_left(self):
        return self._top_left

    @property
    def top_right(self):
        return self._top_right

    @property
    def bottom_right(self):
        return self._bottom_right

    @property
    def bottom_left(self):
        return self._bottom_left

    @property
    def corners(self):
        return self._corners

    @property
    def center(self):
        return self.COORDINATES(x=self._top.x, y=self._right.y)

    @property
    def sides(self):
        return self._sides

    @property
    def stroke(self):
        return self._stroke

    @stroke.setter
    def stroke(self, color):
        self._stroke = color

    @property
    def stroke_width(self):
        return self._stroke_width

    @stroke_width.setter
    def stroke_width(self, stroke_width):
        self._stroke_width = stroke_width

    @property
    def fill(self):
        return self._fill

    @fill.setter
    def fill(self, color):
        self._fill = color

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, opacity):
        self._alpha = opacity

    @property
    def font_family(self):
        return self._font_family

    @font_family.setter
    def font_family(self, font):
        self._font_family = font

    @property
    def font_size(self):
        return self._font_size

    @font_size.setter
    def font_size(self, size):
        self._font_size = size

    @property
    def font_color(self):
        return self._font_color

    @font_color.setter
    def font_color(self, color):
        self._font_color = color

    @property
    def font_weight(self):
        return self._font_weight

    @font_weight.setter
    def font_weight(self, weight):
        self._font_weight = weight

    @property
    def text_anchor(self):
        return self._text_anchor

    @text_anchor.setter
    def text_anchor(self, anchor):
        self._text_anchor = anchor

    @property
    def alignment_baseline(self):
        return self._alignment_baseline

    @alignment_baseline.setter
    def alignment_baseline(self, alignment):
        self._alignment_baseline = alignment

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, layer):
        self._layer = layer

    @property
    def origin(self):
        return self._origin

    @property
    def insert(self):
        return tuple(self._top_left)

    @property
    def size(self):
        return self._w, self._h

    def percent_to_pixels(self, percent, reference=None):
        """
        Convert a percent to pixels using a reference

        Parameters
        ----------
            percent (float): Percentage of pixels to return
            reference (int): Maximum number of pixels at 100%.

        Notes
        -----
            Default reference is Document's width
        """
        if reference is not None:
            reference = self._w

        return percent * reference

    def pixels_to_percent(self, pixels, reference=None):
        """
        Convert a percent to pixels using a reference

        Parameters
        ----------
            pixels (int): Pixels to get percentage for
            reference (int): Maximum number of pixels at 100%.

        Notes
        -----
            Default reference is Document's width
        """
        if reference is not None:
            reference = self._w

        return pixels / reference

    def inch_to_pixels(self, inches):
        """ Convert inches to pixels using (1 in = 90 px) definition """
        return inches * 90

    def pixels_to_inches(self, pixels):
        """ Convert pixels to inches using (1 in = 90 px) definition """
        return pixels / 90

    def mm_to_pixels(self, mm):
        """ Convert millimeters to pixels using (1 mm = 3.54337 px) definition """
        return mm * 3.543307

    def pixels_to_mm(self, pixels):
        """ Convert pixels to millimeters using (1 mm = 3.54337 px) definition """
        return pixels / 3.543307

    def cm_to_pixels(self, cm):
        """ Convert inches to pixels using (1 cm = 35.4337 px) definition """
        return cm * 35.43307

    def pixels_to_cm(self, pixels):
        """ Convert pixels to inches using (1 cm = 35.4337 px) definition """
        return pixels / 35.43307

    def refresh_shape(self):
        """
        Redfined sides and corners for this Shape object.
        """
        self._top = self.side('top')
        self._right = self.side('right')
        self._bottom = self.side('bottom')
        self._left = self.side('left')

        self._top_left = self.corner('top')
        self._top_right = self.corner('right')
        self._bottom_right = self.corner('bottom')
        self._bottom_left = self.corner('left')

        self._sides = [self._top,
                       self._right,
                       self._bottom,
                       self._left]
        self._corners = [self._top_left,
                         self._top_right,
                         self._bottom_right,
                         self._bottom_left]

    def corner(self, corner, pad=False):
        """
        Get the vertex point of the corners of this shape

        Parameters
        ----------
            corner (int, str): Corner of this shape: top, right, left, bottom

        Notes
        -----
            'top' returns the top-left vertex
            'right' returns the top-right vertex
            'bottom' returns the bottom-right vertex
            'left' returns the bottom-left vertext

        Returns
        -------
            Tuple of the side specified (x, y)
        """
        padding = 0
        if pad:
            padding = self._padding
        corners = {'0': (self.x - padding, self.y - padding),
                   '1': (self.x + self.w + padding, self.y - padding),
                   '2': (self.x + self.w + padding, self.y + self.h + padding),
                   '3': (self.x - padding, self.y + self.h + padding)}

        corner_labels1 = ['top', 'right', 'bottom', 'left']
        corner_labels2 = ['topleft', 'topright', 'bottomright', 'bottomleft']

        corner = corner.replace(" ", "").replace("-", "")
        corner = corner.replace("_", "").lower()

        if corner in corner_labels1:
            corner = corner_labels1.index(corner)

        elif corner in corner_labels2:
            corner = corner_labels2.index(corner)

        if isinstance(corner, int) and corner >= 0 and corner <= 3:
            coordinates = corners[f"{corner}"]
            return self.COORDINATES(*coordinates)
        else:
            raise ValueError(f"Corner must be {corner_labels2}")

    def side(self, side, pad=False):
        """
        Get the center point of a specified side of this shape

        Parameters
        ----------
            side (int, str): Side of this shape: top, right, left, bottom

        Returns
        -------
            Tuple of the side specified (x, y)
        """
        padding = 0
        if pad:
            padding = self._padding

        sides = {'0': (self.x + self.w / 2,
                       self.y - padding),
                 '1': (self.x + self.w + padding,
                       self.y + self.h / 2),
                 '2': (self.x + self.w / 2,
                       self.y + self.h + padding),
                 '3': (self.x - padding,
                       self.y + self.h / 2)}

        side_labels = ['top', 'right', 'bottom', 'left']

        if side.lower() in side_labels:
            side = side_labels.index(side)

        if isinstance(side, int) and side >= 0 and side <= 3:
            coordinates = sides[f"{side}"]
            return self.COORDINATES(*coordinates)
        else:
            raise ValueError(f"Side must be {0,1,2,3} ({side_labels})")

    def connect(self, shape, pad=True):
        """
        Get the connection point coordinates ((x1, y1), (x2, y2)) between this
        shape and another shape.

        Parameters
        ----------
            shape (Shape): Comparison shape to connect
            pad (bool): Include padding (default = 5 pixels).

        Notes
        -----
            If the comparison shape overlaps this shape, it will be assigned
            new coordinates to "push it off" this shape. Default direction is
            push the parameter shape right if it overlaps exactly.

        Returns
        -------
            Tuple of tuples with beginning and ending coordinates of a line.
        """
        padding = self._padding if pad else 0

        # Conditions
        shape_above = shape.bottom.y <= self.top.y - padding
        shape_right = shape.left.x >= self.right.x + padding
        shape_below = shape.top.y >= self.bottom.y + padding
        shape_left = shape.right.x <= self.left.x - padding

        if shape_right:
            start, end = self.right, shape.left

            if shape_above:
                end = shape.bottom

            elif shape_below:
                end = shape.top

            # Old idea was to indicate if connection present
            # self._connections[self.RIGHT] = True
            # New idea is to add connection name and coordinates of connection
            self._connections[shape.uid] = (start, end)
            shape.connections[self._uid] = (end, start)
            return start, end

        if shape_left:
            start, end = self.left, shape.right

            if shape_above:
                end = shape.bottom

            elif shape_below:
                end = shape.top

            # Old idea was to indicate if connection present
            # self._connections[self.RIGHT] = True
            # New idea is to add connection name and coordinates of connection
            self._connections[shape.uid] = (start, end)
            shape.connections[self._uid] = (end, start)
            return start, end

        shape_right = shape.left.x >= self.top.x
        shape_left = shape.right.x <= self.top.x

        if shape_above:
            start, end = self.top, shape.bottom

            if self.top.y - padding * 3 > shape.bottom.y:
                if shape_right:
                    end = shape.left

                elif shape_left:
                    end = shape.right

            # self._connections[self.TOP] = True
            self._connections[shape.uid] = (start, end)
            shape.connections[self._uid] = (end, start)
            return start, end

        if shape_below:
            start, end = self.bottom, shape.top

            if self.bottom.y + padding * 3 > shape.top.y:
                if shape_right:
                    end = shape.left

                elif shape_left:
                    end = shape.right

            # self._connections[self.BOTTOM] = True
            self._connections[shape.uid] = (start, end)
            shape.connections[self._uid] = (end, start)
            return start, end

        m = "Shapes are overlapping. Default connection (right to left)."
        r = "For clean connections, it is recommended redefine the shapes."
        print(Warning(f"Warning: {m}\n{r}"))
        return self.right, shape.left

    def overlap(self, shape, side):
        """
        Detect any lateral overlap between this shape and the specified shape.

        Parameters
        ----------
            shape (Shape): Comparison shape to check
            side (str): Area of shape to check for overlap.

        Notes
        -----
            `side` can take on the following values:
                top, left, right, bottom --> Check if shape overlaps on a half
                q1, q2, q3, q4 --> Check if shape overlaps on a quadrant

            quadrants defined by:

                -----------
                | II |  I |
                -----------
                | III| IV |
                -----------

            Because this function looks at lateral overlap, it can return True
            for overlap on one side of the shape, but False for the other side.

        Returns
        -------
            boolean. True if shapes overlap, otherwise False.
        """
        if not self.overlaps(shape):
            return False

        w, x, y, z = shape.corners

        vline = self._top.x
        hline = self._right.y

        top = (w.y <= hline) or (z.y <= hline)
        right = (w.x >= vline) or (y.x >= vline)
        bottom = (w.y > hline) or (z.y > hline)
        left = (w.x < vline) or (y.x < vline)

        q1 = top and right
        q2 = top and left
        q3 = bottom and left
        q4 = bottom and right

        general = q1 and q2 and q3 and q4

        odict = dict(top=top, right=right, bottom=bottom, left=left,
                     q1=q1, q2=q2, q3=q3, q4=q4, general=general)
        return odict.get(side.lower(), False)

    def overlaps(self, shape):
        """
        Detect any overlap between this shape and the specified shape.

        Parameters
        ----------
            shape (Shape): Comparison shape to check

        Returns
        -------
            boolean. True if shapes overlap, otherwise False.
        """
        if self._corners == shape.corners:
            return True

        if self.vertical_overlap(shape) and self.horizontal_overlap(shape):
            return True
        return False

    def is_left_to(self, shape):
        """ Return True if shape is to right of this shape """
        return shape.left.x >= self._right.x

    def is_right_to(self, shape):
        """ Return True if shape is to left of this shape """
        return shape.right.x <= self._left.x

    def is_bottom_to(self, shape):
        """ Return True if shape is above this shape """
        return shape.bottom.y <= self._top.y

    def is_top_to(self, shape):
        """ Return True if shape is to left of this shape """
        return shape.top.y >= self._bottom.y

    def horizontal_overlap(self, shape):
        """ Detect any horizontal overlap """
        not_left = not self.is_left_to(shape)
        not_right = not self.is_right_to(shape)
        return not_left and not_right

    def vertical_overlap(self, shape):
        """ Detect any vertical overlap """
        not_top = not self.is_top_to(shape)
        not_bottom = not self.is_bottom_to(shape)
        return not_top and not_bottom

    def contains(self, shape):
        """
        Determine if the specified shape is inside or perfectly overlaps

        Parameters
        ----------
            shape (Shape): Comparison shape to check

        Returns
        -------
            boolean. True if shapes overlap, otherwise False.
        """
        if not self.overlaps(shape):
            return False

        if self.corners == shape.corners:
            return True

        bound_v1 = self._x <= shape.left.x <= self._right.x
        bound_v2 = self._x <= shape.right.x <= self._right.x
        bound_h1 = self._y <= shape.top.y <= self._bottom.y
        bound_h2 = self._y <= shape.bottom.y <= self._bottom.y

        bounded_vertically = bound_v1 and bound_v2
        bounded_horizontally = bound_h1 and bound_h2

        if bounded_vertically and bounded_horizontally:
            return True
        return False

    def nudge(self, shape):
        """
        Move a shape's coordinates if it overlaps with this shape.

        Parameters
        ----------
            shape (Shape): Comparison shape to check

        Returns
        -------
            Shape object

        Notes
        -----
            Default behavior is to nudge the input shape to the right
        """
        if not self.overlaps(shape):
            return shape

        shape.x = shape.w + self._x + self._w + self._padding

    def locate(self, shape):
        """
        Return an indicator for where the input shape is located

        Parameters
        ----------
            shape (Shape): Comparison shape to check

        Returns
        -------
            Directional indicator 'top', 'right', 'bottom', or 'left'
        """

        if self._corners == shape.corners:
            return f"direct overlap".strip()

        if self.contains(shape):
            return "inside"

        if shape.contains(self):
            return "surrounded"

        location = ""
        if self.overlaps(shape):
            location = "overlapping "

        if not self.is_top_to(shape):
            location = f"{location}top "
        elif not self.is_bottom_to(shape):
            location = f"{location}bottom "

        if not self.is_left_to(shape):
            location = f"{location}left "
        elif not self.is_right_to(shape):
            location = f"{location}right "

        return location.strip()
#
#    def connect_to(self, shape):
#        """
#        Connect this data source to another shape element
#
#        Parameters
#        ----------
#            shape (Shape): Comparison shape to check
#        """
#        start, end = self.connect(shape)
#        self.connections.update({shape.name: (start, end)})

    def get_connection(self, shape):
        """
        Get connection coordinate data for a shape

        Parameters
        ----------
            shape (Shape): Comparison shape to check
        """
        return self.connections.get(shape.uid, None)

    def distance(self, shape, method='euclidean'):
        """
        Return the distance in pixels between this shape and another shape

        Parameters
        ----------
            shape (Shape): Comparison shape to check
        """
        x1, y1 = self.top.x, self.right.y
        x2, y2 = shape.top.x, shape.right.y
        if method.lower() == 'euclidean':
            squares = (x2 - x1)**2 + (y2 - y2)**2
            return np.sqrt(squares)
        elif method.lower() == 'x':
            return abs(x2 - x1)
        elif method.lower() == 'y':
            return abs(y2 - y1)
        else:
            return abs(x2 - x1) + abs(y2 - y1)

    def draw(self):
        """
        Return an svg.shapes object representing this Shape

        Notes
        -----
            Default shape is svg.shapes.Rect

            Other kinds can be specified by using the following strings:
                1. 'text': Return a text label
                2. 'line': Return a list of line SVG shapes
        """
        kwargs = dict(insert=self.insert,
                      size=self.size,
                      fill=self._fill,
                      stroke=self._stroke,
                      stroke_width=self.stroke_width)

        return svg.shapes.Rect(**kwargs)

    def text(self, font_family='arial', font_size=18, text_anchor='middle'):
        """
        Return an svg.text.Text object for this Shape's name property

        Parameters
        ----------
            font_family (str): Font to use (<svg font-family>)
            font_size (str): Size of font (<svg font-size>)
            text_anchor (str): Location of text relative to insertion point
        """
        kwargs = dict(text=self.name,
                      insert=self.center,
                      font_family=font_family,
                      font_size=font_size,
                      text_anchor=text_anchor)

        return svg.text.Text(**kwargs)

    def lines(self):
        """
        Return a list of svg.shape.Line objects for this shape's connections
        """
        connections = []
        for coords in self._connections.values():
            kwargs = dict(start=coords[0], end=coords[1])
            connection = svg.shapes.Line(**kwargs)
            connections.append(connection)
        return connections


class Document(Shape):
    """
    Document Class

    A Document contains Region objects, which in turn, contain Shape objects in
    the form of DataSource, Transport, DataProcessing, DataStorage, and
    UploadModule.

    """

    def __init__(self, name=None, **kwargs):
        """
        Initialize a Document object that can contain regions and shapes.

        Parameters
        ----------
            name (str): Document name to be saved as.
            size (tuple): Width and height of document in pixels.

        Notes
        -----
            If size is not defined, the default Document size is (800, 600)

            1 inch equals 90 pixels.
        """
        size = kwargs.pop('size', ('8.5in', '11in'))
        insert = kwargs.pop("insert", (0, 0))
        kwargs.update({'size': size, 'insert': insert, 'stroke': 'none'})
        super().__init__(name=name, **kwargs)
        # Container for Regions in Document
        self._regions = {}


    def __getitem__(self, item):
        return self._regions.get(item)

    @property
    def regions(self):
        return self._regions

    def add_region(self, name, insert, size, **kwargs):
        """
        Create a new Region within this document.

        Parameters
        ----------
            pass

        Notes
        -----
            See Shape class' parameters for further details into kwargs.
        """
        region = Region(name=name, insert=insert, size=size, **kwargs)
        self._regions[name] = region

    def add_to_region(self, region, shape, name, insert, size, **kwargs):
        """
        Add a shape to an existing Region within this document.

        Parameters
        ----------
            region (str): Name of Region to append new shape
            shape (str): Name of shape type to append to Region
            name (str): Name of new shape
            insert (str): Insertion point within the region
            size (str): Size of new shape

        Notes
        -----
            Shape types must be one of the following:
                - datasource
                - transport
                - datastorage
                - dataprocessing
                - uploadmodule
        """
        shape_type = shape.lower()
        if shape_type == 'datasource':
            shape = DataSource(name=name, insert=insert, size=size, **kwargs)
        elif shape_type == 'transport':
            shape = Transport(name=name, insert=insert, size=size, **kwargs)
        elif shape_type == 'datastorage':
            shape = DataStorage(name=name, insert=insert, size=size, **kwargs)
        elif shape_type == 'dataprocessing':
            shape = DataProcessing(name=name, insert=insert, size=size, **kwargs)
        elif shape_type == 'uploadmodule':
            shape = UploadModule(name=name, insert=insert, size=size, **kwargs)
        else:
            shape = Shape(name=name, insert=insert, size=size, **kwargs)

        self._regions[region].append(shape)

    def draw(self):
        """ Create the SVG file for this Document """
        document = svg.drawing.SVG(insert=self.insert, size=self.size)
        document.add(super().draw())
        for region in self._regions.values():
            document.add(region.draw())
        return document

class Region(Shape):

    def __init__(self, **kwargs):
        name = kwargs.pop('name', 'Region')
        type_ = 'Region'
        super().__init__(name=name, type=type_, padding=0, **kwargs)
        self._margin = kwargs.get("margin", 5)
        self._children = []
        self._xytitle = kwargs.get("xytitle", self._top_left)

        self._grid = kwargs.get("grid", (1, 1))
        self._cols = []
        self._rows = []

        self.__update_grid()

    def __getitem__(self, i):
        try:
            return self._shapes[i]
        except (IndexError, ValueError):
            return None

    def __repr__(self):
        name = self.name
        geometry = f"x={self._x}, y={self._y}, w={self._w}, h={self._h}"
        return (f"Region(name='{name}', {geometry})")

    def __update_grid(self):
        """
        Assign coordinates for the Horizontal and Vertical grid lines

        Notes
        -----
            Reads self.grid attribute, which is by default (None, None)
        """
        rows, cols = self._grid

        # Set up Horizontal Grid (Columns)
        if rows:
            spacing = self._h / rows
            grid_line = self._y

            _rows = []

            max_limit = self.bottom.y - spacing
            while grid_line < max_limit:
                grid_line += spacing
                _rows.append(grid_line)
            self._rows = _rows

        # Set up Horizontal Grid (Columns)
        if cols:
            spacing = self._w / cols
            grid_line = self._x

            _cols = []

            max_limit = self.right.x - spacing
            while grid_line < max_limit:
                grid_line += spacing
                _cols.append(grid_line)
            self._cols = _cols

    @property
    def shapes(self):
        return self._shapes

    @property
    def layer(self):
        return self._layer

    @property
    def children(self):
        return self._children

    @property
    def grid(self):
        return self._grid

    @property
    def cols(self):
        return self._cols

    @property
    def rows(self):
        return self._rows

    def pull(self, shape, relative=True):
        """
        Update shape geometric attributes to overlap with diagram

        Parameters
        ----------
            shape (Shape): Comparison shape to attract
            relative (bool): Recalculate relative distances within region
        """
        # Locate the shape relative to this region
        location = self.locate(shape)

        inside_region = 'inside'

        if location != inside_region:

            # Buffers assume that input shapes' coordinates are relative
            # to the region they belong to

            if self.is_left_to(shape):
                shape.x = self._right.x - shape.w - self._padding

            elif self.is_right_to(shape):
                buffer = 0
                if relative:
                    buffer = shape.x
                shape.x = self._x + self._padding + buffer

            if self.is_top_to(shape):
                shape.y = self._bottom.y - shape.h - self._padding
            elif self.is_bottom_to(shape):
                buffer = 0
                if relative:
                    buffer = shape.y
                shape.y = self._top.y + self._padding + buffer

        return shape

    def append(self, shape):
        """ Add a shape to this region """
        for child in self._children:
            if child.uid == shape.uid:
                break
        else:
            shape = self.pull(shape)
            self._children.append(shape)

    def snap_to_grid(self, how='both'):
        """
        Snap all current shapes to grid lines

        Parameters
        ----------
            how (str): 'horizontal', 'vertical', or 'both'
        """
        if how.lower() in ['horizontal', 'both']:
            # Snap shapes to horizontal gridlines
            col_half = self._w / self._grid[1] / 2
            for i, col in enumerate(self._cols):
                for child in self._children:
                    center = child.center.x

                    right = col + col_half
                    left = col - col_half

                    in_this_col = left < center <= right

                    if i == 0:
                        in_this_col = center < right

                    if i == len(self._cols) - 1:
                        in_this_col = center > left

                    if in_this_col:
                        new_x = col - child.w / 2
                        child.x = max(new_x, 0)
                        continue

        if how.lower() in ['vertical', 'both']:
            # Snap shapes to vertical gridlines
            row_half = self._h / self._grid[0] / 2
            for i, row in enumerate(self._rows):
                for child in self._children:
                    center = child.center.y

                    bottom = row + row_half
                    top = row - row_half

                    in_this_row = top < center <= bottom

                    if i == 0:
                        in_this_row = center < bottom

                    if i == len(self._rows) - 1:
                        in_this_row = center > top

                    if in_this_row:
                        new_y = row - child.h / 2
                        child.y = max(new_y, 0)
                        continue

    def draw(self):
        """
        Create the SVG file for this Region

        Returns
        -------
            SVG shape object
        """
        region = svg.drawing.SVG()
        region.add(super().draw())
        for child in self._children:
            region.add(child.draw())
        return region

class DataSource(Shape):

    def __init__(self, **kwargs):
        name = kwargs.pop('name', 'DataSource')
        type_ = 'DataSource'
        super().__init__(name=name, type=type_, **kwargs)

    def __repr__(self):
        name = self.name
        conn = list(self.connections.keys())
        return (f"DataSource(name={name}, connections={conn})")

    def draw(self):
        """
        Return an svg.shapes object representing this Shape

        Notes
        -----
            Default shape is svg.shapes.Rect

            Other kinds can be specified by using the following strings:
                1. 'text': Return a text label
                2. 'line': Return a list of line SVG shapes
        """
        kwargs = dict(points=self.sides,
                      fill=self._fill,
                      stroke=self._stroke,
                      stroke_width=self.stroke_width)

        return svg.shapes.Polygon(**kwargs)

# Abbreviating config.styles.transport for Transport class
styles_transport = STYLES.transport
shapes_transport = SHAPES.transport

class Transport(Shape):

    def __init__(self, **kwargs):
        name = kwargs.pop('name', 'sFTP')
        type_ = 'Transport'
        super().__init__(name=name, type=type_, **kwargs)
        self._reportability = kwargs.get("reportability", None)
        self._pathology_reports = kwargs.get("pathology_reports", None)
        self._levels = styles_transport.levels
        self._fill = styles_transport.fill
        self._stroke = styles_transport.stroke
        self._stroke_width = styles_transport.stroke_width
        self._font_weight = 800
        
    def __repr__(self):
        name = self.name
        conn = list(self.connections.keys())
        return (f"Transport(name={name}, connections={conn})")
    
    @property
    def levels(self):
        return self._levels
    
    def level_index(self, value):
        """
        Return the index of a value checked against self._levels
        
        Parameters
        ----------
            value (float): Number between 0 and 100
        
        Notes
        -----
            If value is None, index == 0
        """
        if value is not None:
            levels = self._levels[1:]
            remainder = dropwhile(lambda x: x < value, levels)
            remainder = list(remainder)[0]
            return 1 + levels.index(remainder)
        return 0
        
    
    def draw_bar(self, reportability=None, pathology_reports=None):
        """
        Return an svg.path object with the Transport label and frame
        
        Parameters
        ----------
            reportability (float): Percent of reportable pathology reports
            pathology_reports (int): Number of pathology reports sent
        
        Notes
        -----
            The bar content's length is depending on the percent reportability.
            Including N/A (None), there are five levels of reportability:
                - None -> Not Able to Identify or Not Applicable
                - 25 -> 0% <= reportability (r) < 25%
                - 50 -> 25% <= r < 50%
                - 75 -> 50% <= r < 75%
                - 100-> 75% <= r < 100%
        """
        if reportability:
            self._reportability = reportability

        if pathology_reports:
            self._pathology_reports = pathology_reports
        
        percent = self._reportability
        level = self.level_index(percent)
        frame_fill = 'none'
        
        if percent is None:
            percent = self.x
            frame_fill = self._fill[level]            
        else:
            percent = self.x + self.w * percent / 100
                
        coords = dict(x=self.x, 
                      y=self.y, 
                      r=self.h / 2,
                      right_x=self.right.x, 
                      bottom_y=self.bottom.y)

        styles = dict(fill=frame_fill,
                      stroke=self.stroke, 
                      stroke_width=self.stroke_width)
        
        g = svg.drawing.SVG()
        
        # Create the Frame first
        d = shapes_transport.frame
        frame = svg.path.Path(d=d.format(**coords), **styles)
        g.add(frame)
        
        # Create the fill bar
        coords['r'] -= self.padding
        coords['y'] += self.padding
        coords['right_x'] = percent
        coords['bottom_y'] -= self.padding
        
        styles['fill'] = self._fill[level]
        styles['stroke'] = 'none'
    
        d = shapes_transport.meter
        meter = svg.path.Path(d=d.format(**coords), **styles)
        g.add(meter)
        
        # Add the text label
        styles['insert'] = self.center # Old: (self.top.x, self.right.y + self.padding)
        styles['alignment_baseline'] = self.alignment_baseline
        styles['text_anchor'] = self.text_anchor
        styles['fill'] = self.font_color
        styles['font_family'] = self.font_family
        styles['font_size'] = self.font_size
        styles['font_weight'] = self.font_weight
        text = svg.text.Text(self.name, **styles)
        g.add(text)
        
        # Add the Reportability Label and % if applicable
        styles['insert'] = (self.left.x, self.y - self.padding)
        styles['text_anchor'] = 'start'
        styles['fill'] = 'gray'
        styles['font_size'] = 7
        styles['font_weight'] = 400
        if self._reportability:
            text = f"{int(self._reportability)}% Reportable"
        else:
            text = f"No Data Available"
        label = svg.text.Text(text, **styles)
        g.add(label)
        return g

    def draw(self, curve=40):
        """
        Return an svg.shapes object representing this Shape

        Parameters
        ----------
            difference (tuple): Pixels to add to the curves for this shape

        Notes
        -----
            Default shape is svg.shapes.Rect

            Other kinds can be specified by using the following strings:
                1. 'text': Return a text label
                2. 'line': Return a list of line SVG shapes
        """

        top, right, bottom, left = self.sides

        d = config.shapes.transport
        d = d.format(x=self.x,
                     y=self.y,
                     right_x=self.right.x,
                     radius=self.h / 2,
                     bottom_y=self.bottom.y)

        kwargs = dict(d=d,
                      fill=self._fill,
                      stroke=self._stroke,
                      stroke_width=self.stroke_width)

        return svg.path.Path(**kwargs)


class Connection(Transport):
    """
    Connection is a type of Transport represented as a line with attributes.
    The attributes are line style, color, and width. These attributes refer to
    presence, reportability rate, and volume.
    respectively.
    """
    
    def __init__(self, **kwargs):
        self._start = kwargs.pop("start", (0, 0))
        self._end = kwargs.pop("end", (100, 100))
        self._volume = kwargs.pop("volume", None)
        self._reportability = kwargs.pop("reportability", None)
        self._presence = self._volume and self._reportability
        super().__init__(**kwargs)

    @property
    def start(self):
        return self._start
    
    @start.setter
    def start(self, value):
        self._start = self.COORDINATES(*value)

    @property
    def end(self):
        return self._start
    
    @end.setter
    def end(self, value):
        self._end = self.COORDINATES(*value)

    @property
    def presence(self):
        return self._presence

    @property
    def reportability(self):
        return self._reportability
    
    @reportability.setter
    def reportability(self, value):
        assert isinstance(value, float)
        self._reportability = value
        self._presence = self._volume and self._reportability

    @property
    def volume(self):
        return self._start
    
    @volume.setter
    def volume(self, value):
        assert isinstance(value, float)
        self._volume = value
        self._presence = self._volume and self._reportability
    
    def dasharray(self):
        """
        Get a solid line if volume data exists. Otherwise, dashed.
        
        Returns
        -------
            Solid if pathology volume data is present. Otherwise, Dashed.
        """
        if self.presence:
            return [0]
        return [5]
    
    def stroke(self, name='viridis', lut=None):
        """
        Return a color along a colormap corresponding to the reportability.
        
        Parameters
        ----------
            name (str): Colormap name to use
            lut (int): Number of tints to return. Default is None.
        
        Returns
        -------
            Float between 0 and 1 representing reportability rate
        """
        color = np.array([0,0,0,1])        
        if self.reportability is not None:
            cmap = get_cmap(name, lut)
            color = cmap(self.reportability)
        return rgb2hex(color)
    
    def stroke_width(self, max_width=25):
        """
        Return a width corresponding to the percent of volume for this
        Connection.
        
        Parameters
        ----------
            max_width (int): Maximum stroke width to derive widths from.
        
        Returns
        -------
            Float between 0 and 1 representing volume. Volume is defined by
            the number of reports going through this Connection, divided by
            all reports leaving a DataSource.
        """
        return max_width * self.volume
    
    def draw(self, max_width=25):
        """
        Return an svg.shapes object representing this Shape.
        
        Parameters
        ----------
            max_width (int): Maximum stroke width to derive widths from.
            difference (tuple): Pixels to add to the curves for this shape

        Notes
        -----
            Will return a closed shape and a path
        """
        kwargs = dict(start=self.start, 
                      end=self.end, 
                      stroke=self.stroke(),
                      stroke_width=self.stroke_width())
        return svg.shapes.Line(**kwargs).dasharray(self.dasharray())
    


class DataStorage(Shape):

    def __init__(self, **kwargs):
        name = kwargs.pop('name', 'DB Storage')
        type_ = 'Application/Data Storage'
        svg_shape = 'circle'
        super().__init__(name=name, type=type_, svg_shape=svg_shape, **kwargs)

    def __repr__(self):
        name = self.name
        conn = list(self.connections.keys())
        return (f"DataStorage(name={name}, connections={conn})")

    def draw(self, curve=4):
        """
        Return an svg.shapes object representing this Shape

        Parameters
        ----------
            difference (tuple): Pixels to add to the curves for this shape

        Notes
        -----
            Will return a closed shape and a path
        """

        top, right, bottom, left = self.sides

        d = (f"M{left.x} {top.y} "
             f"Q{top.x} {top.y - self.h / curve} {right.x} {top.y} "
             f"V{right.x} {bottom.y} "
             f"Q{bottom.x} {bottom.y + self.h / curve} {left.x} {bottom.y} "
             f"Z")

        kwargs = dict(d=d,
                      fill=self._fill,
                      stroke=self._stroke,
                      stroke_width=self.stroke_width)

        return svg.path.Path(**kwargs)


class DataProcessing(Shape):

    def __init__(self, **kwargs):
        name = kwargs.pop('name', 'Application')
        type_ = 'Application/Data Processing'
        super().__init__(name=name, type=type_, **kwargs)
        self.svg_shape = 'circle'

    def __repr__(self):
        name = self.name
        conn = list(self.connections.keys())
        return (f"DataProcessing(name={name}, connections={conn})")


class UploadModule(Shape):

    def __init__(self, **kwargs):
        name = kwargs.pop('name', 'Upload Module')
        type_ = 'Application/Upload Module'
        super().__init__(name=name, type=type_, **kwargs)
        self.svg_shape = 'rect'

    def __repr__(self):
        name = self.name
        conn = list(self.connections.keys())
        return (f"UploadModule(name={name}, connections={conn})")

def draw_shape(definition, fp='draw_shape_py.svg', pad=5):
    """
    Create a shape from a coordinate definitions

    Parameters
    ----------
        definition (namedtuple): Path definition
    """
    shape = Shape(name="Test", insert=(100, 100), size=(150, 50))
    svgdoc = svg.Drawing(fp)
    g = svgdoc.add(svg.drawing.SVG())
    fields = definition._fields

    # Required variables for health_bar are: x, y, r, right_x, bottom_y, percent
    for i, d in enumerate(definition):
        field = fields[i]
        defs = dict(x=shape.x, y=shape.y, r=shape.h / 2, right_x=shape.right.x, bottom_y=shape.bottom.y)
        kwargs = dict(fill='none', stroke='black', stroke_width=5)

        if field == 'reportability':
            defs = dict(x=shape.x + pad, y=shape.y + pad, r=(shape.h - pad * 2) / 2, right_x=shape.right.x - pad, bottom_y=shape.bottom.y - pad)
            kwargs['stroke'] = 'gray'

        if field == 'meter':
            defs = dict(x=shape.x + pad, y=shape.y + pad, r=(shape.h - pad * 2) / 2, right_x=shape.right.x - pad, bottom_y=shape.bottom.y - pad)
            percent = round(shape.x + shape.w * np.random.rand())
            defs['percent'] = percent
            kwargs['fill'] = 'green'
            kwargs['stroke'] = 'none'

        path = svg.path.Path(d=d.format(**defs), **kwargs)
        g.add(path)

    svgdoc.save()
    return svgdoc
        


if __name__=="__main__":
    
    # Test Drawing the Transport Bar
    names = ['PHINMS', 'sFTP', 'AIM', 'CAS']
    d = svg.Drawing("test_transport.svg")
    y = 25
    for k in range(0, 201, 50):
        name = np.random.choice(names)
        t = Transport(name=name, insert=(25, 25), size=(150, y))
        t.y += k
        reportability = np.random.choice(np.arange(0, 1, 0.01)) * 100
        d.add(t.draw_bar(reportability))
    
    t.y += 50
    t._reportability = None
    d.add(t.draw_bar())
    d.save()
    
    # --- END TESTING TRANSPORT BAR ---
        
#    draw_shape(config.shapes.transport)
#    s1 = Shape(insert=(0,0), size=(10, 10))
#    s2 = Shape(insert=(10,10),size=(10,10))
#    s3 = Shape(insert=(15,15),size=(10,10))
#    s4 = Shape(insert=(16,16),size=(5,5))
#    s5 = Shape(insert=(16,16),size=(10,10))
#    s6 = Shape(insert=(0,16),size=(100,10))
#    s7 = Shape(insert=(30, 30),size=(10,10))
#
#    r1 = Region(insert=(0, 0), size=(20, 20), grid=(5, 3))
#
#    for _ in [s1, s2, s3]:
#        r1.append(_)
#
#    print(s1.center, s2.center, s3.center)
#    print([c.center for c in r1.children])
#    r1.snap_to_grid()
#    print(r1.rows, r1.cols)
#    print([c.center for c in r1.children])
#
#    d = Document('test')
#    d.add_region('Test', (0, 0), (100, 100))
#    d.add_region("New Region", (100, 0), (200, 300), fill='red')
#    d.add_to_region('Test', 'datasource', 'Lab', (0, 0), (10, 10))
#    d.add_to_region("New Region", "transport", "PHINMS", (0, 0), (100, 10))
#    print(d.size, d.regions, d.regions['Test'].children)
#    print(d.regions['New Region'].children[0].corners)
#    print(d['New Region'])
#    drawing = d.draw()
#    print(drawing.tostring())