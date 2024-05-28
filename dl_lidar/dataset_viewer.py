from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import sys
import gzip
import os


class MyGraphicsWindow(pg.GraphicsWindow):
    sigKeyPress = QtCore.pyqtSignal(object)

    def keyPressEvent(self, ev):
        self.sigKeyPress.emit(ev)


class MyGLViewWidget(gl.GLViewWidget):
    sigKeyPress = QtCore.pyqtSignal(object)

    def keyPressEvent(self, ev):
        self.sigKeyPress.emit(ev)


class DatasetViewer:
    def __init__(self, data, labels, name, rows=16):
        self.data = np.swapaxes(data, 1, 2)      # transpose to (n, 2048, 16, 4) to put points in scan-order
        self.labels = np.swapaxes(labels, 1, 2)  # transpose to (n, 2048, 16) to put labels in scan-order
        self.name = name
        self.rows = rows
        self.index = 0
        self.data_size = len(self.data)

        self.plot = None
        self.text = None
        self.labels_image = None
        self.intensity_image = None
        self.input_box = None

    def _update(self):
        xyz = np.reshape(self.data[self.index, :, :, 0:3], (self.rows*2048, 3))        # list of points in scan-order
        label = np.reshape(self.labels[self.index, :, :], self.rows*2048).astype(int)  # list of labels in scan-order

        # convert labels to one-hot encoding
        colors = np.zeros((self.rows*2048, 4))
        colors[np.arange(label.size), label] = 1
        # add a fourth column to make colors work as an RGBA array
        colors[np.arange(label.size), 3] = 1

        image1 = np.reshape(colors, (2048, self.rows, 4))    # 2048x16 column-major RGBA image
        image2 = self.data[self.index, :, :, 3].astype(int)  # 2048x16 column-major int image

        # flip vertical axis because ImageItem origin is bottom left
        image1 = np.flip(image1, axis=1)
        image2 = np.flip(image2, axis=1)

        self.plot.setData(pos=xyz, color=colors)
        self.text.setText('index = {} of {}'.format(self.index, self.data_size - 1), color='000000', bold=True)
        self.labels_image.setImage(image1)
        self.intensity_image.setImage(image2)

    def _keyPressed(self, evt):
        # handle arrow keys to change index and displayed image
        if evt.key() == QtCore.Qt.Key_Down:
            if self.index > 0:
                self.index -= 1
                self._update()
        elif evt.key() == QtCore.Qt.Key_Up:
            if self.index < self.data_size - 1:
                self.index += 1
                self._update()

    def _enterPressed(self):
        text_input = int(self.input_box.text())
        if 0 <= text_input < self.data_size:
            self.index = text_input
        self._update()
        self.input_box.clear()

    def show(self):
        app = QtGui.QApplication([])

        # window
        title = 'Dataset Viewer: ' + self.name
        window = MyGraphicsWindow(title=title)
        window.sigKeyPress.connect(self._keyPressed)
        pg.setConfigOption('background', 'w')

        # 3D plot
        plot_widget = MyGLViewWidget()
        plot_widget.sigKeyPress.connect(self._keyPressed)
        plot_widget.opts['distance'] = 20
        plot_grid = gl.GLGridItem()
        plot_widget.addItem(plot_grid)
        self.plot = gl.GLScatterPlotItem(size=5)
        plot_widget.addItem(self.plot)

        # 2D images
        image_widget = pg.GraphicsLayoutWidget()

        self.text = pg.LabelItem('index = {} of {}'.format(self.index, self.data_size - 1), color='000000', bold=True)
        image_widget.addItem(self.text, 0, 0)

        self.labels_image = pg.ImageItem()
        labels_view = image_widget.addViewBox(1, 0, enableMouse=False)
        labels_view.addItem(self.labels_image)

        self.intensity_image = pg.ImageItem()
        intensity_view = image_widget.addViewBox(2, 0, enableMouse=False)
        intensity_view.addItem(self.intensity_image)

        # input box for going to a specific point cloud
        self.input_box = QtGui.QLineEdit()                        # creates the input box widget
        self.input_box.setValidator(QtGui.QIntValidator())        # prevents non-integers from being typed
        self.input_box.returnPressed.connect(self._enterPressed)  # causes enter key to call enterPressed function
        box_label = QtGui.QLabel()
        box_label.setText('Go to Point Cloud #')
        box_label.setStyleSheet('color: white')

        # load first point cloud
        self._update()

        # arrange widgets
        layoutgb = QtGui.QGridLayout()
        input_box_layout = QtGui.QGridLayout()
        window.setLayout(layoutgb)
        layoutgb.addWidget(plot_widget, 1, 0)       # places 3D visualization to window at top
        layoutgb.addWidget(image_widget, 2, 0)      # places 2D images and current index text in middle

        # creates a layout just for input box, putting them side by side
        input_box_layout.addWidget(self.input_box, 0, 1)
        input_box_layout.addWidget(box_label, 0, 0)

        # adds input box layout to rest of layout, below everything else
        layoutgb.addLayout(input_box_layout, 3, 0)

        # from stackoverflow https://stackoverflow.com/a/53795923
        image_widget.sizeHint = lambda: pg.QtCore.QSize(100, 100)
        plot_widget.sizeHint = lambda: pg.QtCore.QSize(100, 100)
        plot_widget.setSizePolicy(image_widget.sizePolicy())

        QtGui.QApplication.instance().exec_()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python {} data.npy.gz labels.npy.gz'.format(sys.argv[0]))
        sys.exit(1)

    f_data = gzip.GzipFile(sys.argv[1], 'r')
    f_labels = gzip.GzipFile(sys.argv[2], 'r')

    data = np.load(f_data)
    labels = np.load(f_labels)

    f_data.close()
    f_labels.close()

    dv = DatasetViewer(data, labels, os.path.basename(sys.argv[1]) + ' and ' + os.path.basename(sys.argv[2]))
    dv.show()
