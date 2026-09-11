# Visualization layer

The plotting helpers in this package use Matplotlib's object-oriented API. They return `(Figure, Axes)` and do not call `plt.show()`, so they can be used directly in scripts or embedded later in a PySide6 canvas.
