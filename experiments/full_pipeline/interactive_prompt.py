#!/usr/bin/env python


import tkinter as tk
import numpy as np


class DrawApp:
  def __init__(self, root, length = 128, min_height = 0, max_height = 10.):
    self.root = root
    self.root.title("Draw a Timeserie")

    # length of the time series
    self.length = length
    # vertical quantization
    self.quantization = 10
    # how large is a pixel?
    self.cell_size = 10

    # compute canvass properties
    self.width  =  self.length * self.cell_size
    self.height =  self.quantization * self.cell_size
    self.canvas = tk.Canvas(root,
                            width = self.width,
                            height = self.height, bg="white")
    self.canvas.pack()

    self.canvas.bind("<B1-Motion>", self.paint)
    self.canvas.bind("<Button-1>", self.paint)

    self.cells = np.array([[0 for _ in range(self.length)] for _ in range(self.quantization)])

    self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
    self.clear_button.pack()

  def paint(self, event):
    x = event.x // self.cell_size
    y = event.y // self.cell_size

    if 0 <= x < self.length and 0 <= y < self.quantization:
      self.cells[:,x] =  np.zeros(self.quantization)
      self.cells[y,x] =  1.
      self.delete_column(x)
      self.draw_cell(x, y)

  def draw_cell(self, x, y, color = 'black'):
    x1 = x * self.cell_size
    y1 = y * self.cell_size
    x2 = x1 + self.cell_size
    y2 = y1 + self.cell_size

    self.canvas.create_rectangle(x1, y1, x2, y2, fill= color)

  def delete_column(self, x):
    for i in range(self.quantization):
      y = i
      self.draw_cell(x, i , "white")


  def clear_canvas(self):
    self.canvas.delete("all")
    self.cells = np.array([[0 for _ in range(self.length)] for _ in range(self.quantization)])

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()

