#!/usr/bin/env python


import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt

class DrawPrompt:
  def __init__(self,original, min_quota,  max_quota):

    self.max_quota = max_quota #original.max()
    self.min_quota = min_quota #original.min()

    root = tk.Tk()
    self.root = root
    self.original = original
    # length of the time series
    self.length   = len(original)

    self.root.title("Draw a Timeserie")

    # vertical quantization
    self.quantization = 50
    # how large is a pixel?
    self.horizontal_cell_size = 10
    self.vertical_cell_size   = 10

    # compute canvass properties
    self.width  =  self.length * self.horizontal_cell_size
    self.height =  self.quantization * self.vertical_cell_size
    self.canvas = tk.Canvas(root,
                            width = self.width,
                            height = self.height, bg="white")
    self.canvas.pack()

    self.canvas.bind("<B1-Motion>", self.paint)
    self.canvas.bind("<Button-1>", self.paint)

    self.clear_canvas()

    self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
    self.clear_button.pack()

    self.save_button = tk.Button(root, text="Save", command=self.save_canvas)
    self.save_button.pack()
    root.mainloop()


  def get_vertical_pos(self,v, x):
    query = v[x]
    m = self.min_quota #v.min()
    M = self.max_quota #v.max()
    return self.quantization - (v[x] - m)/(M - m) * self.quantization

  def paint(self, event):
    x = event.x // self.horizontal_cell_size
    y = event.y // self.vertical_cell_size

    if 0 <= x < self.length and 0 <= y < self.quantization:
      self.cells[:,x] =  np.zeros(self.quantization)
      self.cells[y,x] =  1.
      self.delete_column(x)
      self.draw_cell(x, y)
      # disegno la timeseries di sfondo
      self.draw_cell(x, self.get_vertical_pos(self.original, x), 'orange')

  def draw_cell(self, x, y, color = 'black'):
    x1 = x * self.horizontal_cell_size
    y1 = y * self.vertical_cell_size
    x2 = x1 + self.horizontal_cell_size
    y2 = y1 + self.vertical_cell_size

    self.canvas.create_rectangle(x1, y1, x2, y2, fill= color, outline = '')

  def delete_column(self, x):
    for i in range(self.quantization):
      y = i
      self.draw_cell(x, i , "white")

  def clear_canvas(self):
    self.canvas.delete("all")
    for x in range(self.length):
      self.draw_cell(x, self.get_vertical_pos(self.original, x), 'orange')

    self.cells = np.array([[0 for _ in range(self.length)] for _ in range(self.quantization)])

  def save_canvas(self):
    self.root.destroy();
    vertical_positions = (self.cells[-1::-1].argmax(axis = 0))
    self.data = vertical_positions /(self.quantization * 1.) * (self.max_quota - self.min_quota)  + self.min_quota
    self.data = np.interp(np.arange(len(vertical_positions)),
                          np.arange(len(vertical_positions))[vertical_positions != 0],
                          self.data[vertical_positions != 0])
if __name__ == "__main__":
#    root = tk.Tk()
    original = np.sin(np.linspace(0,2 * np.pi,128))
    app = DrawPrompt(original)
#    root.mainloop()
    plt.figure()
    plt.plot(app.data)
    plt.show()
