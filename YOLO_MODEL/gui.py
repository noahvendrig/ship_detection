from tkinter import *

root = Tk()

# Creating a label widget
myLabel1 = Label(root, text="hello world")
myLabel2 = Label(root, text="row 2 text ewrfew")

# Shoving it onto the screen
# myLabel.pack()

# Making it a grid
myLabel1.grid(row=0, column=0)
myLabel2.grid(row=1, column=0)

root.mainloop()