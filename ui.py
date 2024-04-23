import tkinter as tk

def simulate_traffic_light(color):
    root = tk.Tk()
    root.title("Traffic Light Simulation")

    canvas = tk.Canvas(root, width=100, height=300, bg="white")
    canvas.pack()

    if color == "red":
        canvas.create_oval(25, 25, 75, 75, fill="red")  # Red light
        canvas.create_oval(25, 100, 75, 150, outline="gray")  # Yellow light (not lit)
        canvas.create_oval(25, 175, 75, 225, outline="gray")  # Green light (not lit)
    elif color == "green":
        canvas.create_oval(25, 25, 75, 75, outline="gray")  # Red light (not lit)
        canvas.create_oval(25, 100, 75, 150, outline="gray")  # Yellow light (not lit)
        canvas.create_oval(25, 175, 75, 225, fill="green")  # Green light
    else:
        print("Invalid color provided.")
    
    root.mainloop()
        
