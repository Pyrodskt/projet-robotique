import cv2
import tkinter as tk
from PIL import Image, ImageTk

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('/dev/video2')

# Create a Tkinter window
window = tk.Tk()
window.title("OpenCV and Tkinter")

# This function is called every time the video frame is updated
def update_frame():
    # Read the next frame from the video capture object
    _, frame = cap.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which Tkinter can display)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a PhotoImage object from the frame
    image = ImageTk.PhotoImage(image=Image.fromarray(frame))

    # Update the label's image
    label.config(image=image)
    label.image = image

    # Set the delay between frames
    window.after(15, update_frame)

# Create a label to display the video frame
label = tk.Label(master=window)
label.pack()

# Start the frame update loop
update_frame()

# Run the Tkinter event loop
window.mainloop()

# Release the video capture object
cap.release()
