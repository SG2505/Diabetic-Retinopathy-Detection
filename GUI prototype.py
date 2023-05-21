import tkinter as tk
import customtkinter as ctk
import numpy as np
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog as fd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import cv2
import pyautogui
from keras.models import load_model
from tensorflow.keras.preprocessing import image

# from tkinter import ttk
# import matplotlib.animation as animation
# import random

cropped_image_proc = None

ctk.set_appearance_mode('light')
root = tk.Tk()
root.title('Eye blindness detection')
root.state('zoomed')
root.resizable(False, False)

# Frames
frame = ctk.CTkFrame(root, width=600, corner_radius=10)
frame.grid(row=0, column=0, padx=30, pady=10, sticky='nsw')
frame.grid_propagate(False)
frame2=ctk.CTkFrame(frame,width=250,height=250,corner_radius=10,fg_color= 'slate gray3')
frame3=ctk.CTkFrame(frame,width=250,height=250,corner_radius=10,fg_color='slate gray3')

# Labels
label = ctk.CTkLabel(frame, text='Original image', text_color='black', font=('Bold', 14), fg_color='ivory4',
                     corner_radius=10)
label_OImage = ctk.CTkLabel(frame2, text='')
label_OImage.place(x=220, y=280)
label2 = ctk.CTkLabel(frame, text='Processed image', text_color='black', font=('Bold', 14), fg_color='ivory4',
                      corner_radius=10)
label_Pimage = ctk.CTkLabel(frame3, text='')
label_Pimage.place(x=220, y=500)
patient_label = ctk.CTkLabel(frame, text='Patient\'s name:', text_color='black', font=('Bold', 14), fg_color='ivory3',
                             corner_radius=8)
patient_label.place(x=335, y=30)
detail_label = ctk.CTkLabel(root, text='Details:', text_color='black', font=('Bold', 16), fg_color='ivory3',
                            corner_radius=10)
status_label = ctk.CTkLabel(root, width=90, height=40, font=('Bold', 18), text_color='black', fg_color='ivory3',
                            corner_radius=10)

# Entry
patient_entry = ctk.CTkEntry(frame)
patient_entry.place(x=450, y=30)


def load_image():
    # Ask the user to choose an image file
    file_path = fd.askopenfilename()
    if not file_path:
        return

    # Open the image file
    Oimage_file = Image.open(file_path)
    Oimage_resized = Oimage_file.resize((290, 290))
    # Convert the image to a PhotoImage object
    final_Oimage = ImageTk.PhotoImage(Oimage_resized)

    # Display the original image in the label
    label_OImage.configure(image=final_Oimage)
    label_OImage.image = final_Oimage  # This line is necessary to prevent the image from being garbage collected

    # Perform circular crop on the image
    Oimage_arr = np.asarray(Oimage_file)
    cropped_image = circle_crop(Oimage_arr)
    cropped_image_proc = Image.fromarray(cropped_image)
    cropped_image_proc.save('D:\\University\\CESS\\semester 6 junior\\AI\\GUI\\preprocessed.png')
    cropped_image = cropped_image_proc.resize((290, 290))
    cropped_photo = ImageTk.PhotoImage(cropped_image)
    # Display the cropped image in the label
    label_Pimage.configure(image=cropped_photo)
    label_Pimage.image = cropped_photo
    label.place(x=250, y=80)
    label2.place(x=240, y=380)
    frame2.place(x=190, y=113)
    frame3.place(x=190, y=413)
    label_OImage.place(x=9, y=9)
    label_Pimage.place(x=9, y=9)
    predict_button.configure(state='normal')


def save_image():
    # Use PyAutoGUI to take a screenshot
    # Get the screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Capture the screenshot
    screenshot = pyautogui.screenshot()

    # Convert the screenshot to PIL Image object
    image = Image.frombytes("RGB", screenshot.size, screenshot.tobytes())

    # Save the image to the specified file path
    text = patient_entry.get()
    image.save('D:\\University\\CESS\\semester 6 junior\\AI\\GUI\\{} result.png'.format(text))

    # Show a message box to indicate successful screenshot capture
    tk.messagebox.showinfo("Screenshot Captured", "Screenshot saved successfully.")


def model_predict():
    # Define the data for the bar chart
    # Load and preprocess the image
    img = image.load_img('D:\\University\\CESS\\semester 6 junior\\AI\\GUI\\preprocessed.png', target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # Assuming it's a classification task
    print(predicted_class)
    if predicted_class == 0:
        status_label.configure(text='No Diabetic Retinopathy')
    elif predicted_class == 1:
        status_label.configure(text='Mild Diabetic Retinopathy')
    elif predicted_class == 2:
        status_label.configure(text='Moderate Diabetic Retinopathy')
    elif predicted_class == 3:
        status_label.configure(text='Severe Diabetic Retinopathy')
    elif predicted_class == 4:
        status_label.configure(text='Proliferative Diabetic Retinopathy')

    # Create a Matplotlib figure and add a bar chart to it
    fig = Figure(figsize=(6, 5), dpi=90)
    ax = fig.add_subplot(111)
    ax.bar(range(len(predictions[0])), predictions[0])
    ax.set_xlabel('State')
    ax.set_ylabel('Percentage')
    # Create a FigureCanvasTkAgg widget to display the figure in the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().place(x=1070, y=280)
    detail_label.place(x=760, y=240)
    status_label.place(x=950, y=120)
    text = patient_entry.get()
    if text:
        save_button.configure(state='normal')
    else:
        tk.messagebox.showinfo("Error", "Please enter patient\'s name")


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img


def circle_crop(img, sigmaX=30):
    """
    Create circular crop around image centre
    """
    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    return img


# Buttons
save_button = ctk.CTkButton(frame, text="save", state='disabled', font=('Bold', 16), width=100, corner_radius=15,
                            command=save_image)
save_button.place(x=10, y=30)
load_button = ctk.CTkButton(frame, text='Load image', font=('Bold', 16), width=200, corner_radius=15,
                            command=load_image)
load_button.place(x=200, y=750)
predict_button = ctk.CTkButton(root, text='Predict', state='disabled', hover=True, hover_color='royal blue',
                               font=('Bold', 16), width=200, corner_radius=15, command=model_predict)
predict_button.place(x=990, y=750)

# class HoverButton(ctk.CTkButton):
#     def __init__(self, master, **kw):
#         ctk.CTkButton.__init__(self,master=master,**kw)
#         self.defaultBackground = self["background"]
#         self.bind("<Enter>", self.on_enter)
#         self.bind("<Leave>", self.on_leave)

#     def on_enter(self, e):
#         self.configure(fg_color='maroon')

#     def on_leave(self, e):
#         self.configure(fg_color='red4')


frame.columnconfigure(0, weight=1)
frame.rowconfigure(0, weight=1)
frame.rowconfigure(1, weight=1)
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

model_path = "D:\\projects\\AI projects\\DenseNet-diabetic retinopathy class 3&4 augmented.h5"
model = load_model(model_path)

root.mainloop()