# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 09:15:07 2024

@author: ASUS
"""

import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, Toplevel, HORIZONTAL, CENTER
from PIL import Image, ImageTk, ImageDraw, ImageEnhance, ImageFont
import os
import numpy as np
import tkinter as ttk
import cv2
from rembg import remove, new_session
import io 
import math

class CollageMaker:
    def __init__(self, root):
        self.root = root
        self.root.title("Collage Maker")
        
        self.instructions = tk.Label(root, text="Step 1: Upload images to create a collage.")
        self.instructions.pack()

        self.upload_button = tk.Button(root, text="Upload Images", command=self.upload_images)
        self.upload_button.pack()

        # Store references to all PhotoImage objects to avoid garbage collection
        self.image_tk_objects = []

        # Initialize canvas and control panel with scrollable frame
        self.canvas = tk.Canvas(root, width=800, height=800, bg="white")
        self.canvas.pack(side=tk.LEFT, padx=100, pady=10)

        self.control_frame = tk.Frame(root)
        self.control_canvas = tk.Canvas(self.control_frame, width=450)
        self.control_scrollbar = tk.Scrollbar(self.control_frame, orient=tk.VERTICAL, command=self.control_canvas.yview)
        self.scrollable_frame = tk.Frame(self.control_canvas, bg="white")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.control_canvas.configure(
                scrollregion=self.control_canvas.bbox("all")
            )
        )

        self.control_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)

        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=30, pady=10)
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.control_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        #self.create_control_buttons()
        
    

        self.selected_image_item = None
        self.image_items = []
        self.image_objects = []
        self.image_tk_objects = []
        self.images = []
        
        # Track text items separately
        self.text_items = []

        self.create_control_buttons()

        self.bg_color = "white"
        self.bg_image = None
        self.paint_color = "black"
        self.painting = False
        self.erasing = False

        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.strokes = []

        self.buffer = Image.new("RGBA", (800, 800), (255, 255, 255, 0))
        self.buffer_draw = ImageDraw.Draw(self.buffer)
        self.buffer_tk = ImageTk.PhotoImage(self.buffer)
        self.buffer_image_item = self.canvas.create_image(0, 0, image=self.buffer_tk, anchor=tk.NW)
        
        # Temporary buffer for painting
        self.temp_buffer = Image.new("RGBA", (800, 800), (255, 255, 255, 0))
        self.temp_buffer_draw = ImageDraw.Draw(self.temp_buffer)
        self.temp_buffer_tk = ImageTk.PhotoImage(self.temp_buffer)
        self.temp_buffer_image_item = self.canvas.create_image(0, 0, image=self.temp_buffer_tk, anchor=tk.NW)

        self.temp_strokes = []

        self.cropping = False
        self.crop_start_x = None
        self.crop_start_y = None
        self.crop_rect = None

        # Initialize text dragging variables
        self.dragging_text = None
        self.drag_start_x = None
        self.drag_start_y = None
        
    def create_control_buttons(self):
        # Control sections for better organization
        self.add_section_label("Image Operations", "red")
        
        self.crop_button = self.create_centered_button("Crop Selected Image", self.start_crop)
        self.apply_crop_button = self.create_centered_button("Apply Crop", self.apply_crop)
        self.cancel_crop_button = self.create_centered_button("Cancel Crop", self.cancel_crop)

        self.create_centered_label("Enter rotation angle (degrees):")
        self.rotate_angle_entry = self.create_centered_entry()
        
        self.create_centered_label("Enter new size (width,height):")
        self.resize_entry = self.create_centered_entry()
        
        self.rotate_button = self.create_centered_button("Rotate Selected Image", self.rotate_image)
        self.resize_button = self.create_centered_button("Resize Selected Image", self.resize_image)

        self.add_section_label("Background Options", "red")  # Red heading
        self.bg_color_button = self.create_centered_button("Select Background Color", self.select_bg_color)
        self.bg_image_button = self.create_centered_button("Upload Background Image", self.upload_bg_image)

        self.add_section_label("Filters and Effects", "red")  # Red heading
        self.filter_button = self.create_centered_button("Apply Filter to Selected Image", self.open_filter_window)
        self.remove_bg_button = self.create_centered_button("Remove Background", self.remove_background)

        self.add_section_label("Painting Tools", "red")  # Red heading
        self.select_color_button = self.create_centered_button("Select Paint Color", self.select_paint_color)
        
        self.create_centered_label("Select Paint Thickness:")
        self.paint_thickness = self.create_centered_scale(1, 10)
        
        self.eraser_button = self.create_centered_button("Eraser", self.enable_eraser)

        self.create_centered_label("Select Eraser Thickness:")
        self.erase_thickness = self.create_centered_scale(1, 10)
        
        self.apply_paint_button = self.create_centered_button("Apply Paint", self.apply_paint)
        self.reset_paint_button = self.create_centered_button("Reset Paint", self.reset_paint)

        self.add_section_label("Text Options", "red")  # Red heading
        self.create_centered_label("Enter text:")
        self.text_entry = self.create_centered_entry()

        self.create_centered_label("Select Font Size:")
        self.font_size_entry = self.create_centered_entry()
        self.font_size_entry.insert(0, "20")  # Default font size
        
        self.create_centered_label("Select Font:")
        self.font_choice = tk.StringVar(self.scrollable_frame)
        self.fonts = ["Arial", "Courier", "Times New Roman", "Helvetica", "Comic Sans MS"]
        self.font_choice.set(self.fonts[0])  # Default font
        self.font_menu = self.create_centered_option_menu(self.font_choice, *self.fonts)
        
        self.text_color_button = self.create_centered_button("Select Text Color", self.select_text_color)
        self.add_text_button = self.create_centered_button("Add Text to Canvas", self.add_text)
        self.reset_text_button = self.create_centered_button("Reset Text", self.reset_text)

        self.add_section_label("Save Options", "red")  # Red heading
        self.save_button = self.create_centered_button("Save Collage", self.save_collage, color="#34A853")  # Green color

    def add_section_label(self, text, color="black"):
        section_label = tk.Label(self.scrollable_frame, text=text, font=("Helvetica", 14, "bold"), fg=color)
        section_label.pack(pady=10)

    def create_centered_button(self, text, command, color=None):
        frame = tk.Frame(self.scrollable_frame)
        button = tk.Button(frame, text=text, command=command, bg=color if color else "SystemButtonFace")
        button.pack(expand=True)
        frame.pack(anchor="center", pady=5)
        return button

    def create_centered_label(self, text):
        frame = tk.Frame(self.scrollable_frame)
        label = tk.Label(frame, text=text)
        label.pack(expand=True)
        frame.pack(anchor="center", pady=5)
        return label

    def create_centered_entry(self):
        frame = tk.Frame(self.scrollable_frame)
        entry = tk.Entry(frame)
        entry.pack(expand=True)
        frame.pack(anchor="center", pady=5)
        return entry

    def create_centered_scale(self, from_, to):
        frame = tk.Frame(self.scrollable_frame)
        scale = tk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL)
        scale.pack(expand=True)
        frame.pack(anchor="center", pady=5)
        return scale

    def create_centered_option_menu(self, variable, *options):
        frame = tk.Frame(self.scrollable_frame)
        option_menu = tk.OptionMenu(frame, variable, *options)
        option_menu.pack(expand=True)
        frame.pack(anchor="center", pady=5)
        return option_menu
    


    def disable_paint(self):
        self.painting = False
        self.erasing = False
        self.instructions.config(text="")

    def upload_images(self):
        self.disable_paint()
        image_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if not image_paths:
            return

        new_images = []
        new_image_tk_objects = []
        for path in image_paths:
            img = Image.open(path).convert("RGBA")
            new_images.append(img)

        self.images.extend(new_images)
        self.display_images(new_images)
        self.instructions.config(text="Step 2: Select and position images on canvas. Use 'Rotate' and 'Resize' buttons for adjustments.")

    def display_images(self, new_images=None):
        if new_images is None:
            new_images = self.images

        for img in new_images:
            img_tk = ImageTk.PhotoImage(img)
            self.image_tk_objects.append(img_tk)  # Keep a reference to prevent garbage collection
            image_item = self.canvas.create_image(50, 50, image=img_tk, anchor=tk.NW)
            self.image_items.append(image_item)
            self.image_objects.append(img)

    def canvas_click(self, event):
        self.canvas.config(cursor="arrow")
        if self.painting or self.erasing:
            self.paint(event)
        else:
            self.select_image(event)
            
    def canvas_drag(self, event):
        self.canvas.config(cursor="arrow")
        if self.painting or self.erasing:
            self.paint(event)
        elif self.cropping:
            self.update_crop(event)
        else:
            self.drag_image(event)

    def select_image(self, event):
        self.canvas.config(cursor="arrow")
        item = self.canvas.find_closest(event.x, event.y)
        if item and item[0] in self.image_items:
            self.selected_image_item = item[0]
            self.instructions.config(text=f"Image selected at ({event.x}, {event.y})")

    def drag_image(self, event):
        self.canvas.config(cursor="arrow")
        if self.selected_image_item:
            self.canvas.coords(self.selected_image_item, event.x, event.y)
            self.canvas.tag_raise(self.selected_image_item)
            
    def start_crop(self):
        self.disable_paint()
        self.cropping = True
        self.canvas.bind("<Button-1>", self.initiate_crop)
        self.canvas.bind("<B1-Motion>", self.update_crop)
        self.canvas.bind("<ButtonRelease-1>", self.end_crop)
        self.instructions.config(text="Cropping mode: Click and drag to select the crop area.")

    def initiate_crop(self, event):
        self.crop_start_x = event.x
        self.crop_start_y = event.y
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)
        self.crop_rect = self.canvas.create_rectangle(self.crop_start_x, self.crop_start_y, event.x, event.y, outline='red', width=3)

    def update_crop(self, event):
        if self.cropping:
            self.canvas.coords(self.crop_rect, self.crop_start_x, self.crop_start_y, event.x, event.y)

    def end_crop(self, event):
        self.cropping = False
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")

        if self.selected_image_item:
            self.instructions.config(text="Adjust the crop area as needed, then click 'Apply Crop' or 'Cancel Crop'.")

        # Re-enable standard mouse events
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)

    def apply_crop(self):
        if self.selected_image_item and self.crop_rect:
            x0, y0, x1, y1 = self.canvas.coords(self.crop_rect)
            index = self.image_items.index(self.selected_image_item)
            img = self.image_objects[index]
            img_width, img_height = img.size

            canvas_coords = self.canvas.coords(self.selected_image_item)
            crop_box = (
                max(0, int(x0 - canvas_coords[0])),
                max(0, int(y0 - canvas_coords[1])),
                min(img_width, int(x1 - canvas_coords[0])),
                min(img_height, int(y1 - canvas_coords[1]))
            )

            if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                cropped_img = img.crop(crop_box)
                self.image_objects[index] = cropped_img
                img_tk = ImageTk.PhotoImage(cropped_img)
                self.image_tk_objects[index] = img_tk
                self.canvas.itemconfig(self.selected_image_item, image=img_tk)

            self.canvas.delete(self.crop_rect)
            self.crop_rect = None

        self.instructions.config(text="Cropping applied. Select another image or apply other adjustments.")

        # Re-enable standard mouse events
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)

    def cancel_crop(self):
        # Remove the crop rectangle
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)
            self.crop_rect = None

        self.instructions.config(text="Cropping canceled. Select another image or apply other adjustments.")

        # Re-enable standard mouse events
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
    
    @staticmethod
    def manual_rotate_image(image, angle):
        # Convert the angle to radians
        angle_rad = math.radians(angle)
        
        # Compute the cosine and sine of the angle
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)
        
        # Get the width and height of the image
        width, height = image.size
        
        # Create the rotation matrix
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        # Compute the new width and height of the rotated image
        corners = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])
        
        new_corners = np.dot(corners, rotation_matrix)
        min_x = min(new_corners[:, 0])
        max_x = max(new_corners[:, 0])
        min_y = min(new_corners[:, 1])
        max_y = max(new_corners[:, 1])
        
        new_width = int(math.ceil(max_x - min_x))
        new_height = int(math.ceil(max_y - min_y))
        
        # Create a new blank image with the computed dimensions
        new_image = Image.new("RGBA", (new_width, new_height), (255, 255, 255, 0))
        pixels = new_image.load()
        
        # Get the center coordinates
        center_x = width // 2
        center_y = height // 2
        new_center_x = new_width // 2
        new_center_y = new_height // 2
        
        # Iterate over the new image
        for y in range(new_height):
            for x in range(new_width):
                # Apply the inverse rotation matrix to get the original coordinates
                original_coords = np.dot(np.array([x - new_center_x, y - new_center_y]), np.linalg.inv(rotation_matrix))
                original_x, original_y = original_coords[0] + center_x, original_coords[1] + center_y
                
                # If the original coordinates are within the bounds of the original image, copy the pixel
                if 0 <= original_x < width and 0 <= original_y < height:
                    original_pixel = image.getpixel((int(original_x), int(original_y)))
                    pixels[x, y] = original_pixel
        
        return new_image

    def rotate_image(self):
        self.disable_paint()
        if self.selected_image_item:
            try:
                angle = float(self.rotate_angle_entry.get())
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid number for the rotation angle.")
                return

            index = self.image_items.index(self.selected_image_item)
            img = self.image_objects[index]

            # Use the manual rotation function instead of img.rotate
            rotated_img = self.manual_rotate_image(img, angle)

            self.image_objects[index] = rotated_img

            img_tk = ImageTk.PhotoImage(rotated_img)
            self.image_tk_objects[index] = img_tk
            self.canvas.itemconfig(self.selected_image_item, image=img_tk)

    

    def resize_image(self):
        self.disable_paint()
        if self.selected_image_item:
            try:
                size = tuple(map(int, self.resize_entry.get().split(',')))
                if len(size) != 2:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid size (width,height).")
                return

            index = self.image_items.index(self.selected_image_item)
            img = self.image_objects[index]
            img = img.resize(size, Image.LANCZOS)
            self.image_objects[index] = img
            img_tk = ImageTk.PhotoImage(img)
            self.image_tk_objects[index] = img_tk
            self.canvas.itemconfig(self.selected_image_item, image=img_tk)
            
    def select_bg_color(self):
        self.disable_paint()
        color_code = colorchooser.askcolor(title="Choose background color")
        if color_code:
            self.bg_color = color_code[1]
            self.bg_image = None
            self.canvas.delete("bg")  # Remove the background image
            self.canvas.config(bg=self.bg_color)
            self.update_canvas()

    def upload_bg_image(self):
        self.disable_paint()
        bg_image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if bg_image_path:
            self.bg_image = Image.open(bg_image_path).convert("RGBA")
            self.bg_color = None
            self.display_bg_image()

    def display_bg_image(self):
        if self.bg_image:
            self.bg_image_tk = ImageTk.PhotoImage(self.bg_image.resize((800, 800)))
            self.canvas.create_image(0, 0, image=self.bg_image_tk, anchor=tk.NW, tags="bg")
            
            # Bring other items to the front (including painted shapes)
            self.canvas.tag_lower("bg")

            #Bring other images to the front
            for item in self.image_items:
                self.canvas.tag_raise(item)
            for c in self.strokes:
                self.canvas.tag_raise(c)
            for t in self.text_items:
                self.canvas.tag_raise(t)
            
            self.update_canvas()
            
    def update_canvas(self):
        # Update the canvas with the buffer
        self.buffer_tk = ImageTk.PhotoImage(self.buffer)
        self.canvas.itemconfig(self.buffer_image_item, image=self.buffer_tk)
        
    def open_filter_window(self):
        self.disable_paint()
        if self.selected_image_item:
            filter_window = Toplevel(self.root)
            filter_window.title("Apply Filter")
            filter_window.transient(self.root)  # Make the window a child of the root window
            filter_window.grab_set()  # Make sure it can receive focus

            # Define the layout of the filter window
            brightness_label = tk.Label(filter_window, text="Brightness:")
            brightness_label.pack()
            brightness_scale = tk.Scale(filter_window, from_=0.0, to=2.0, resolution=0.1, orient=HORIZONTAL, command=self.update_filter_preview)
            brightness_scale.set(1)  # Set initial value to 1 for no change
            brightness_scale.pack()

            contrast_label = tk.Label(filter_window, text="Contrast:")
            contrast_label.pack()
            contrast_scale = tk.Scale(filter_window, from_=0.0, to=2.0, resolution=0.1, orient=HORIZONTAL, command=self.update_filter_preview)
            contrast_scale.set(1)  # Set initial value to 1 for no change
            contrast_scale.pack()

            sharpness_label = tk.Label(filter_window, text="Sharpness:")
            sharpness_label.pack()
            sharpness_scale = tk.Scale(filter_window, from_=0.0, to=2.0, resolution=0.1, orient=HORIZONTAL, command=self.update_filter_preview)
            sharpness_scale.set(1)  # Set initial value to 1 for no change
            sharpness_scale.pack()

            self.current_filters = {
                'brightness': brightness_scale,
                'contrast': contrast_scale,
                'sharpness': sharpness_scale,
            }
            
            

            apply_button = tk.Button(filter_window, text="Apply", command=lambda: self.apply_filter(
                brightness_scale.get(), contrast_scale.get(), sharpness_scale.get(), filter_window))
            apply_button.pack()
            
            reset_button = tk.Button(filter_window, text="Reset", command=self.reset_filter)
            reset_button.pack()

            

            self.original_image = self.image_objects[self.image_items.index(self.selected_image_item)].copy()

    
            
    # Method to manually adjust brightness
    def adjust_brightness(self, image, factor):
        
        # Convert image to array
        image_array = np.array(image, dtype=np.float32)
        # Apply brightness factor
        enhanced_array = np.clip(image_array * factor, 0, 255)
        # Convert array back to image
        return Image.fromarray(np.uint8(enhanced_array))

    # Method to manually adjust contrast
    def adjust_contrast(self, image, factor):
        
        # Convert image to array
        image_array = np.array(image, dtype=np.float32)
        # Calculate mean brightness
        mean = np.mean(image_array, axis=(0, 1), keepdims=True)
        # Apply contrast factor
        enhanced_array = np.clip((image_array - mean) * factor + mean, 0, 255)
        # Convert array back to image
        return Image.fromarray(np.uint8(enhanced_array))

    def gaussian_kernel(self, size, sigma=1):
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)
            ), (size, size)
        )
        return kernel / np.sum(kernel)

    def apply_gaussian_blur(self, image, radius):
        if radius == 0:
            return image

        img_array = np.array(image)
        kernel_size = int(2 * round(radius) + 1)
        kernel = self.gaussian_kernel(kernel_size, radius)
        blurred_img = np.zeros_like(img_array)
        for channel in range(3):
            blurred_img[:, :, channel] = self.convolve2d(img_array[:, :, channel], kernel)
        return Image.fromarray(np.uint8(blurred_img))

    def convolve2d(self, image, kernel):
        kernel_height, kernel_width = kernel.shape
        image_height, image_width = image.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
        convolved_image = np.zeros_like(image)

        for i in range(image_height):
            for j in range(image_width):
                region = padded_image[i:i + kernel_height, j:j + kernel_width]
                convolved_image[i, j] = np.sum(region * kernel)
        return convolved_image

    def adjust_sharpness(self, image, factor):
        
        # Create a blurred version of the image using manual Gaussian blur
        blurred_image = self.apply_gaussian_blur(image, radius=2)
        # Convert images to arrays
        image_array = np.array(image, dtype=np.float32)
        blurred_array = np.array(blurred_image, dtype=np.float32)
        # Apply sharpness factor
        enhanced_array = np.clip(image_array * factor + blurred_array * (1 - factor), 0, 255)
        # Convert array back to image
        return Image.fromarray(np.uint8(enhanced_array))
    
    
    def update_filter_preview(self, event=None):
        self.disable_paint()
        if self.selected_image_item:
            brightness = self.current_filters['brightness'].get()
            contrast = self.current_filters['contrast'].get()
            sharpness = self.current_filters['sharpness'].get()

            index = self.image_items.index(self.selected_image_item)
            img = self.original_image.copy()

            # Apply manual filters
            img = self.adjust_brightness(img, brightness)
            img = self.adjust_contrast(img, contrast)
            img = self.adjust_sharpness(img, sharpness)

            img_tk = ImageTk.PhotoImage(img)
            self.image_tk_objects[index] = img_tk
            self.canvas.itemconfig(self.selected_image_item, image=img_tk)

    def apply_filter(self, brightness, contrast, sharpness, filter_window):
        self.disable_paint()
        if self.selected_image_item:
            index = self.image_items.index(self.selected_image_item)
            img = self.original_image.copy()

            # Apply manual filters
            img = self.adjust_brightness(img, brightness)
            img = self.adjust_contrast(img, contrast)
            img = self.adjust_sharpness(img, sharpness)

            self.image_objects[index] = img
            img_tk = ImageTk.PhotoImage(img)
            self.image_tk_objects[index] = img_tk
            self.canvas.itemconfig(self.selected_image_item, image=img_tk)

            filter_window.destroy()
            self.instructions.config(text="Filter applied successfully!")
            
    

    def remove_background(self):
        if self.selected_image_item:
            index = self.image_items.index(self.selected_image_item)
            img = self.image_objects[index]
            
            # Convert PIL Image to byte stream
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Path to the manually downloaded model
            model_path = "C:\\Users\\ASUS\\.u2net\\u2net.onnx"
            
            # Create a new session with the custom model
            session = new_session(model_name=model_path)
            
            # Use rembg with the manually downloaded model
            result = remove(img_byte_arr, session=session)
            
            # Convert byte stream back to PIL Image
            new_img = Image.open(io.BytesIO(result)).convert("RGBA")
            
            self.image_objects[index] = new_img
            img_tk = ImageTk.PhotoImage(new_img)
            self.image_tk_objects[index] = img_tk
            self.canvas.itemconfig(self.selected_image_item, image=img_tk)
            
            self.instructions.config(text="Background removed using rembg with manual model loading.")
            
    
    def reset_filter(self):
        self.disable_paint()
        if self.selected_image_item and self.original_image:
            index = self.image_items.index(self.selected_image_item)
            self.image_objects[index] = self.original_image
            img_tk = ImageTk.PhotoImage(self.original_image)
            self.image_tk_objects[index] = img_tk
            self.canvas.itemconfig(self.selected_image_item, image=img_tk)

    
    def enable_paint(self):
        self.painting = True
        self.erasing = False
        self.canvas.config(cursor="pencil")
        self.instructions.config(text="Painting mode enabled. Hold and drag the mouse to paint.")
        
        # Re-enable standard mouse events
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)

    def enable_eraser(self):
        self.painting = True
        self.erasing = True
        self.canvas.config(cursor="circle")
        self.instructions.config(text="Eraser mode: Click and drag to erase.")
        
        # Re-enable standard mouse events
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        
    
    def select_paint_color(self):
        self.enable_paint()
        color_code = colorchooser.askcolor(title="Choose paint color")
        if color_code:
            self.paint_color = color_code[1]
            
        # Re-enable standard mouse events
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)

    def paint(self, event):
        if self.painting:
            x1, y1 = (event.x - 1), (event.y - 1)
            x2, y2 = (event.x + 1), (event.y + 1)
            thickness = self.erase_thickness.get() if self.erasing else self.paint_thickness.get()

            if self.erasing:
                color = (255, 255, 255, 0)
                # Draw a transparent circle on the temp buffer
                self.temp_buffer_draw.ellipse((x1-thickness, y1-thickness, x2+thickness, y2+thickness), fill=color, outline=color)
                # Immediately update the canvas with the erased part
                self.update_temp_canvas()
            
            else:
                color = self.paint_color

            self.temp_buffer_draw.ellipse((x1-thickness, y1-thickness, x2+thickness, y2+thickness), fill=color, outline=color)
            
            # Draw directly on the canvas for immediate visual feedback
            #self.canvas.create_oval(x1-thickness, y1-thickness, x2+thickness, y2+thickness, fill=color, outline=color, width=0)
            
            
        

            self.update_temp_canvas()

            self.temp_strokes.append((x1, y1, x2, y2, color, thickness))
            # Draw directly on the canvas for immediate visual feedback
            #self.canvas.create_oval(x1-thickness, y1-thickness, x2+thickness, y2+thickness, fill=color, outline=color, width=0)
            # Bring other images to the front
            #for item in self.image_items:
                #self.canvas.tag_lower(item)
            # Re-enable standard mouse events
            self.canvas.bind("<Button-1>", self.canvas_click)
            self.canvas.bind("<B1-Motion>", self.canvas_drag)
            # Bring the paint buffer to the front
            #self.canvas.tag_raise(self.buffer_image_item)
            
        
        

    def update_temp_canvas(self):
        self.temp_buffer_tk = ImageTk.PhotoImage(self.temp_buffer)
        self.canvas.itemconfig(self.temp_buffer_image_item, image=self.temp_buffer_tk)
        

    def apply_paint(self):
        # Merge temp_buffer with main buffer
        self.buffer = Image.alpha_composite(self.buffer, self.temp_buffer)
        self.buffer_draw = ImageDraw.Draw(self.buffer)

        # Clear temp_buffer
        self.temp_buffer = Image.new("RGBA", (800, 800), (255, 255, 255, 0))
        self.temp_buffer_draw = ImageDraw.Draw(self.temp_buffer)
        self.temp_strokes.clear()

        self.update_canvas()
        self.update_temp_canvas()
        self.disable_paint()
        # Capture the strokes applied for later use in save_collage
        self.strokes.extend(self.temp_strokes)
        # Re-enable standard mouse events
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
            
    def reset_paint(self):
        # Clear all paint strokes
        self.strokes.clear()
        self.temp_strokes.clear()
        self.buffer = Image.new("RGBA", (800, 800), (255, 255, 255, 0))
        self.buffer_draw = ImageDraw.Draw(self.buffer)
        # Reset the temporary buffer
        self.temp_buffer = Image.new("RGBA", (800, 800), (255, 255, 255, 0))
        self.temp_buffer_draw = ImageDraw.Draw(self.temp_buffer)
        self.update_canvas()
        self.disable_paint()
        # Re-enable standard mouse events
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        
        
    
        
    def select_text_color(self):
        color_code = colorchooser.askcolor(title="Choose text color")
        if color_code:
            self.text_color = color_code[1]

    def add_text(self):
        text = self.text_entry.get()
        font_size = self.font_size_entry.get()
        font_name = self.font_choice.get()

        if not text:
            messagebox.showerror("Input Error", "Please enter some text.")
            return

        try:
            font_size = int(font_size)
        except ValueError:
            messagebox.showerror("Input Error", "Font size must be a number.")
            return

        self.text_to_add = text
        self.text_font_name = font_name
        self.text_font_size = font_size

        # Set text color
        self.text_color = self.text_color or "#000000"  # Ensure there's a default value if not set

        print(f"Adding text: '{self.text_to_add}' with font '{self.text_font_name}', size {self.text_font_size}, color {self.text_color}")

        self.instructions.config(text="Click on the canvas to place the text.")

        self.canvas.bind("<Button-1>", self.place_text_on_click)
        
       
        

    def place_text_on_click(self, event):
        x, y = event.x, event.y

        text_font = (self.text_font_name, self.text_font_size)
        text_id = self.canvas.create_text(x, y, text=self.text_to_add, font=text_font, fill=self.text_color)
        self.text_items.append(text_id)

        self.canvas.unbind("<Button-1>")
        self.instructions.config(text="Text placed. Add more text or perform other actions.")
        
        # Re-enable standard mouse events
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)


    def reset_text(self):
        # Remove all text items from the canvas
        for text_id in self.text_items:
            self.canvas.delete(text_id)
        self.text_items.clear()
        self.instructions.config(text="All text items have been reset.")
        # Re-enable standard mouse events
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        
    

    def save_collage(self):
        self.disable_paint()
        try:
            # Create a blank canvas to save all elements, including the background
            canvas_height, canvas_width = 800,800
            collage = np.full((canvas_height, canvas_width, 4), 255, dtype=np.uint8)  # White background with alpha channel

            # Draw the background image if present, else use a solid background color
            if self.bg_image:
                bg_resized = self.bg_image.resize((canvas_width, canvas_height), Image.LANCZOS)
                bg_array = np.array(bg_resized)
                if bg_array.shape[2] == 3:  # Add alpha channel if not present
                    bg_array = np.dstack((bg_array, np.full((bg_array.shape[0], bg_array.shape[1]), 255)))
                collage = cv2.addWeighted(collage, 0, bg_array, 1, 0)
                print("Background image applied.")
            elif self.bg_color:
                # Convert hex to BGR
                bg_color = self.hex_to_bgr(self.bg_color)
                print(f"Background color applied: {bg_color}")
                collage[:, :] = list(bg_color) + [255]  # Apply the background color
            else:
                collage[:, :] = [255, 255, 255, 255]  # Default to white background

            # Draw strokes (including transparency handling)
            for stroke in self.strokes:
                x1, y1, x2, y2, color, thickness = stroke
                color = tuple(map(int, color[:3]))  # Convert color to BGR format for OpenCV
                print(color)
                if color == (255, 255, 255):  # Assuming white is the eraser color
                    color = (0, 0, 0, 0)
                cv2.circle(collage, (x1, y1), thickness, color, -1)

            # Include images (preserving transparency)
            for item, img in zip(self.image_items, self.image_objects):
                x, y = map(int, self.canvas.coords(item))
                img_array = np.array(img)
                if img_array.shape[2] == 3:  # Add alpha channel if not present
                    img_array = np.dstack((img_array, np.full((img_array.shape[0], img_array.shape[1]), 255)))
                img_height, img_width = img_array.shape[:2]

                # Adjust for bounds and dimensions
                if x + img_width > canvas_width:
                    img_width = canvas_width - x
                    img_array = img_array[:, :img_width]
                if y + img_height > canvas_height:
                    img_height = canvas_height - y
                    img_array = img_array[:img_height, :]

                # Check if there are valid image dimensions before proceeding
                if img_width <= 0 or img_height <= 0:
                    print(f"Skipping image with invalid dimensions at ({x}, {y}).")
                    continue

                alpha_s = img_array[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                for c in range(0, 3):
                    collage[y:y+img_height, x:x+img_width, c] = (alpha_s * img_array[:, :, c] +
                                                                 alpha_l * collage[y:y+img_height, x:x+img_width, c])

            # Apply paint strokes from the buffer
            buffer_array = np.array(self.buffer)
            if buffer_array.shape[2] == 4:
                buffer_alpha = buffer_array[:, :, 3] / 255.0
                for c in range(0, 3):
                    collage[:, :, c] = (buffer_alpha * buffer_array[:, :, c] +
                                        (1 - buffer_alpha) * collage[:, :, c])
            else:
                print("Buffer array does not have an alpha channel. Skipping alpha blending for buffer.")

            # Draw text items using OpenCV
            for text_id in self.text_items:
                coords = self.canvas.coords(text_id)
                text = self.canvas.itemcget(text_id, 'text')
                font_name = self.font_choice.get()
                font_size = int(self.font_size_entry.get())
                color = self.canvas.itemcget(text_id, 'fill')
                try:
                    bgr_color = self.hex_to_bgr(color)
                except ValueError:
                    # If color conversion fails, use a default color
                    bgr_color = (0, 0, 0)  # Black color

                # OpenCV font mapping to available fonts
                font_map = {
                    "Helvetica": cv2.FONT_HERSHEY_SIMPLEX,
                    "Courier": cv2.FONT_HERSHEY_COMPLEX,
                    "Arial": cv2.FONT_HERSHEY_COMPLEX,
                    "Times New Roman": cv2.FONT_HERSHEY_COMPLEX,
                    "Comic Sans MS": cv2.FONT_HERSHEY_COMPLEX,
                    "Default": cv2.FONT_HERSHEY_SIMPLEX  # Default fallback
                }

                # Get the OpenCV font type
                font = font_map.get(font_name, cv2.FONT_HERSHEY_SIMPLEX)  # Fallback to a default OpenCV font if not found
                font_scale = font_size / 27.0  # Adjust font scale based on the size
                thickness = max(int(font_size / 13), 1)  # Ensure thickness is at least 1
                
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                # Draw the text using OpenCV
                cv2.putText(collage, text, (int(coords[0] - text_width/2), int(coords[1] + text_height/2)), font, font_scale, bgr_color, thickness, lineType=cv2.LINE_AA)

            save_path = filedialog.askdirectory(parent=self.root, title="Select directory to save collage image")
            if not save_path:
                return

            # Ensure the save directory exists
            os.makedirs(save_path, exist_ok=True)

            # Convert to BGR (for compatibility with JPEG) and save the collage image
            collage_bgr = cv2.cvtColor(collage, cv2.COLOR_RGBA2BGR)

            # Save the collage image
            cv2.imwrite(os.path.join(save_path, 'collage.jpg'), collage_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            messagebox.showinfo("Collage Saved", f"Collage saved as {os.path.join(save_path, 'collage.jpg')}")

        except Exception as e:
            messagebox.showerror("Error Saving Collage", f"An error occurred while saving the collage: {e}")

    
    
    
    def hex_to_bgr(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))   
    
    '''            
    def save_collage(self):
        self.disable_paint()
        try:
            # Create a blank canvas to save all elements, including the background
            collage = Image.new("RGBA", (1000, 800), (255, 255, 255, 0))  # Using RGBA to support transparency
            draw = ImageDraw.Draw(collage)

            # Draw the background image if present, else use a solid background color
            if self.bg_image:
                # Resize background image to fit the canvas
                bg_resized = self.bg_image.resize((1000, 800), Image.LANCZOS)
                collage.paste(bg_resized, (0, 0))
            else:
                draw.rectangle([0, 0, 1000, 800], fill=self.bg_color)

            # Draw strokes (including transparency handling)
            for stroke in self.strokes:
                x1, y1, x2, y2, color, thickness = stroke
                if color == (255, 255, 255, 0):  # Check for transparent color
                    draw.ellipse((x1 - thickness, y1 - thickness, x2 + thickness, y2 + thickness), fill=color, outline=color)
                else:
                    draw.ellipse((x1 - thickness, y1 - thickness, x2 + thickness, y2 + thickness), fill=color, outline=color)

            # Include images (preserving transparency)
            for item, img in zip(self.image_items, self.image_objects):
                x, y = self.canvas.coords(item)
                # Ensure the image is pasted with its transparency preserved
                collage.paste(img, (int(x), int(y)), img)

            # Merge the buffer to the collage
            collage = Image.alpha_composite(collage, self.buffer)
            

            # Draw text items
            for text_id in self.text_items:
                coords = self.canvas.coords(text_id)
                text = self.canvas.itemcget(text_id, 'text')
                font_name = self.canvas.itemcget(text_id, 'font').split()[0] 
                font_size = self.canvas.itemcget(text_id, 'font').split()[-1]
                try:
                    font_size = int(font_size)  # Convert font size to integer
                except ValueError:
                    font_size = 20
                
                
                color = self.canvas.itemcget(text_id, 'fill')

                # Debug logs to verify text properties
                print(f"Text: {text}, Position: {coords}, Font: {font_name}, Size: {font_size}, Color: {color}")

                try:
                    # Apply the correct font
                    if font_name == "Arial":
                        text_font = ImageFont.truetype("arial.ttf", font_size)
                    else:
                        text_font = ImageFont.truetype(f"{font_name}.ttf", font_size)
                except IOError:
                    # If the font is not found, use the default font
                    text_font = ImageFont.load_default()
                    print("Default font is used due to missing font file.")

                # Ensure coordinates are properly adjusted if needed
                if not coords or len(coords) < 2:
                    print("Skipping text placement due to invalid coordinates.")
                    continue

                # Convert coordinates to integer tuples
                coords = (int(coords[0]), int(coords[1]))

                # Draw the text on the collage
                draw.text(coords, text, font=text_font, fill=color)
        
            save_path = filedialog.askdirectory(parent=self.root, title="Select directory to save collage image")
            if not save_path:
                return

            # Ensure the save directory exists
            os.makedirs(save_path, exist_ok=True)
            collage = collage.convert('RGB')

            # Save the collage image
            collage.save(os.path.join(save_path, 'collage.jpg'), quality=95)
            messagebox.showinfo("Collage Saved", f"Collage saved as {os.path.join(save_path, 'collage.jpg')}")
            
        except Exception as e:
            messagebox.showerror("Error Saving Collage", f"An error occurred while saving the collage: {e}")'''
    
            
    
if __name__ == "__main__":
    root = tk.Tk()
    app = CollageMaker(root)
    root.mainloop()
