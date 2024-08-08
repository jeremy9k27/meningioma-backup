import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
from datetime import datetime

class ScanTypeCleanupApp:
    def __init__(self, root, data, output_dir):
        self.root = root
        self.data = data
        self.output_dir = output_dir
        self.options = [
            '----', 'AX_2D_T2', 'AX_3D_T1_POST', 'AX_3D_T1_PRE',
            'AX_ADC', 'AX_DIFFUSION', 'AX_PD', 'AX_SWI', 'AX_STIR',
            'SAG_3D_FLAIR', 'SAG_3D_T2',
            'DISCARD', 'OTHER'
        ]
        self.current_index = 0
        self.responses = []
        self.username = ""

        self.root.title("Image Text Input")

        # Create UI elements for the landing page
        self.instructions_label = tk.Label(root, text="Welcome to the Image Text Input App!\n\nPlease enter your name below and click 'Start' to begin.\nYou will be shown an image and a description, and you need to select your response from the drop-down menu.", wraplength=400)
        self.instructions_label.pack(pady=20)

        self.name_label = tk.Label(root, text="Your Name:")
        self.name_label.pack()

        self.name_entry = tk.Entry(root, width=50)
        self.name_entry.pack()
        self.name_entry.bind("<Return>", self.start_app)

        self.start_button = ttk.Button(root, text="Start", command=self.start_app)
        self.start_button.pack(pady=10)

        # Create UI elements for the main app, but keep them hidden initially
        self.content_frame = tk.Frame(root)

        # Progress label
        self.progress_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.progress_label.pack(pady=10)

        self.image_label = tk.Label(self.content_frame)
        self.image_label.pack(side=tk.LEFT)

        self.text_label = tk.Label(self.content_frame, text="", wraplength=400, justify='left', anchor='w')
        self.text_label.pack(side=tk.LEFT, padx=(10, 0))  # Add padding to the right of the image

        self.response_var = tk.StringVar(root)
        self.response_menu = ttk.OptionMenu(root, self.response_var, self.options[0], *self.options, command=self.check_other)
        self.response_entry = tk.Entry(root, width=50)
        self.response_entry.bind("<Return>", self.save_and_next)
        self.continue_button = ttk.Button(root, text="Continue", command=self.save_and_next)

    def start_app(self, event=None):
        self.username = self.name_entry.get().strip()
        if not self.username:
            messagebox.showwarning("Input Error", "Please enter your name to proceed.")
            return

        self.instructions_label.pack_forget()
        self.name_label.pack_forget()
        self.name_entry.pack_forget()
        self.start_button.pack_forget()

        self.content_frame.pack(padx=20, pady=10, anchor='w')
        self.response_menu.pack(anchor='w')
        self.continue_button.pack(anchor='w')

        self.update_progress()
        self.load_image_text()

    def load_image_text(self):
        if self.current_index < len(self.data):
            item = self.data[self.current_index]
            if item['image_path'] is not None:
                img = Image.open(item['image_path'])
                photo = ImageTk.PhotoImage(img)

                self.image_label.config(image=photo)
                self.image_label.image = photo

            self.text_label.config(text=item['text'])

            self.response_var.set(self.options[0])
            self.response_entry.pack_forget()
            self.root.bind("<Return>", self.save_and_next)  # Bind Enter key to save_and_next
            self.update_progress()
        else:
            self.finish()

    def check_other(self, value):
        if value == "OTHER":
            self.response_entry.pack(anchor='w')
            self.root.bind("<Return>", self.save_and_next)  # Bind Enter key to save_and_next when OTHER is selected
        else:
            self.response_entry.pack_forget()

    def save_and_next(self, event=None):
        if self.response_var.get() == "OTHER":
            user_input = self.response_entry.get()
        else:
            user_input = self.response_var.get()

        if not user_input:
            messagebox.showwarning("Input Error", "Please enter a response to proceed.")
            return

        item = self.data[self.current_index]
        self.responses.append({'image_id': item['id'], 'text': user_input})
        self.current_index += 1
        self.load_image_text()

    def update_progress(self):
        total = len(self.data)
        current = self.current_index + 1
        self.progress_label.config(text=f"Scan {current}/{total}")

    def finish(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/handchecked_{self.username}_{timestamp}.csv"
        pd.DataFrame(self.responses).to_csv(filename, index=False)
        messagebox.showinfo("Finished", f"Thank you for your responses! They have been saved to {filename}.")
        self.root.quit()