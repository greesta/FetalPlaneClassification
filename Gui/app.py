import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

# Lazy TensorFlow loader: import only when needed (avoids long imports at startup)
def import_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        return tf, load_model
    except Exception as e:
        raise


window = tk.Tk()
window.geometry("1366x768")
window.title("Fetal Plane Classification")
window.config(background='white')

icon_path = r"C:\Users\ASUS\OneDrive\Documents\SKRIPSI\GUI FETAL PLANE\icon fetal.png"
try:
    icon = tk.PhotoImage(file=icon_path)
    window.iconphoto(False, icon)
except Exception:
    # If icon can't be loaded, continue without it
    pass


def on_load_model():
    """Example handler that imports TensorFlow only when the user requests to load a model."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        from PIL import Image, ImageTk
        import numpy as np

        # Globals
        _tf = None
        _load_model = None
        MODEL = None
        MODEL_INPUT_SHAPE = None
        class_names = ['Head', 'Abdomen', 'Femur']  # adjust if your model uses different labels


        # Lazy TensorFlow loader: import only when needed (avoids long imports at startup)
        def import_tensorflow():
            global _tf, _load_model
            if _tf is not None and _load_model is not None:
                return _tf, _load_model
            try:
                import tensorflow as tf
                from tensorflow.keras.models import load_model
                _tf = tf
                _load_model = load_model
                return tf, load_model
            except Exception as e:
                raise


        window = tk.Tk()
        window.geometry("1000x700")
        window.title("Fetal Plane Classification")
        window.config(background='white')

        icon_path = r"C:\Users\ASUS\OneDrive\Documents\SKRIPSI\GUI FETAL PLANE\icon fetal.png"
        try:
            icon = tk.PhotoImage(file=icon_path)
            window.iconphoto(False, icon)
        except Exception:
            pass


        # UI: frames
        top_frame = tk.Frame(window, bg='white')
        top_frame.pack(pady=10)

        btn_frame = tk.Frame(window, bg='white')
        btn_frame.pack(pady=10)

        image_frame = tk.Frame(window, bg='white')
        image_frame.pack(pady=10)

        result_frame = tk.Frame(window, bg='white')
        result_frame.pack(pady=10)

        # Image display
        original_img_label = tk.Label(image_frame, text='No image', bg='#f0f0f0', width=60, height=20)
        original_img_label.pack()

        prediction_label = tk.Label(result_frame, text='Prediction: -', font=(None, 16), bg='white')
        prediction_label.pack()


        def on_load_model():
            """Load an HDF5/Keras model. This will import TensorFlow lazily."""
            global MODEL, MODEL_INPUT_SHAPE
            try:
                tf, load_model = import_tensorflow()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import TensorFlow:\n{e}")
                return

            model_path = filedialog.askopenfilename(title="Select model (.h5)", filetypes=[('Keras model', '*.h5;*.hdf5'), ('All files', '*.*')])
            if not model_path:
                return

            try:
                MODEL = load_model(model_path)
                # Try to infer input shape (batch, height, width, channels) or (batch, channels, h, w)
                try:
                    shape = MODEL.input_shape
                    # shape may be like (None, 224, 224, 3)
                    if isinstance(shape, (list, tuple)):
                        if len(shape) == 4:
                            MODEL_INPUT_SHAPE = (shape[1], shape[2])
                except Exception:
                    MODEL_INPUT_SHAPE = None

                messagebox.showinfo("Success", f"Model loaded: {model_path}")
            except Exception as e:
                messagebox.showerror("Load error", f"Failed to load model:\n{e}")


        def on_load_image():
            path = filedialog.askopenfilename(title='Select image', filetypes=[('Image', '*.png;*.jpg;*.jpeg;*.bmp'), ('All', '*.*')])
            if not path:
                return
            try:
                img = Image.open(path).convert('RGB')
                # keep original for display
                disp = img.copy()
                disp.thumbnail((480, 480))
                imgtk = ImageTk.PhotoImage(disp)
                original_img_label.configure(image=imgtk)
                original_img_label.image = imgtk
                # store path for prediction
                original_img_label.img_path = path
            except Exception as e:
                messagebox.showerror('Image error', f'Failed to open image:\n{e}')


        def preprocess_image_for_model(img_path, target_size=None):
            # Loads image from path, resizes to target_size (h,w), returns float32 array ready for model
            img = Image.open(img_path).convert('RGB')
            if target_size is None:
                target_size = (224, 224)  # assumption: default to 224x224 if model doesn't specify
            img = img.resize((target_size[1], target_size[0]))
            arr = np.array(img).astype('float32') / 255.0
            return arr


        def on_predict():
            global MODEL
            if MODEL is None:
                messagebox.showwarning('No model', 'Please load a model first.')
                return
            img_path = getattr(original_img_label, 'img_path', None)
            if not img_path:
                messagebox.showwarning('No image', 'Please load an image first.')
                return

            try:
                # determine input size
                input_size = None
                try:
                    if MODEL_INPUT_SHAPE is not None:
                        input_size = MODEL_INPUT_SHAPE
                    else:
                        # try to read from model input shape
                        shape = MODEL.input_shape
                        if isinstance(shape, (list, tuple)) and len(shape) >= 4:
                            input_size = (shape[1], shape[2])
                except Exception:
                    input_size = None

                arr = preprocess_image_for_model(img_path, target_size=input_size)
                # expand dims
                x = np.expand_dims(arr, axis=0)

                try:
                    # use TensorFlow for softmax if available
                    tf = _tf
                    preds = MODEL.predict(x)
                    # preds may already be probabilities or logits
                    if preds.ndim == 2 and preds.shape[1] > 1:
                        # multiclass
                        probs = tf.nn.softmax(preds[0]).numpy()
                        idx = int(np.argmax(probs))
                        conf = float(probs[idx])
                        label = class_names[idx] if idx < len(class_names) else str(idx)
                        prediction_label.config(text=f'Prediction: {label} ({conf*100:.1f}%)')
                    else:
                        # single output
                        val = float(preds.flatten()[0])
                        prediction_label.config(text=f'Prediction value: {val:.4f}')
                except Exception as e:
                    messagebox.showerror('Predict error', f'Prediction failed:\n{e}')
            except Exception as e:
                messagebox.showerror('Error', f'Processing failed:\n{e}')


        # Buttons
        load_btn = tk.Button(btn_frame, text="Load Model (.h5)", command=on_load_model)
        load_btn.grid(row=0, column=0, padx=8)

        img_btn = tk.Button(btn_frame, text="Load Image", command=on_load_image)
        img_btn.grid(row=0, column=1, padx=8)

        pred_btn = tk.Button(btn_frame, text="Predict", command=on_predict)
        pred_btn.grid(row=0, column=2, padx=8)

        window.mainloop()
