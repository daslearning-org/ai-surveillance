# python core modules
import os
os.environ['KIVY_GL_BACKEND'] = 'sdl2'
import sys
from threading import Thread, Event
import queue
import requests
import time
import datetime
import numpy as np
import cv2
from onnxruntime import InferenceSession

from kivy.lang import Builder
from kivy.properties import StringProperty, NumericProperty, ObjectProperty
from kivy.core.window import Window
from kivy.metrics import dp, sp
from kivy.utils import platform
from kivy.uix.image import Image
from kivy.uix.camera import Camera
from kivy.clock import Clock

from kivymd.app import MDApp
from kivymd.uix.navigationdrawer import MDNavigationDrawerMenu
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.label import MDLabel
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton, MDFloatingActionButton

# IMPORTANT: Set this property for keyboard behavior
Window.softinput_mode = "below_target"

# Import your local screen classes & modules
from screens.cam_obj_detect import CamObjDetBox
from screens.setting import SettingsBox
#from onnx_detect import OnnxDetect

## Global definitions
__version__ = "0.0.1" # The APP version

FREQUENCY_HZ = 1.0 # using 10fps detection can be increated upto original camera fps
PERIOD_SECONDS = 1.0 / FREQUENCY_HZ

detect_model_url = "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx"
# Determine the base path for your application's resources
if getattr(sys, 'frozen', False):
    # Running as a PyInstaller bundle
    base_path = sys._MEIPASS
else:
    # Running in a normal Python environment
    base_path = os.path.dirname(os.path.abspath(__file__))
kv_file_path = os.path.join(base_path, 'main_layout.kv')

## define custom kivymd classes
class ContentNavigationDrawer(MDNavigationDrawerMenu):
    screen_manager = ObjectProperty()
    nav_drawer = ObjectProperty()


## Main App
class AiCctvApp(MDApp):
    is_downloading = ObjectProperty(None)
    onnx_detect = ObjectProperty(None)
    cam_found = ObjectProperty(None)
    camera = ObjectProperty(None)
    detect_model_path = StringProperty("")
    sess = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #Window.bind(on_keyboard=self.events)
        self.process = None
        self.img_queue = queue.Queue()

    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.accent_palette = "Orange"
        return Builder.load_file(kv_file_path)

    def on_start(self):
        # paths setup
        if platform == "android":
            from android.permissions import request_permissions, Permission
            from jnius import autoclass, PythonJavaClass, java_method
            sdk_version = 28
            try:
                VERSION = autoclass('android.os.Build$VERSION')
                sdk_version = VERSION.SDK_INT
                print(f"Android SDK: {sdk_version}")
            except Exception as e:
                print(f"Could not check the android SDK version: {e}")
            permissions = [Permission.CAMERA]
            if sdk_version >= 33:  # Android 13+
                permissions.append(Permission.READ_MEDIA_IMAGES)
            else:  # Android 10–12
                permissions.extend([Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE])
            request_permissions(permissions)
            context = autoclass('org.kivy.android.PythonActivity').mActivity
            android_path = context.getExternalFilesDir(None).getAbsolutePath()
            self.model_dir = os.path.join(android_path, 'model_files')
            self.op_dir = os.path.join(android_path, 'outputs')
            self.internal_storage = android_path
            try:
                Environment = autoclass("android.os.Environment")
                self.external_storage = Environment.getExternalStorageDirectory().getAbsolutePath()
            except Exception:
                self.external_storage = os.path.abspath("/storage/emulated/0/")
        else:
            self.internal_storage = os.path.expanduser("~")
            self.external_storage = os.path.expanduser("~")
            self.model_dir = os.path.join(self.user_data_dir, 'model_files')
            self.op_dir = os.path.join(self.user_data_dir, 'outputs')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.op_dir, exist_ok=True)
        self.detect_model_path = os.path.join(self.model_dir, "ssd_mobilenet_v1_10.onnx")

        if not os.path.exists(self.detect_model_path):
            self.popup_detect_model()
        print("Initialisation is successfull")

    def show_toast_msg(self, message, is_error=False):
        from kivymd.uix.snackbar import MDSnackbar
        bg_color = (0.2, 0.6, 0.2, 1) if not is_error else (0.8, 0.2, 0.2, 1)
        MDSnackbar(
            MDLabel(
                text = message,
                font_style = "Subtitle1"
            ),
            md_bg_color=bg_color,
            y=dp(24),
            pos_hint={"center_x": 0.5},
            duration=3
        ).open()

    def show_text_dialog(self, title, text="", buttons=[]):
        self.txt_dialog = MDDialog(
            title=title,
            text=text,
            buttons=buttons
        )
        self.txt_dialog.open()

    def txt_dialog_closer(self, instance):
        if self.txt_dialog:
            self.txt_dialog.dismiss()

    def popup_detect_model(self):
        buttons = [
            MDFlatButton(
                text="Cancel",
                theme_text_color="Custom",
                text_color=self.theme_cls.primary_color,
                on_release=self.txt_dialog_closer
            ),
            MDFlatButton(
                text="Ok",
                theme_text_color="Custom",
                text_color="green",
                on_release=self.download_detect_model
            ),
        ]
        self.show_text_dialog(
            "Downlaod the model file",
            f"You need to downlaod the file for the first time (~30MB)",
            buttons
        )

    def download_detect_model(self, instance):
        self.download_model_file(detect_model_url, self.detect_model_path, instance)

    def download_model_file(self, model_url, download_path, instance=None):
        self.txt_dialog_closer(instance)
        filename = download_path.split("/")[-1]
        print(f"Starting the download for: {filename}")
        result_box = self.root.ids.cam_detect_box.ids.cam_result_image
        result_box.clear_widgets()
        self.download_progress = MDLabel(
            text="Progress: 0%",
            halign="center"
        )
        result_box.add_widget(self.download_progress)
        Thread(target=self.download_file, args=(model_url, download_path), daemon=True).start()

    def download_file(self, download_url, download_path):
        filename = download_url.split("/")[-1]
        try:
            self.is_downloading = filename
            with requests.get(download_url, stream=True) as req:
                req.raise_for_status()
                total_size = int(req.headers.get('content-length', 0))
                downloaded = 0
                with open(download_path, 'wb') as f:
                    for chunk in req.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            Clock.schedule_once(lambda dt: self.update_download_progress(downloaded, total_size))
            if os.path.exists(download_path):
                Clock.schedule_once(lambda dt: self.show_toast_msg(f"Download complete: {download_path}"))
            else:
                Clock.schedule_once(lambda dt: self.show_toast_msg(f"Download failed for: {download_path}", is_error=True))
            self.is_downloading = False
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the onnx file: {e} 😞")
            Clock.schedule_once(lambda dt: self.show_toast_msg(f"Download failed for: {download_path}", is_error=True))
            self.is_downloading = False

    def update_download_progress(self, downloaded, total_size):
        if total_size > 0:
            percentage = (downloaded / total_size) * 100
            self.download_progress.text = f"Progress: {percentage:.1f}%"
        else:
            self.download_progress.text = f"Progress: {downloaded} bytes"

    def on_cam_obj_detect(self):
        if not os.path.exists(self.detect_model_path) and self.is_downloading != "ssd_mobilenet_v1_10.onnx":
            self.popup_detect_model()
        self.show_toast_msg("Start your AI powered CCTV on phone!")
        self.cam_uix = self.root.ids.cam_detect_box.ids.capture_image
        self.cam_uix.clear_widgets()
        if platform == "android":
            cam_indx = 0
            resolution = (960, 720)
        else:
            resolution = (640, 480)
            available_cameras = []
            for i in range(3): # increase the numbers if your desktop / server has more cameras
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"Camera found at index: {i}")
                    available_cameras.append(i)
                    cap.release()
            if len(available_cameras) >= 1:
                cam_indx = available_cameras[0] # starts the first camera if available, change as needed
            else:
                self.show_toast_msg(f"No camera found on {platform}!", is_error=True)
                self.cam_found = False
                return
        try:
            self.camera = Camera(
                index = cam_indx,
                resolution = resolution,
                fit_mode = "contain",
                play = True
            )
            self.cam_uix.add_widget(self.camera)
            self.cam_found = True
            self.camera.bind(texture=self.on_texture_change)
        except Exception as e:
            print(f"Error setting up the camera: {e}")
            self.show_toast_msg(f"Error setting up the camera: {e}", is_error=True)
            self.cam_found = False

    def on_texture_change(self, instance, value):
        # Called when the texture property changes (e.g., when camera initializes)
        print("Texture changed or initialized:", value)
        if value:
            # Texture is available; you can start processing frames
            self.start_frame_processing()

    def start_frame_processing(self):
        # Schedule frame capture on the main thread
        Clock.schedule_interval(self.process_frame, 0.033)

    def process_frame(self, dt):
        if not self.camera or not self.camera.texture:
            return
        try:
            texture = self.camera.texture
            pixels = texture.pixels
            width, height = texture.size
            arr = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 4))
            #arr = np.flipud(arr)  # Flip if needed
            img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            # Process the frame (e.g., save or analyze)
            self.img_queue.put(img)
            #print("Frame captured...")
        except Exception as e:
            print(f"Error processing frame: {e}")

    def on_cam_obj_dt_leave(self):
        if self.cam_found:
            self.camera.play = False
            self.cam_uix.clear_widgets()
        if self.process:
            self.process = False

    def detection_loop(self):
        """
        This method runs a controlable contineous loop thread, it detects if there is human
        """
        while self.process:
            # do the detection
            try:
                img = self.img_queue.get(timeout=1)
                original_height, original_width = img.shape[:2]
                # Resize to 300x300 for model input, keep as RGB uint8
                img_resized = cv2.resize(img, (300, 300))
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                # Add batch dimension: shape (1, 300, 300, 3), keep uint8
                img_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)
                # Run inference
                try:
                    results = self.sess.run(self.output_names, {self.input_name: img_data})
                    # Parse outputs
                    detection_boxes = results[0][0]  # Shape (100, 4): [y1, x1, y2, x2] normalized [0,1]
                    detection_classes = results[1][0]  # Shape (100,): class indices
                    detection_scores = results[2][0]  # Shape (100,): confidence scores
                    num_detections = results[3]  # Shape (1,): number of detections
                    # Extract num_detections as scalar
                    if num_detections.size == 1:
                        num_detections = int(num_detections.item())
                        # Filter detections by score threshold and draw boxes on original image
                        threshold = 0.5
                        for i in range(min(num_detections, len(detection_scores))):
                            score = detection_scores[i]
                            if score > threshold:
                                class_id = int(detection_classes[i])
                                if class_id == 1:
                                    print("Human found") # use a flag instead & break
                    else:
                        print(f"Error: Unexpected num_detections shape {num_detections.shape}")
                except Exception as e:
                    print(f"Inference error: {e}")
            except queue.Empty:
                continue # continue playing or simple loop through
            except Exception as e:
                print(f"Queue error: {e}")

    def start_detect_session(self):
        try:
            self.sess = InferenceSession(self.detect_model_path)
            # Get input and output names
            self.input_name = self.sess.get_inputs()[0].name
            self.output_names = [o.name for o in self.sess.get_outputs()]
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def capture_n_onnx_detect(self):
        if not self.cam_found:
            self.show_toast_msg("Camera could not be loaded!", is_error=True)
            return
        if self.is_downloading == "ssd_mobilenet_v1_10.onnx":
            self.show_toast_msg("Please wait for the model download to finish!", is_error=True)
            return
        if not os.path.exists(self.detect_model_path) and self.is_downloading != "ssd_mobilenet_v1_10.onnx":
            self.sess = False
            self.popup_detect_model()
            return
        if not self.sess:
            self.start_detect_session()
        # thread
        self.process = True
        Thread(target=self.detection_loop, daemon=True).start()

    def onnx_detect_callback(self, onnx_resp):
        status = onnx_resp["status"]
        message = onnx_resp["message"]
        caller = onnx_resp["caller"]
        self.is_detect_running = False
        result_box = self.root.ids.cam_detect_box.ids.cam_result_image
        if status is True:
            self.show_toast_msg(f"Output generated at: {message}")
            self.op_img_path = message
            result_box.clear_widgets()
            fitImage = Image(
                source = message,
                fit_mode = "contain"
            )
            result_box.add_widget(fitImage)
            down_btn = MDFloatingActionButton(
                icon="download",
                type="small",
                theme_icon_color="Custom",
                md_bg_color='#e9dff7',
                icon_color='#211c29',
            )
            down_btn.bind(on_release=self.open_op_file_manager)
            result_box.add_widget(down_btn)
        else:
            self.show_toast_msg(message, is_error=True)

if __name__ == '__main__':
    AiCctvApp().run()
