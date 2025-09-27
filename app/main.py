# python core modules
import os
os.environ['KIVY_GL_BACKEND'] = 'sdl2'
import sys
from threading import Thread
import multiprocessing
import requests
import time
import numpy as np
import cv2

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
from onnx_detect import OnnxDetect

## Global definitions
__version__ = "0.0.1" # The APP version

FREQUENCY_HZ = 10.0 # using 10fps detection can be increated upto original camera fps
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
    onnx_detect_sess = ObjectProperty(None)
    cam_found = ObjectProperty(None)
    camera = ObjectProperty(None)
    detect_model_path = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #Window.bind(on_keyboard=self.events)
        manager = multiprocessing.Manager()
        self.shared_dict = manager.dict()  # shared dictionary
        self.loop_running = multiprocessing.Value('b', True)
        self.process = None

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

        # create onnx objects
        self.onnx_detect = OnnxDetect(
            save_dir=self.op_dir,
            model_dir=self.model_dir,
        )
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
        except Exception as e:
            print(f"Error setting up the camera: {e}")
            self.show_toast_msg(f"Error setting up the camera: {e}", is_error=True)

    def on_cam_obj_dt_leave(self):
        if self.cam_found:
            self.camera.play = False
            self.cam_uix.clear_widgets()
        if self.process:
            self.loop_running.value = False
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
            self.process = None

    def detection_loop(self):
        """
        This method runs a controlable contineous loop via multi process
        """
        next_time = time.perf_counter() + PERIOD_SECONDS
        while self.loop_running.value:
            sleep_duration = next_time - time.perf_counter()

            ## Kivy camera capture process
            texture = self.camera.texture
            # Get raw bytes (typically in 'rgba' format)
            pixels = texture.pixels
            width, height = texture.size
            # Convert to numpy array (height x width x 4 for RGBA)
            arr = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 4))
            # Convert to BGR for OpenCV
            img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

            # send it to detect with callback
            onnx_thread = Thread(target=self.onnx_detect.run_detect, args=(img, self.onnx_detect_callback, "camObjDetect"), daemon=True)
            onnx_thread.start()

            # not to bombard the processor
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            next_time += PERIOD_SECONDS

    def capture_n_onnx_detect(self):
        if not self.cam_found:
            self.show_toast_msg("Camera could not be loaded!", is_error=True)
            return
        if self.is_downloading == "ssd_mobilenet_v1_10.onnx":
            self.show_toast_msg("Please wait for the model download to finish!", is_error=True)
            return
        if not os.path.exists(self.detect_model_path) and self.is_downloading != "ssd_mobilenet_v1_10.onnx":
            self.onnx_detect_sess = False
            self.popup_detect_model()
            return
        if not self.onnx_detect_sess:
            self.onnx_detect_sess = self.onnx_detect.start_detect_session()
            self.capture_n_onnx_detect()
        self.process = multiprocessing.Process(target=self.detection_loop)
        self.process.start()

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
