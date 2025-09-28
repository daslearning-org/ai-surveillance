# python core modules
import os
os.environ['KIVY_GL_BACKEND'] = 'sdl2'
import sys
from threading import Thread
import queue
import requests
import time
import datetime
import json

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

if platform == "android":
    from jnius import autoclass

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
from screens.init_screen import ConfigInput

## Global definitions
__version__ = "0.0.1" # The APP version

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
    last_detect_time = ObjectProperty()
    config_data = ObjectProperty()
    sms_send_count = NumericProperty(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #Window.bind(on_keyboard=self.events)
        self.process = None
        self.is_loop_started = False
        self.img_queue = queue.Queue()
        self.sms_queue = queue.Queue()

    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.accent_palette = "Orange"
        return Builder.load_file(kv_file_path)

    def on_start(self):
        # paths setup
        if platform == "android":
            from android.permissions import request_permissions, Permission
            sdk_version = 28
            try:
                VERSION = autoclass('android.os.Build$VERSION')
                sdk_version = VERSION.SDK_INT
                print(f"Android SDK: {sdk_version}")
            except Exception as e:
                print(f"Could not check the android SDK version: {e}")
            permissions = [Permission.CAMERA, Permission.SEND_SMS, Permission.WAKE_LOCK]
            if sdk_version >= 33:  # Android 13+
                permissions.append(Permission.READ_MEDIA_IMAGES)
            else:  # Android 9–12
                permissions.extend([Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE])
            request_permissions(permissions)
            context = autoclass('org.kivy.android.PythonActivity').mActivity
            android_path = context.getExternalFilesDir(None).getAbsolutePath()
            self.model_dir = os.path.join(android_path, 'model_files')
            self.op_dir = os.path.join(android_path, 'outputs')
            config_dir = os.path.join(android_path, 'config')
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
            config_dir = os.path.join(self.user_data_dir, 'config')
            self.op_dir = os.path.join(self.user_data_dir, 'outputs')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.op_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        self.config_path = os.path.join(config_dir, 'config.json')
        self.detect_model_path = os.path.join(self.model_dir, "ssd_mobilenet_v1_10.onnx")
        Window.keep_screen_on = True

        # check if config exists with a valid phone number
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config_data = json.load(f)
            if len(self.config_data.get('phone', 'na')) >= 5:
                self.root.ids.screen_manager.current = "camObjDetect"

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

    def show_custom_dialog(self, title, custom_class, buttons=[]):
        self.custom_dialog = MDDialog(
            title=title,
            type="custom",
            content_cls=custom_class,
            buttons=buttons
        )
        self.custom_dialog.open()
        print((self.custom_dialog.children))

    def custom_dialog_closer(self, instance):
        if self.custom_dialog:
            self.custom_dialog.dismiss()

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
        # check if config exists with a valid phone number
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config_data = json.load(f)
            if len(self.config_data.get('phone', 'na')) >= 5:
                self.root.ids.screen_manager.current = "camObjDetect"
            else:
                self.change_sms_number()
        else:
            self.change_sms_number()

        # check if the models exists
        if not os.path.exists(self.detect_model_path) and self.is_downloading != "ssd_mobilenet_v1_10.onnx":
            self.popup_detect_model()
        self.show_toast_msg("Start your AI powered CCTV on phone!")

        # set the Camera widget
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
        Clock.schedule_interval(self.process_frame, 0.2)

    def process_frame(self, dt):
        if not self.camera or not self.camera.texture or not self.process:
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
            #print(f"Frame captured") # debug
        except Exception as e:
            print(f"Error processing frame: {e}")

    def on_cam_obj_dt_leave(self):
        if self.cam_found:
            try:
                self.camera.play = False
            except Exception as e:
                print(f"Cam stop error: {e}")
            self.cam_uix.clear_widgets()
            self.camera = False
        if self.process:
            self.process = False

    def send_sms(self, img_path):
        if platform == "android" and self.sms_send_count< 2:
            # currently sending only for two times per session, will change it to some time basis interval
            SmsManager = autoclass('android.telephony.SmsManager')
            sms_manager = SmsManager.getDefault()
            msg = f"Human detected: {img_path}"
            phone = self.config_data['phone']
            try:
                sms_manager.sendTextMessage(phone, None, msg, None, None)
                print(f"✅ SMS sent to {phone}")
                self.show_toast_msg(f"✅ SMS sent to {phone}")
                self.sms_send_count += 1
            except Exception as e:
                print("SMS ❌ Failed:", e)
                self.show_toast_msg(f"sms failed: {e}", is_error=True)
        else:
            print("This works only on Android!")

    def sms_loop(self):
        """
        This method runs a controlable contineous loop thread, it detects if there is human
        """
        while self.process:
            detect_flag = False
            try:
                img_path = self.sms_queue.get(timeout=0.2)
                img_path = str(img_path)
                print(f"Detected: {img_path}") # debug
                Thread(target=self.send_sms, args=(img_path,), daemon=True).start()
            except queue.Empty:
                continue # continue playing or simple loop through
            except Exception as e:
                print(f"SMS Queue error: {e}")

    def detection_loop(self):
        """
        This method runs a controlable contineous loop thread, it detects if there is human
        """
        detect_count = 0
        while self.process:
            detect_flag = False
            now = datetime.datetime.now()
            current_time = str(now.strftime("%H%M%S"))
            current_date = str(now.strftime("%Y%m%d"))
            image_filename = f"cam-{current_date}-{current_time}.png"
            op_img_path = os.path.join(self.op_dir, image_filename)
            # do the detection
            try:
                img = self.img_queue.get(timeout=0.2)
                #print("In detection loop") # debug
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
                                    detect_flag = True
                                    box = detection_boxes[i]
                                    # Scale boxes to original image size
                                    y1 = int(box[0] * original_height)
                                    x1 = int(box[1] * original_width)
                                    y2 = int(box[2] * original_height)
                                    x2 = int(box[3] * original_width)
                                    percent = int(score*100)
                                    # Prepare original image for drawing (convert to RGB then back to BGR for OpenCV)
                                    output_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert original image to RGB
                                    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)  # Convert back to BGR
                                    # Draw rectangle and label
                                    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(output_img, f"person: {percent}%", (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                        if detect_flag:
                            detect_count += 1
                        if detect_count >= 5:
                            # if detection happens for atleast 5 framse i.e. 1/2 sec
                            cv2.imwrite(op_img_path, output_img)
                            self.sms_queue.put(op_img_path)
                            detect_count = 0
                    else:
                        print(f"Error: Unexpected num_detections shape {num_detections.shape}")
                except Exception as e:
                    print(f"Inference error: {e}")
            except queue.Empty:
                continue # continue playing or simple loop through
            except Exception as e:
                print(f"Detect Queue error: {e}")

    def start_detect_session(self):
        try:
            self.sess = InferenceSession(self.detect_model_path)
            # Get input and output names
            self.input_name = self.sess.get_inputs()[0].name
            self.output_names = [o.name for o in self.sess.get_outputs()]
            print("Onnx session started")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def capture_n_onnx_detect(self):
        if self.process:
            self.show_toast_msg("Capture session is already started!")
            return
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
        start_process = self.start_detect_session()
        # thread
        if start_process and not self.process:
            self.process = True
            Thread(target=self.detection_loop, daemon=True).start()
            Thread(target=self.sms_loop, daemon=True).start()

    def stop_cctv_loop(self):
        self.process = False

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

    ## Settings section
    def change_sms_number(self):
        self.root.ids.screen_manager.current = "initScreen"

    def save_config(self, instance, input_widget):
        phone_num = input_widget.text.strip()
        if len(phone_num) <= 4:
            self.show_toast_msg("Phone number should be 5 digits or more!", is_error=True)
            self.change_sms_number()
            return
        self.config_data = {'phone': phone_num}
        with open(self.config_path, 'w') as f:
            json.dump(self.config_data, f, indent=4)
        self.root.ids.screen_manager.current = "camObjDetect"

    def open_link(self, instance, url):
        import webbrowser
        webbrowser.open(url)

    def update_link_open(self, instance):
        self.txt_dialog_closer(instance)
        self.open_link(instance=instance, url="https://github.com/daslearning-org/ai-surveillance/releases")

    def update_checker(self, instance):
        buttons = [
            MDFlatButton(
                text="Cancel",
                theme_text_color="Custom",
                text_color=self.theme_cls.primary_color,
                on_release=self.txt_dialog_closer
            ),
            MDFlatButton(
                text="Releases",
                theme_text_color="Custom",
                text_color="green",
                on_release=self.update_link_open
            ),
        ]
        self.show_text_dialog(
            "Check for update",
            f"Your version: {__version__}",
            buttons
        )

    def show_delete_alert(self):
        op_img_count = 0
        for filename in os.listdir(self.op_dir):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                op_img_count += 1
        self.show_text_dialog(
            title="Delete all output files?",
            text=f"There are total: {op_img_count} image files. This action cannot be undone!",
            buttons=[
                MDFlatButton(
                    text="CANCEL",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_release=self.txt_dialog_closer
                ),
                MDFlatButton(
                    text="DELETE",
                    theme_text_color="Custom",
                    text_color="red",
                    on_release=self.delete_op_action
                ),
            ],
        )

    def delete_op_action(self, instance):
        # Custom function called when DISCARD is clicked
        for filename in os.listdir(self.op_dir):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                file_path = os.path.join(self.op_dir, filename)
                try:
                    os.unlink(file_path)
                    print(f"Deleted {file_path}")
                except Exception as e:
                    print(f"Could not delete the audion files, error: {e}")
        self.show_toast_msg("Executed the audio cleanup!")
        self.txt_dialog_closer(instance)

    ## run on app exit
    def on_stop(self):
        self.process = False
        try:
            self.camera.play = False
        except Exception as e:
            print(f"Cam stop error: {e}")
        self.camera = False

if __name__ == '__main__':
    AiCctvApp().run()
