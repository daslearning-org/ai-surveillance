from kivy.lang import Builder
from kivy.metrics import dp, sp

from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.button import MDFillRoundFlatIconButton


Builder.load_string('''

<CamObjDetBox@MDBoxLayout>:
    orientation: 'vertical'
    spacing: dp(4)

    BoxLayout: # original image
        size_hint_y: 0.4
        id: capture_image
        #Camera:
        #    id: camera
        #    #index: 1
        #    resolution: (640, 480)
        #    play: False

    MDGridLayout: # buttons
        cols: 2
        size_hint_y: 0.1
        spacing: dp(4)
        padding: 14, 0, 14, 0 # left, top, right, bottom

        MDFillRoundFlatIconButton:
            id: btn_capture
            text: "Start"
            icon: "camera"
            font_size: sp(18)
            md_bg_color: 'orange'
            pos_hint: {"center_x": .5, "center_y": .5}
            size_hint_x: 0.7
            on_release: app.capture_n_onnx_detect()

        MDFillRoundFlatIconButton:
            id: btn_reset_cam
            text: "Stop"
            icon: "stop"
            font_size: sp(18)
            md_bg_color: '#333036'
            pos_hint: {"center_x": .5, "center_y": .5}
            size_hint_x: 0.3
            on_release: app.stop_cctv_loop()

    BoxLayout: # converted image
        size_hint_y: 0.5
        id: cam_result_image
        # add result here


''')

class CamObjDetBox(MDBoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "cam_detect_bx"
