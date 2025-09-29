from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.textfield import MDTextField
from kivymd.uix.button import MDFillRoundFlatIconButton
from kivymd.uix.label import MDLabel

from kivy.lang import Builder
from kivy.properties import StringProperty, NumericProperty, ObjectProperty
from kivy.metrics import dp, sp
from kivy.uix.widget import Widget

Builder.load_string('''
<ConfigInput@MDBoxLayout>:
    orientation: 'vertical'
    spacing: dp(30)
    padding: dp(10)
    #adaptive_height: True
    MDGridLayout:
        cols: 2
        adaptive_height: True
        MDLabel:
            text: "SMS number"
            halign: "left"
            font_size: sp(18)
            size_hint_x: 0.2
        MDTextField:
            id: phone_num
            hint_text: "Enter the number"
            mode: "rectangle"
            helper_text: "ex: +919876543210"
            helper_text_mode: "persistent"
            size_hint_x: 0.8
            font_size: sp(18)
            multiline: False
            required: True
    MDGridLayout:
        cols: 2
        adaptive_height: True
        MDLabel:
            text: "SMS frequency (min)"
            halign: "left"
            font_size: sp(18)
            size_hint_x: 0.2
        MDTextField:
            id: sms_freq_input
            text: "10"
            hint_text: "number of minutes"
            mode: "rectangle"
            helper_text: "default: 10"
            helper_text_mode: "persistent"
            size_hint_x: 0.8
            font_size: sp(18)
            multiline: False
            input_filter: 'int'
    MDFillRoundFlatIconButton:
        text: "Save"
        icon: "floppy"
        pos_hint: {'center_x': 0.5}
        size_hint_x: 0.6
        font_size: sp(24)
        on_release: app.save_config(self, phone_num, sms_freq_input)
    Widget:
        size_hint_y: 1
''')

class ConfigInput(MDBoxLayout):
    """ Takes configuration inputs """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
