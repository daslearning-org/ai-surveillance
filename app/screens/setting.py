from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.list import MDList, OneLineIconListItem, IconLeftWidget, IconRightWidget, OneLineAvatarIconListItem

from kivy.uix.accordion import Accordion, AccordionItem
from kivy.lang import Builder
from kivy.properties import StringProperty, NumericProperty, ObjectProperty
from kivy.metrics import dp, sp

# local imports

Builder.load_string('''

<SettingsBox@MDBoxLayout>:

    Accordion:
        orientation: 'vertical'

        AccordionItem:
            title: "Settings"
            spacing: dp(8)
            canvas.before:
                Color:
                    rgba: (168, 183, 191, 1)
                RoundedRectangle:
                    size: self.width, self.height
                    pos: self.pos

            MDScrollView:
                MDList:
                    OneLineIconListItem:
                        text: "Delete all output images"
                        on_release: app.show_delete_alert()
                        IconLeftWidget:
                            icon: "broom"
                    OneLineIconListItem:
                        text: "Change target SMS number"
                        on_release: app.change_sms_number()
                        IconLeftWidget:
                            icon: "phone"

        AccordionItem:
            title: "Help & Support"
            spacing: dp(8)
            canvas.before:
                Color:
                    rgba: (170, 191, 184, 1)
                RoundedRectangle:
                    size: self.width, self.height
                    pos: self.pos

            MDScrollView:
                MDList:
                    OneLineIconListItem:
                        text: "Demo (How to use)"
                        on_release: app.open_link(self, "https://youtube.com/watch?v=Sbc2sClECdk")
                        IconLeftWidget:
                            icon: "youtube"
                    OneLineIconListItem:
                        text: "Documentation (Blog)"
                        on_release: app.open_link(self, "https://blog.daslearning.in/llm_ai/ml/ai-cctv.html")
                        IconLeftWidget:
                            icon: "file-document-check"
                    OneLineIconListItem:
                        text: "Contact Developer"
                        on_release: app.open_link(self, "https://daslearning.in/contact/")
                        IconLeftWidget:
                            icon: "card-account-phone"
                    OneLineIconListItem:
                        text: "Check for update"
                        on_release: app.update_checker(self)
                        IconLeftWidget:
                            icon: "github"
                    OneLineIconListItem:
                        text: "Try our other Apps"
                        on_release: app.open_link(self, "https://daslearning.in/apps/")
                        IconLeftWidget:
                            icon: "google-play"

''')

class SettingsBox(MDBoxLayout):
    """ The main settings box which contains the setting, help & other required sections """
