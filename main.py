from kivy.app import App
from kivy.uix.widget import Widget
from kivy.core.audio import SoundLoader
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
import cv2
import os
import time
import numpy as np
from kivy.core.window import Window

os.environ['KIVY_CAMERA'] = 'opencv'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

class Blind(Widget):
    pass


class BlindApp(App):
    def build(self):
        return Blind()
    
    def capture_photo(self):
        camera = self.root.ids.camera
        
        # Check if the camera is actually playing and has a texture
        if not camera.play or not camera.texture:
            print("Error: Camera is not ready or no frame available.")
            return
        timestr = time.strftime("%Y%m%d_%H%M%S")
        try:
            # Extract texture data
            texture = camera.texture
            texture_data = texture.pixels
            
            # Convert texture to OpenCV format
            img_width = texture.width
            img_height = texture.height
            img_array = np.frombuffer(texture_data, np.uint8).reshape((img_height, img_width, 4))
            
            # Convert RGBA to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            
            # Save as JPEG with high quality (95%)
            filename = f"IMG_{timestr}.jpg"
            cv2.imwrite(filename, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"Saved: {filename} (1280x720, JPEG Quality: 95)")
        except Exception as e:
            print(f"Capture failed: {e}")


if __name__ == '__main__':
    BlindApp().run()