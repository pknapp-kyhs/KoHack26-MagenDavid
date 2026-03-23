from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.lang import Builder
import cv2
import os
import time
import numpy as np
from kivy.core.window import Window
import pytesseract
from gtts import gTTS
from threading import Thread
try:
    from googletrans import Translator
    translator = Translator()
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False
    print("Note: googletrans not available, translations will be skipped")

os.environ['KIVY_CAMERA'] = 'opencv'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

class Blind(Widget):
    hebrew_text = ""
    english_text = ""
    audio_file = ""


class BlindApp(App):
    def build(self):
        Window.size = (900, 900)
        root = Blind()
        root.app = self
        return root
    
    def capture_photo(self):
        # Capture directly from OpenCV camera (more reliable than Kivy Camera provider)
        capture_button = self.root.ids.capture_button
        capture_button.disabled = True
        original_text = capture_button.text
        capture_button.text = "Processing..."

        self.root.ids.hebrew_label.text = "Capturing image..."
        self.root.ids.english_label.text = "Processing..."

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.root.ids.hebrew_label.text = "Error: Cannot open camera"
            capture_button.disabled = False
            capture_button.text = original_text
            return

        ret, frame = cap.read()
        cap.release()

        if not ret:
            self.root.ids.hebrew_label.text = "Error: Failed to read frame from camera"
            capture_button.disabled = False
            capture_button.text = original_text
            return

        timestr = time.strftime("%Y%m%d_%H%M%S")
        filename = f"IMG_{timestr}.jpg"
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved: {filename}")

        # Process the image for OCR in a separate thread
        thread = Thread(target=self._process_image_thread, args=(filename, capture_button, original_text))
        thread.daemon = True
        thread.start()
    
    def _process_image_thread(self, image_path, capture_button, original_text):
        """Process image in a separate thread"""
        try:
            self.process_image(image_path)
        finally:
            # Re-enable button
            capture_button.disabled = False
            capture_button.text = original_text
    
    def process_image(self, image_path):
        """Extract Hebrew text, translate it, and generate speech"""
        try:
            print(f"Processing image: {image_path}")
            
            # Read and preprocess the image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f"Failed to read image: {image_path}")
                
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            denoised_image = cv2.medianBlur(gray_image, 3)
            
            _, threshold_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((5, 5), np.uint8)
            dilated_image = cv2.dilate(threshold_image, kernel, iterations=1)
            eroded_image = cv2.erode(dilated_image, kernel, iterations=1)
            
            edges_image = cv2.Canny(eroded_image, 50, 150, apertureSize=3)
            
            coords = np.column_stack(np.where(edges_image > 0))
            if len(coords) > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
            else:
                angle = 0
            
            h, w = edges_image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(eroded_image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            resized_image = cv2.resize(rotated_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            
            # OCR extraction
            try:
                custom_config = r'--oem 3 --psm 6 -l heb'
                hebrew_text = pytesseract.image_to_string(resized_image, config=custom_config)
                print(f"OCR Result: {hebrew_text}")
            except Exception as ocr_error:
                print(f"OCR Error: {ocr_error}")
                hebrew_text = ""
                self.root.ids.hebrew_label.text = "Hebrew: Tesseract OCR not installed. Install from: https://github.com/UB-Mannheim/tesseract/wiki"
                return
            
            if hebrew_text.strip():
                # Store Hebrew text
                self.hebrew_text = hebrew_text
                self.root.ids.hebrew_label.text = f"Hebrew: {hebrew_text}"
                print(f"Hebrew text extracted: {hebrew_text}")
                
                # Translate to English if translator available
                if HAS_TRANSLATOR:
                    try:
                        translation = translator.translate(hebrew_text, src='he', dest='en').text
                        self.english_text = translation
                        self.root.ids.english_label.text = f"English: {translation}"
                        print(f"English translation: {translation}")
                    except Exception as e:
                        print(f"Translation failed: {e}")
                        self.english_text = "(Translation unavailable)"
                        self.root.ids.english_label.text = "English: Translation unavailable"
                else:
                    self.english_text = "(Translation library not available)"
                    self.root.ids.english_label.text = "English: Translation library not available"
                
                # Generate speech
                self.generate_speech(hebrew_text)
            else:
                print("No Hebrew text detected in image")
                self.root.ids.hebrew_label.text = "Hebrew: No text detected"
        except Exception as e:
            print(f"Processing failed: {e}")
            self.root.ids.hebrew_label.text = f"Error: {str(e)}"
    
    def generate_speech(self, text):
        """Generate Hebrew speech from text"""
        try:
            timestr = time.strftime("%Y%m%d_%H%M%S")
            audio_file = f"output_{timestr}"
            myobj = gTTS(text=text, lang='iw', slow=False)
            myobj.save(f"{audio_file}.mp3")
            self.audio_file = audio_file
            print(f"Audio generated: {audio_file}.mp3")
            self.root.ids.play_button.disabled = False
        except Exception as e:
            print(f"Speech generation failed: {e}")
    
    def play_audio(self):
        """Play the generated audio file"""
        if self.audio_file:
            try:
                os.system(f"start {self.audio_file}.mp3")
                print(f"Playing: {self.audio_file}.mp3")
            except Exception as e:
                print(f"Failed to play audio: {e}")


if __name__ == '__main__':
    BlindApp().run()