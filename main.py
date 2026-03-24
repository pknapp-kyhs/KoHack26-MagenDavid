from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image
from kivy.uix.slider import Slider
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
import cv2
import os
import time
import numpy as np
import pytesseract
from gtts import gTTS
from threading import Thread, Lock
from googletrans import Translator
translator = Translator()

try:
    from bidi.algorithm import get_display
    BIDIRECTIONAL_AVAILABLE = True
except ImportError:
    BIDIRECTIONAL_AVAILABLE = False

# Fallback bidi routine for simple Hebrew direction if python-bidi isn't available
def bidi_display(text):
    if not text.strip():
        return text
    if BIDIRECTIONAL_AVAILABLE:
        return get_display(text)

    # fallback: reverse word order for crude right-to-left presentation
    lines = text.split('\n')
    relined = []
    for line in lines:
        words = line.split(' ')
        relined.append(' '.join(reversed(words)))
    return '\n'.join(relined)

os.environ['KIVY_CAMERA'] = 'opencv'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

class Blind(Widget):
    hebrew_text = ""
    english_text = ""
    audio_file = ""
    hebrew_audio_file = ""
    english_audio_file = ""
    hebrew_sound = None
    english_sound = None
    current_sound = None
    current_language = ""
    last_image = ""
    camera_lock = Lock()

timestr = time.strftime("%Y%m%d_%H%M%S")

def format_time(seconds):
    """Convert seconds to MM:SS format"""
    if seconds < 0:
        seconds = 0
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins}:{secs:02d}"

class BlindApp(App):
    camera_cap = None
    slider_event = None
    slider_is_scrubbing = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize audio attributes
        self.hebrew_sound = None
        self.english_sound = None
        self.current_sound = None
        self.current_language = ""
        self.slider_event = None
        self.playback_start_time = None
        self.playback_seek_pos = 0.0
    
    def order_points(self, pts):
        """Order points for perspective transform: tl, tr, br, bl"""
        rect = np.zeros((4, 2), dtype='float32')
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect
    
    def build(self):
        # Set fullscreen 1080p
        Window.size = (1920, 1080)
        Window.fullscreen = True
        root = Blind()
        root.app = self
        root.size_hint = (1, 1)
        root.pos_hint = {'x': 0, 'y': 0}
        
        # Start continuous camera feed
        self.camera_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if self.camera_cap.isOpened():
            self.camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            Clock.schedule_interval(self.update_camera_feed, 1.0 / 30.0)  # 30 FPS
        
        return root
    
    def update_camera_feed(self, dt):
        """Continuously update camera preview"""
        if self.camera_cap and self.camera_cap.isOpened():
            ret, frame = self.camera_cap.read()
            if ret:
                # Flip frame vertically (across x-axis)
                frame = cv2.flip(frame, 0)
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to texture and display
                texture = self.frame_to_texture(frame)
                self.root.ids.camera_image.texture = texture
    
    def frame_to_texture(self, frame):
        """Convert OpenCV frame to Kivy Texture"""
        buf = frame.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        return texture
    
    def capture_photo(self):
        # Reset audio UI and stop existing audio before new capture
        if self.current_sound:
            self.current_sound.stop()
            self.current_sound = None
        self.root.ids.play_hebrew_btn.disabled = True
        self.root.ids.play_english_btn.disabled = True
        self.root.ids.stop_btn.disabled = True
        self.root.ids.audio_slider.value = 0
        self.root.ids.audio_slider.disabled = True

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

        filename = f"IMG_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved: {filename}")
        self.root.ids.filename_label.text = f"File: {filename}"
        self.root.last_image = filename

        # Process the image for OCR in a separate thread
        thread = Thread(target=self._process_image_thread, args=(filename, capture_button, original_text))
        thread.daemon = True
        thread.start()
    
    def _process_image_thread(self, image_path, capture_button, original_text):
        """Process image in a separate thread"""
        self.process_image(image_path)
        # Re-enable button
        Clock.schedule_once(lambda dt: self._restore_button(capture_button, original_text), 0)
    
    def _restore_button(self, capture_button, original_text):
        """Restore button state on main thread"""
        capture_button.disabled = False
        capture_button.text = original_text
        self.root.ids.audio_slider.disabled = True
        self.root.ids.audio_slider.value = 0
    
    def process_image(self, image_path):
        """Extract Hebrew text, translate it, and generate speech"""
        print(f"Processing image: {image_path}")
        start_time = time.time()

        image = cv2.imread(image_path)
        if image is None:
            raise Exception(f"Failed to read image: {image_path}")

        if max(image.shape[:2]) > 1200:
            scale = 1200.0 / max(image.shape[:2])
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

        edges = cv2.Canny(blurred, 50, 150)
        contours_data = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        page_contour = None
        image_area = image.shape[0] * image.shape[1]
        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.02 * image_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                page_contour = approx
                break

        if page_contour is not None:
            rect = self.order_points(page_contour.reshape(4, 2))
            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = int(max(widthA, widthB))
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxHeight = int(max(heightA, heightB))

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype='float32')

            M = cv2.getPerspectiveTransform(rect, dst)
            page_image = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            cropped_gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
            self.root.ids.hebrew_label.text = "Page detected and cropped"
        else:
            cropped_gray = gray_image
            self.root.ids.hebrew_label.text = "No page detected; using full frame"

        if cropped_gray.max() > 220:
            cropped_gray = cv2.equalizeHist(cropped_gray)

        denoised = cv2.fastNlMeansDenoising(cropped_gray, h=10, templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        equalized = clahe.apply(denoised)

        # Brighten the image a bit for easier OCR and display
        bright_img = cv2.convertScaleAbs(equalized, alpha=1.0, beta=15)

        # Sharpen the image (unsharp mask) to reduce blur before OCR
        sharpened_img = cv2.addWeighted(bright_img, 1.5, bright_img, -0.5, 0)

        adaptive = cv2.adaptiveThreshold(sharpened_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
        otsu = cv2.threshold(sharpened_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Save the processed image used for OCR
        edit_image_path = f"IMG_EDIT_{timestr}.jpg"
        cv2.imwrite(edit_image_path, sharpened_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved OCR image: {edit_image_path}")

        variants = [sharpened_img, adaptive, cv2.bitwise_not(adaptive), otsu, cv2.bitwise_not(otsu)]
        configs = [r'--oem 3 --psm 6 -l heb', r'--oem 3 --psm 11 -l heb', r'--oem 3 --psm 6 -l heb+eng']

        def hebrew_score(text):
            heb_chars = [c for c in text if '\u0590' <= c <= '\u05FF']
            return len(heb_chars) * 3 + len(text)

        best_text = ""
        best_score = 0

        for variant in variants:
            candidate = cv2.resize(variant, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            for config in configs:
                text = pytesseract.image_to_string(candidate, config=config).strip()
                score = hebrew_score(text)
                if score > best_score:
                    best_score = score
                    best_text = text

        hebrew_text = best_text.strip() or ""

        if not hebrew_text:
            hebrew_text = pytesseract.image_to_string(cv2.resize(equalized, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), config=r'--oem 3 --psm 6 -l heb').strip()

        elapsed = time.time() - start_time
        print(f"OCR finished in {elapsed:.2f}s, score={best_score}, text length={len(hebrew_text)}")

        with open(f"heb_{timestr}.txt", "w", encoding="utf-8") as f:
            f.write(hebrew_text)

        translation = translator.translate(hebrew_text if hebrew_text else " ", src='he', dest='en').text

        with open(f"eng_{timestr}.txt", "w", encoding="utf-8") as f:
            f.write(translation)

        Clock.schedule_once(lambda dt: self._update_text_ui(hebrew_text, translation), 0)
    
    def _update_text_ui(self, hebrew_text, translation):
        """Update text labels on main thread"""
        hebrew_text = bidi_display(hebrew_text)

        self.hebrew_text = hebrew_text
        self.english_text = translation
        self.root.ids.hebrew_label.text = hebrew_text if hebrew_text.strip() else "No text detected"
        self.root.ids.english_label.text = translation if translation.strip() else "No translation available"
        # Generate audio for both languages
        self.generate_speech(hebrew_text, translation)
    
    def generate_speech(self, hebrew_text, english_text):
        """Generate speech for both Hebrew and English"""
        hebrew_text = hebrew_text.strip()
        english_text = english_text.strip()

        if not hebrew_text and not english_text:
            # nothing to speak
            self.root.ids.hebrew_label.text = "No OCR text to generate audio"
            self.root.ids.play_hebrew_btn.disabled = True
            self.root.ids.play_english_btn.disabled = True
            self.root.ids.stop_btn.disabled = True
            return

        if not hebrew_text:
            hebrew_text = "No text detected"
        if not english_text:
            english_text = "No translation available"

        # Generate Hebrew audio
        hebrew_file = f"output_hebrew_{timestr}.mp3"
        hebrew_tts = gTTS(text=hebrew_text, lang='iw', slow=False)
        hebrew_tts.save(hebrew_file)
        self.hebrew_audio_file = hebrew_file
        self.hebrew_sound = SoundLoader.load(hebrew_file)
        print(f"Hebrew audio generated: {hebrew_file}")

        # Generate English audio
        english_file = f"output_english_{timestr}.mp3"
        english_tts = gTTS(text=english_text, lang='en', slow=False)
        english_tts.save(english_file)
        self.english_audio_file = english_file
        self.english_sound = SoundLoader.load(english_file)
        print(f"English audio generated: {english_file}")
        
        # Enable audio buttons and set default to Hebrew
        Clock.schedule_once(lambda dt: self._enable_audio_controls(), 0)
    
    def _enable_audio_controls(self):
        """Enable audio control buttons on main thread"""
        self.root.ids.play_hebrew_btn.disabled = False
        self.root.ids.play_english_btn.disabled = False
        self.root.ids.stop_btn.disabled = False
    
    def play_hebrew_audio(self):
        """Play Hebrew audio"""
        if self.hebrew_sound:
            if self.current_sound:
                self.current_sound.stop()
            self.current_sound = self.hebrew_sound
            self.current_language = "hebrew"
            if self.hebrew_sound.length > 0:
                self.root.ids.audio_slider.max = self.hebrew_sound.length
                self.root.ids.time_total_label.text = format_time(self.hebrew_sound.length)
            self.playback_seek_pos = 0.0
            self.playback_start_time = time.time()
            self.hebrew_sound.play()
            self.start_slider_update()
    
    def play_english_audio(self):
        """Play English audio"""
        if self.english_sound:
            if self.current_sound:
                self.current_sound.stop()
            self.current_sound = self.english_sound
            self.current_language = "english"
            if self.english_sound.length > 0:
                self.root.ids.audio_slider.max = self.english_sound.length
                self.root.ids.time_total_label.text = format_time(self.english_sound.length)
            self.playback_seek_pos = 0.0
            self.playback_start_time = time.time()
            self.english_sound.play()
            self.start_slider_update()
    
    def stop_audio(self):
        """Stop audio playback"""
        if self.current_sound:
            self.current_sound.stop()
            self.current_sound = None
            self.root.ids.audio_slider.value = 0
            self.root.ids.time_current_label.text = "0:00"

        if self.slider_event is not None:
            Clock.unschedule(self.slider_event)
            self.slider_event = None

        self.root.ids.audio_slider.disabled = True
        self.slider_is_scrubbing = False
    
    def on_slider_touch_down(self, slider):
        """Handle slider touch down for scrubbing"""
        self.slider_is_scrubbing = True
        if self.slider_event is not None:
            Clock.unschedule(self.slider_event)
            self.slider_event = None
    
    def on_slider_touch_move(self, slider):
        """Handle slider touch move for live scrubbing"""
        if self.slider_is_scrubbing and self.current_sound:
            self.current_sound.seek(slider.value)
            self.root.ids.time_current_label.text = format_time(slider.value)
    
    def on_slider_touch_up(self, slider):
        """Handle slider touch up to finish scrubbing"""
        if self.current_sound:
            self.current_sound.seek(slider.value)
            self.playback_seek_pos = slider.value
            self.playback_start_time = time.time()
            self.root.ids.time_current_label.text = format_time(slider.value)
        self.slider_is_scrubbing = False
        if self.current_sound and self.current_sound.state == 'play':
            self.start_slider_update()
    
    def on_slider_change(self, value):
        """Handle slider scrubbing (legacy)"""
        if self.current_sound:
            self.current_sound.seek(value)
            self.playback_seek_pos = value
            self.playback_start_time = time.time()

    def start_slider_update(self):
        """Update slider as audio plays"""
        self.root.ids.audio_slider.disabled = False
        if self.slider_event is not None:
            Clock.unschedule(self.slider_event)
        self.slider_event = Clock.schedule_interval(self.update_slider, 0.1)
    
    def get_current_audio_position(self):
        """Get current playback position trustingly, with fallback."""
        if not self.current_sound:
            return 0

        if hasattr(self.current_sound, 'get_pos'):
            pos = self.current_sound.get_pos()
            if pos is not None and pos >= 0:
                return min(pos, self.root.ids.audio_slider.max)

        if self.playback_start_time is not None:
            pos = self.playback_seek_pos + (time.time() - self.playback_start_time)
            max_val = self.root.ids.audio_slider.max
            if max_val > 0:
                return min(pos, max_val)
            return pos

        return self.playback_seek_pos

    def update_slider(self, dt):
        """Update slider position during playback"""
        if self.current_sound and self.current_sound.state == 'play' and not self.slider_is_scrubbing:
            pos = self.get_current_audio_position()
            max_val = self.root.ids.audio_slider.max
            if max_val > 0:
                self.root.ids.audio_slider.value = min(pos, max_val)
            self.root.ids.time_current_label.text = format_time(pos)
            if max_val > 0 and pos >= max_val:
                self.stop_audio()
                return False
            return True

        # Audio stopped or no current sound
        if not self.slider_is_scrubbing and self.slider_event is not None:
            self.root.ids.audio_slider.disabled = True
            self.slider_event = None
        return False  # Stop scheduling

if __name__ == '__main__':
    BlindApp().run()