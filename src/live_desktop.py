"""Dedicated portrait camera window, with the resulting map in the same window."""

import signal
from threading import Thread
import time

import numpy as np
from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QImage, QKeySequence, QPainter, QShortcut, QTransform
from PySide6.QtMultimedia import QCamera, QMediaCaptureSession, QMediaDevices, QVideoSink
from PySide6.QtWidgets import QApplication, QInputDialog, QLabel, QPushButton, QWidget
from PySide6.QtWebEngineWidgets import QWebEngineView

from scripts.open_map import create_server
from .live_session import LiveSession
from .live_video import portrait_frame


def camera_image(video_frame):
    image = video_frame.toImage()
    if image.isNull():
        return None
    angle = video_frame.rotation().value
    if angle:
        image = image.transformed(QTransform().rotate(angle))
    if video_frame.mirrored():
        image = image.flipped(Qt.Orientation.Horizontal)
    image = image.convertToFormat(QImage.Format.Format_BGR888)
    pixels = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    return pixels[:, :image.width() * 3].reshape(image.height(), image.width(), 3).copy()


class CameraCanvas(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.image = QImage()

    def display(self, frame):
        height, width = frame.shape[:2]
        self.image = QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888).copy()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if not self.image.isNull():
            painter.drawImage(self.rect(), self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() < self.height() * .10:
            self.window().windowHandle().startSystemMove()


class LiveWindow(QWidget):
    def __init__(self, location, output_root, device='auto', road_config=None, *, open_camera=True):
        super().__init__()
        self.location, self.output_root, self.device, self.road_config = location, output_root, device, road_config
        self.session = self.pending = self.latest = None
        self.map_server = self.map_thread = self.map_view = None
        self.closing = False
        self.presented = False
        self.last_camera_epoch = None
        self.camera_reason = 'Conectá una cámara · C para elegir'
        self.camera_name = 'Cámara'
        self.camera = None
        self.start_error = None
        self.setWindowTitle('Urban Vision')
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        available = QApplication.primaryScreen().availableGeometry()
        height = max(160, min(960, available.height() - 32) // 16 * 16)
        width = height * 9 // 16
        self.setFixedSize(width, height)
        self.move(available.center().x() - width // 2, available.center().y() - height // 2)
        self.canvas = CameraCanvas(self)
        self.status = QLabel(self)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setWordWrap(True)
        self.status.setStyleSheet('background: rgba(6, 14, 21, 215); color: #e8f6fa; border-radius: 10px; padding: 10px; font: 13px "Segoe UI";')
        self.start_button = QPushButton('Iniciar recorrido', self)
        self.start_button.setStyleSheet('QPushButton {background: #47ebdd; color: #071b22; border: 0; border-radius: 12px; font: bold 16px "Segoe UI";} QPushButton:disabled {background: #273a42; color: #95a9af;}')
        self.start_button.clicked.connect(self.start_trip)
        self.new_button = QPushButton('Nuevo recorrido', self)
        self.new_button.setStyleSheet(self.start_button.styleSheet())
        self.new_button.clicked.connect(self.new_trip)
        self.new_button.hide()
        self.shortcuts = []
        for key, callback in [('Esc', self.finish_trip), ('F', lambda: self.finish_trip() if self.session else None), ('C', self.choose_camera)]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)
        self.sink = QVideoSink(self)
        self.sink.videoFrameChanged.connect(self.on_video_frame)
        self.capture = QMediaCaptureSession(self)
        self.capture.setVideoSink(self.sink)
        self.media_devices = QMediaDevices(self)
        self.media_devices.videoInputsChanged.connect(self.devices_changed)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)
        self.layout_overlays()
        if open_camera:
            self.devices_changed()

    def layout_overlays(self):
        self.canvas.setGeometry(self.rect())
        if self.map_view:
            self.map_view.setGeometry(self.rect())
        self.start_button.setGeometry(24, self.height() - 82, self.width() - 48, 54)
        self.new_button.setGeometry(24, self.height() - 74, self.width() - 48, 48)
        self.status.setGeometry(24, self.height() // 2 - 50, self.width() - 48, 100)

    def devices_changed(self):
        if self.presented or self.closing:
            return
        devices = QMediaDevices.videoInputs()
        if self.camera and any(device.id() == self.camera.cameraDevice().id() for device in devices):
            return
        if self.session and not self.session.done.is_set():
            self.finish_trip('camera_disconnected')
        elif devices:
            default = QMediaDevices.defaultVideoInput()
            self.use_camera(default if not default.isNull() else devices[0])
        else:
            self.latest = None
            self.last_camera_epoch = None
            self.camera_reason = 'No hay cámaras disponibles · C para elegir'

    def use_camera(self, device):
        if self.camera:
            self.camera.stop()
            self.camera.deleteLater()
        self.latest = None
        self.last_camera_epoch = None
        self.camera_name = device.description()
        self.start_error = None
        self.camera_reason = 'Abriendo cámara…'
        self.camera = QCamera(device, self)
        # Prefer HD at 30 fps; portrait cameras retain their native orientation.
        formats = [f for f in device.videoFormats() if f.maxFrameRate() >= 30]
        if formats:
            selected = min(formats, key=lambda f: abs(max(f.resolution().width(), f.resolution().height()) - 1920))
            self.camera.setCameraFormat(selected)
        self.camera.errorOccurred.connect(self.camera_error)
        self.capture.setCamera(self.camera)
        self.camera.start()

    def camera_error(self, error, message):
        self.camera_reason = message or 'No se pudo abrir la cámara · C para elegir'
        self.latest = None
        self.last_camera_epoch = None
        if self.session and not self.session.done.is_set():
            self.finish_trip('camera_disconnected')

    def choose_camera(self):
        if self.session is not None:
            return
        devices = QMediaDevices.videoInputs()
        if not devices:
            self.camera_reason = 'No hay cámaras disponibles'
            return
        labels = [f'{index + 1}. {device.description()}' for index, device in enumerate(devices)]
        current = next((i for i, device in enumerate(devices) if self.camera and device.id() == self.camera.cameraDevice().id()), 0)
        selected, accepted = QInputDialog.getItem(self, 'Elegir cámara', 'Cámara', labels, current, False)
        if accepted:
            self.use_camera(devices[labels.index(selected)])

    def on_video_frame(self, video_frame):
        if self.map_view or self.closing:
            return
        image = camera_image(video_frame)
        if image is not None:
            self.accept_frame(image)

    def accept_frame(self, image, epoch=None):
        self.last_camera_epoch = epoch or time.time()
        self.latest = (portrait_frame(image), self.last_camera_epoch)
        if not self.session or self.session.state in {'paused', 'preparing'}:
            self.canvas.display(self.latest[0])

    def start_trip(self):
        if self.session or self.latest is None:
            return
        try:
            self.start_error = None
            session = LiveSession(self.location, self.output_root, self.device, self.road_config, desktop=True)
            session.start(self.camera_name)
            self.session = session
            self.start_button.hide()
        except (ValueError, OSError) as exc:
            self.start_error = str(exc)
            self.status.setText(str(exc))

    def finish_trip(self, reason='user_finished'):
        if self.session and not self.session.done.is_set():
            self.session.finish(reason)
        elif not self.session and reason == 'user_finished':
            self.close()

    def tick(self):
        if self.pending and self.pending.done():
            try:
                image = self.pending.result()
                if image is not None and self.session.state != 'paused':
                    self.canvas.display(image)
            except Exception as exc:
                self.status.setText(str(exc))
            self.pending = None
        if self.session:
            state = self.session.snapshot()
            if not state['location']['valid'] and self.session.video:
                # The GUI's GPS watchdog also stops the encoder during slow inference.
                self.session.video.pause()
            if self.session.done.is_set():
                if self.closing:
                    self.close()
                    return
                if not self.presented:
                    self.presented = True
                    self.show_map()
                return
            if (self.last_camera_epoch is not None and time.time() - self.last_camera_epoch > 5):
                self.finish_trip('camera_disconnected')
            if not self.pending and self.latest and not self.session.stop.is_set():
                image, epoch = self.latest
                self.latest = None
                try:
                    self.pending = self.session.submit(image, epoch)
                except ValueError:
                    pass
            visible = state['state'] != 'running'
            self.status.setVisible(visible)
            if visible:
                prefix = 'Registro pausado\n' if state['state'] == 'paused' else ''
                self.status.setText(prefix + state['reason'])
        else:
            gps, _ = self.location.snapshot()
            camera_ready = self.latest is not None and time.time() - self.last_camera_epoch <= 2
            self.start_button.setEnabled(camera_ready and gps['valid'])
            self.status.show()
            accuracy = gps['accuracy_m']
            detail = f'Ubicación de Windows · margen ±{accuracy:.0f} m\n' if accuracy is not None else 'Ubicación de Windows\n'
            self.status.setText(self.start_error or ((detail + gps['reason'] if not gps['valid'] else 'C para elegir cámara · Esc para salir')
                                if camera_ready else self.camera_reason))

    def show_map(self):
        if self.camera:
            self.camera.stop()
        self.new_button.show()
        path = self.session.directory / 'map.html'
        if not path.is_file():
            self.status.setText('No se pudo generar el mapa.\n' + self.session.reason)
            self.status.show()
            return
        try:
            self.map_server, url = create_server(path, self.session.directory, 0)
            self.map_thread = Thread(target=self.map_server.serve_forever, daemon=True)
            self.map_thread.start()
            self.map_view = QWebEngineView(self)
            self.map_view.setGeometry(self.rect())
            self.map_view.loadFinished.connect(self.map_loaded)
            self.map_view.load(QUrl(url))
            self.map_view.show()
            self.new_button.raise_()
            self.status.setText('Abriendo mapa…')
            self.status.show()
            self.status.raise_()
        except Exception as exc:
            self.status.setText(f'No se pudo abrir el mapa: {exc}')
            self.status.show()

    def map_loaded(self, ok):
        if ok:
            # Original evidence links usually open a browser tab. Keep photos here.
            self.map_view.page().runJavaScript('''
                if (!window.nativeEvidenceViewer) {
                  window.nativeEvidenceViewer = true;
                  document.addEventListener('click', event => {
                    const link = event.target.closest('a');
                    if (!link) return;
                    const url = new URL(link.href, location.href);
                    if (url.origin !== location.origin || !/\\.jpg$/i.test(url.pathname)) return;
                    event.preventDefault();
                    const overlay = document.createElement('div');
                    overlay.id = 'native-evidence';
                    overlay.style.cssText = 'position:fixed;inset:0;z-index:10000;background:#09141b;display:flex;flex-direction:column;align-items:center;padding:16px;box-sizing:border-box';
                    const back = document.createElement('button');
                    back.textContent = 'Volver al mapa';
                    back.style.cssText = 'background:#47ebdd;border:0;border-radius:8px;padding:12px 20px;font:bold 14px Arial;cursor:pointer;margin-bottom:12px';
                    back.onclick = () => overlay.remove();
                    const photo = document.createElement('img');
                    photo.src = url.href;
                    photo.alt = 'Evidencia original del pozo';
                    photo.style.cssText = 'width:100%;flex:1;min-height:0;object-fit:contain;padding-bottom:80px';
                    overlay.append(back, photo);
                    document.body.append(overlay);
                  });
                }
            ''')
        errors = self.session.export_errors
        if not ok or errors or self.session.error:
            self.status.setText('No se pudo cargar el mapa' if not ok else '\n'.join(errors + ([self.session.error] if self.session.error else [])))
            self.status.show()
        else:
            self.status.hide()

    def stop_map(self):
        if self.map_view:
            self.map_view.close()
            self.map_view.deleteLater()
            self.map_view = None
        if self.map_server:
            self.map_server.shutdown()
            self.map_server.server_close()
            self.map_thread.join()
            self.map_server = self.map_thread = None

    def new_trip(self):
        self.stop_map()
        self.session = self.pending = self.latest = None
        self.presented = False
        self.start_error = None
        self.last_camera_epoch = None
        self.new_button.hide()
        self.start_button.show()
        if self.camera:
            self.camera.start()
        else:
            self.devices_changed()

    def closeEvent(self, event):
        if self.session and not self.session.done.is_set():
            self.closing = True
            self.session.finish('window_closed')
            event.ignore()
            return
        self.timer.stop()
        if self.camera:
            self.camera.stop()
        self.stop_map()
        self.location.close()
        event.accept()


def run_desktop(location, output_root, device, road_config):
    app = QApplication.instance() or QApplication([])
    window = LiveWindow(location, output_root, device, road_config)
    previous = signal.signal(signal.SIGINT, lambda *_: window.close())
    window.show()
    try:
        return app.exec()
    finally:
        signal.signal(signal.SIGINT, previous)
        if window.session and window.session.thread.is_alive():
            window.session.finish('window_closed')
            window.session.thread.join()
        window.stop_map()
        if window.camera:
            window.camera.stop()
