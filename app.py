# -*- coding: utf-8 -*-
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, \
    QMessageBox, QWidget, QHeaderView, QTableWidgetItem, QAbstractItemView
import sys
import os
from PIL import ImageFont
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
sys.path.append('UIProgram')
import sys
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QCoreApplication
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import QSystemTrayIcon, QMenu
import detect_tools as tools
import cv2
from UIProgram.PrecessBar import ProgressBar
from UIProgram.CameraProgress import CamProgress
import numpy as np
import pandas as pd
import torch
import warnings
from PyQt5 import uic
from DiffBIR import diff_bir
import ctypes  # 需要用到的库

ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('myappid')
warnings.filterwarnings('ignore', category=DeprecationWarning)


class MainWindow(QMainWindow):
    update_camera_ui_signal = pyqtSignal(int)
    update_progress_signal = pyqtSignal(int, int)  # 新增进度更新信号
    detection_finished_signal = pyqtSignal(dict)  # 新增检测完成信号

    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.ui = uic.loadUi('UIProgram/StomaD2.ui')
        self.ui.show()

        self.ui.setFixedSize(1000, 600)
        # self.ui.setupUi(self)
        # self.ui.noir.setChecked(True)
        self.using_ir = False
        self.ui.shuangziye.setChecked(True)
        self.ui.nondestructive.setChecked(True)

        self._initMain()
        self.signalconnect()
        self.ui.setWindowTitle('StomaD\u00B2')

        self.conf = 0.5
        self.iou = 0.7
        self.step = 5

        # 添加线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=4)  # 根据CPU核心数调整
        self.running_tasks = set()
        self.task_lock = threading.Lock()

    def signalconnect(self):
        self.ui.select_images.clicked.connect(self.open_img)
        self.ui.select_videos.clicked.connect(self.video_show)
        self.ui.open_camera.clicked.connect(self.camera_show)
        self.ui.save_results.clicked.connect(self.save_detect_result)
        self.ui.exit.clicked.connect(QCoreApplication.quit)
        self.ui.batch_detection.clicked.connect(self.folder)
        self.ui.conf.valueChanged.connect(self.conf_value_change)
        self.ui.iou.valueChanged.connect(self.iou_value_change)
        self.ui.step.valueChanged.connect(self.step_value_change)
        self.update_camera_ui_signal.connect(self.update_camera_process)
        self.ui.det.clicked.connect(self.load_det_weight)
        self.ui.restoration_det.clicked.connect(self.load_restoration_det_weight)
        self.ui.start.clicked.connect(self.start)
        self.detection_finished_signal.connect(self.update_ui_with_result)
        self.update_progress_signal.connect(self.update_progress_bar)

    def _initMain(self):
        self.show_width = 330
        self.show_height = 330

        self.org_path = None

        self.is_camera_open = False
        self.cap = None

        self.save_camera_flag = False  # 摄像头保存标志

        self.device = 0 if torch.cuda.is_available() else 'cpu'

        # 加载检测模型
        self.model = YOLO('weights/dicotyledons_nondestructive.pt', task='obb')
        # self.model = YOLO(r'E:\ZQL\StomaD2\runs\obb\train\weights\best.pt', task='obb')
        self.model(np.zeros((48, 48, 3)), device=self.device)  # 预先加载推理模型
        # self.fontC = ImageFont.truetype("Font/platech.ttf", 25, 0)

        self.ir_model = None

        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()

        # 更新视频图像
        self.timer_camera = QTimer()

        # 更新检测信息表格
        # self.timer_info = QTimer()
        # 保存视频
        self.timer_save_video = QTimer()

        self.ui.iou.setRange(0, 1)
        self.ui.iou.setWrapping(True)
        self.ui.iou.setSingleStep(0.1)

        self.ui.conf.setRange(0, 1)
        self.ui.conf.setWrapping(True)
        self.ui.conf.setSingleStep(0.1)

        self.ui.step.setRange(0, 50)
        self.ui.step.setWrapping(True)

        self.ui.density.setAlignment(Qt.AlignCenter)
        self.ui.conductance.setAlignment(Qt.AlignCenter)
        self.ui.stoma_aspect.setAlignment(Qt.AlignCenter)
        self.ui.stoma_length.setAlignment(Qt.AlignCenter)
        self.ui.stoma_width.setAlignment(Qt.AlignCenter)
        self.ui.aperture_aspect.setAlignment(Qt.AlignCenter)
        self.ui.aperture_length.setAlignment(Qt.AlignCenter)
        self.ui.aperture_width.setAlignment(Qt.AlignCenter)

    def load_det_weight(self):

        if self.ui.shuangziye.isChecked() and self.ui.nondestructive.isChecked():
            weight_path = 'weights/dicotyledons_nondestructive.pt'
            # weight_path = r'E:\ZQL\StomaD2\runs\obb\train\weights\best.pt'
        elif self.ui.danziye.isChecked() and self.ui.nondestructive.isChecked():
            weight_path = 'weights/monocotyledons_nondestructive.pt'
        elif self.ui.shuangziye.isChecked():
            weight_path = 'weights/dicotyledons_destructive.pt'
            # weight_path = r'E:\ZQL\StomaD2\runs\obb\train\weights\best.pt'
        elif self.ui.danziye.isChecked():
            weight_path = 'weights/monocotyledons_destructive.pt'
        else:
            QMessageBox.warning(self, "Warning", "Please select a plant type (Dicotyledons or Monocotyledons).")
            return

        self.model = YOLO(weight_path, task='obb')
        self.model(np.zeros((48, 48, 3)), device=self.device)  # 预加载推理模型
        self.using_ir = False
        QMessageBox.information(self, "Tip", f"Successfully load detection model:\n{weight_path}")

    def load_restoration_det_weight(self):

        if self.ui.shuangziye.isChecked() and self.ui.nondestructive.isChecked():
            weight_path = 'weights/dicotyledons_nondestructive.pt'
            # weight_path = r'E:\ZQL\StomaD2\runs\obb\train\weights\best.pt'
        elif self.ui.danziye.isChecked() and self.ui.nondestructive.isChecked():
            weight_path = 'weights/monocotyledons_nondestructive.pt'
        elif self.ui.shuangziye.isChecked():
            weight_path = 'weights/dicotyledons_destructive.pt'
            # weight_path = r'E:\ZQL\StomaD2\runs\obb\train\weights\best.pt'
        elif self.ui.danziye.isChecked():
            weight_path = 'weights/monocotyledons_destructive.pt'
        else:
            QMessageBox.warning(self, "Warning", "Please select a plant type (Dicotyledons or Monocotyledons).")
            return

        # 加载模型
        self.model = YOLO(weight_path, task='obb')
        self.model(np.zeros((48, 48, 3)), device=self.device)  # 预加载推理模型

        if self.ir_model is None:
            self.ir_model = diff_bir(device=self.check_device(self.device), steps=self.step)
        self.using_ir = True
        QMessageBox.information(self, "Tip", f"Successfully load restoration detection model:\n{weight_path}")

    def conf_value_change(self):
        # 改变置信度值
        cur_conf = round(self.ui.conf.value(), 2)
        self.conf = cur_conf

    def iou_value_change(self):
        # 改变iou值
        cur_iou = round(self.ui.iou.value(), 2)
        self.iou = cur_iou

    def step_value_change(self):
        cur_step = round(self.ui.step.value(), 2)
        self.step = cur_step

    def open_img(self):
        if self.cap:
            # 打开图片前关闭摄像头
            self.video_stop()
            self.is_camera_open = False
            self.cap = None
            self.ui.open_camera.setText('Camera on')
            self.ui.select_videos.setText('Video Start')

        # 弹出的窗口名称：'打开图片'
        # 默认打开的目录：'./'
        # 只能打开.jpg与.gif结尾的图片文件
        file_path, _ = QFileDialog.getOpenFileName(None, 'Open Image', './', "Image files (*.jpg *.jpeg *.png *.bmp)")
        if not file_path:
            return
        self.org_path = [file_path]

    def folder(self):
        if self.cap:
            # 打开图片前关闭摄像头
            self.video_stop()
            self.is_camera_open = False
            self.cap = None
            self.ui.open_camera.setText('Camera On')
            self.ui.select_videos.setText('Video Start')

        directory = QFileDialog.getExistingDirectory(self,
                                                     "Select Directory",
                                                     "./")  # 起始路径
        if not directory:
            return
        self.org_path = []

        for file_name in os.listdir(directory):
            full_path = os.path.join(directory, file_name)
            self.org_path.append(full_path)

    def start(self):

        img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
        if not self.org_path:
            QMessageBox.warning(self, "Warning", "Please select images first.")
            return

        # 准备任务
        futures = []
        for img_path in self.org_path:
            if os.path.basename(img_path).split('.')[-1].lower() not in img_suffix:
                continue

            future = self.thread_pool.submit(
                self.process_single_image,
                img_path,
                self.conf,
                self.iou,
                self.using_ir
            )
            futures.append(future)
            with self.task_lock:
                self.running_tasks.add(future)

        # 监控任务完成情况
        self.monitor_thread = threading.Thread(
            target=self.monitor_tasks,
            args=(futures,),
            daemon=True
        )
        self.monitor_thread.start()

        self.save_results = [
            ['file name', 'stomata average height (um)', 'stomata average width (um)', 'stomata aspect ratio',
             'aperture average height (um)', 'aperture average width (um)', 'aperture aspect ratio',
             'stomata density (stomata * mm-2)', 'conductance (mol m-2 s-1)', 'stoma count', 'aperture count',
             'image area']
        ]

    def process_single_image(self, img_path, conf, iou, use_diffbir):
        """处理单张图片的线程函数"""
        try:
            # 读取并预处理图像
            img = tools.img_cvread(img_path)
            result_data = {
                'org_img': img,
            }
            if use_diffbir:
                self.ir_model.args.input = img_path
                self.ir_model.args.steps = self.step
                img = self.ir_model.run()[0]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 执行检测
            results = self.model(img, conf=conf, iou=iou)[0]
            # 处理结果
            result_data.update({
                'img_path': img_path,
                'results': results,
                'success': True,
            })
            return result_data

        except Exception as e:
            return {
                'img_path': img_path,
                'error': str(e),
                'success': False
            }

    def monitor_tasks(self, futures):
        """监控任务完成情况并更新UI"""
        total = len(futures)
        completed = 0

        for future in as_completed(futures):
            completed += 1
            self.update_progress_signal.emit(completed, total)

            result = future.result()
            if result['success']:
                self.detection_finished_signal.emit(result)

            with self.task_lock:
                self.running_tasks.discard(future)

    def update_progress_bar(self, completed, total):
        """更新进度条显示"""
        if completed == 1:
            self.progress_bar = ProgressBar(self)
            self.progress_bar.show()

        value = int(completed / total * 100)
        self.progress_bar.setValue(completed, total, value)

        if completed >= total:
            self.progress_bar.close()
            QMessageBox.about(
                self, 'Tip', 'All images processed successfully and detection pictures saved in the {}.'.format(
                    os.path.dirname(self.org_path[0]))
            )

    def update_ui_with_result(self, result_data):
        """根据检测结果更新UI"""
        # 在主线程中更新UI
        if not result_data['success']:
            return

        location = result_data['results'].obb.xywhr

        if len(location) != 0:

            cls_list = result_data['results'].obb.cls.tolist()
            cls_list = np.array([int(i) for i in cls_list])
            stoma_index = cls_list == 0
            aperture_index = cls_list == 1

            stoma_wh_ratio = location[stoma_index, 3] / location[stoma_index, 2]
            stoma_wh_ratio = stoma_wh_ratio.mean()

            stoma_average_w = location[stoma_index, 2].mean() / 224 * float(self.ui.scale.text())
            stoma_average_h = location[stoma_index, 3].mean() / 224 * float(self.ui.scale.text())

            aperture_wh_ratio = location[aperture_index, 3] / location[aperture_index, 2]
            aperture_wh_ratio = aperture_wh_ratio.mean()
            aperture_average_w = location[aperture_index, 2].mean() / 224 * float(self.ui.scale.text())
            aperture_average_h = location[aperture_index, 3].mean() / 224 * float(self.ui.scale.text())

            number = stoma_index.sum()
            aperture_count = aperture_index.sum()

            img_height, img_width = result_data['results'].orig_shape
            # scale = float(self.ui.scale.text())
            image_area = (
                    (img_height / 224) * (img_width / 224) * (float(self.ui.scale.text()) / 1000) ** 2)  # mm²
            stoma_density = number / image_area

            alpha_mean = (aperture_average_h / 2) ** 2 * torch.pi

            # condance_max = (24.9 * 1e-6) * stoma_density * alpha_mean / (1.6 * (22.4 * 1e-3) *
            #                                                                 (location[aperture_index,
            #                                                                  3].mean() + torch.sqrt(
            #                                                                     alpha_mean * torch.pi / 4))
            #                                                                 )
            condance_mean = (24.9 * 1e-6) * stoma_density * alpha_mean / (1.6 * (22.4 * 1e-3) *
                                                                          (
                                                                                  aperture_average_w + torch.sqrt(
                                                                              alpha_mean * torch.pi / 4)
                                                                          )
                                                                          )

            self.ui.density.setText('{:.4f}'.format(stoma_density))
            self.ui.conductance.setText('{:.4f}'.format(condance_mean))
            self.ui.stoma_aspect.setText('{:.4f}'.format(stoma_wh_ratio))
            self.ui.stoma_length.setText('{:.4f}'.format(stoma_average_w))
            self.ui.stoma_width.setText('{:.4f}'.format(stoma_average_h))
            self.ui.aperture_aspect.setText('{:.4f}'.format(aperture_wh_ratio))
            self.ui.aperture_length.setText('{:.4f}'.format(aperture_average_w))
            self.ui.aperture_width.setText('{:.4f}'.format(aperture_average_h), )

            self.save_results.append(
                [
                    result_data['img_path'].split("\\")[-1].split('.')[0],
                    stoma_average_h.cpu().numpy(),
                    stoma_average_w.cpu().numpy(),
                    stoma_wh_ratio.cpu().numpy(),
                    aperture_average_h.cpu().numpy(),
                    aperture_average_w.cpu().numpy(),
                    aperture_wh_ratio.cpu().numpy(),
                    stoma_density,
                    condance_mean.cpu().numpy(),
                    number.item(),
                    aperture_count.item(),
                    image_area
                ]
            )
        else:
            self.ui.density.setText('No Detections')
            self.ui.conductance.setText('No Detections')
            self.ui.stoma_aspect.setText('No Detections')
            self.ui.stoma_length.setText('No Detections')
            self.ui.stoma_width.setText('No Detections')
            self.ui.aperture_aspect.setText('No Detections')
            self.ui.aperture_length.setText('No Detections')
            self.ui.aperture_width.setText('No Detections')
            self.save_results.append(
                [
                    result_data['img_path'].split('\\')[-1].split('.')[0],
                    'nan',
                    'nan',
                    'nan',
                    'nan',
                    'nan',
                    'nan',
                    'nan',
                    'nan',
                    'nan',
                    'nan',
                    'nan'
                ]
            )

        draw_seg = False
        draw_box = True
        now_img = result_data['results'].plot(boxes=draw_box, masks=draw_seg, conf=False, labels=False, line_width=1,)

        # 保存图片
        cv2.imwrite(
            os.path.dirname(result_data['img_path']) + '\\' + os.path.basename(result_data['img_path']).split('.')[0]
            + '_detection.' + os.path.basename(result_data['img_path']).split('.')[-1], now_img)

        # 获取缩放后的图片尺寸
        height, width, channel = now_img.shape
        bytes_per_line = 3 * width
        qimg = QImage(now_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg)

        # 使用 Qt 的 scaled 方法等比例缩放
        scaled_pixmap = pixmap.scaled(self.show_width, self.show_height,
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # 将缩放后的图片设置到 QLabel
        self.ui.output_data.setPixmap(scaled_pixmap)
        self.ui.output_data.setAlignment(Qt.AlignCenter)  # 确保 QLabel 居中对齐

        # 绘制窗口1图片
        height, width, channel = result_data['org_img'].shape
        bytes_per_line = 3 * width
        qimg = QImage(result_data['org_img'].data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg)

        # 使用 Qt 的 scaled 方法等比例缩放
        scaled_pixmap = pixmap.scaled(self.show_width, self.show_height,
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # 将缩放后的图片设置到 QLabel
        self.ui.input_data.setPixmap(scaled_pixmap)
        self.ui.input_data.setAlignment(Qt.AlignCenter)  # 确保 QLabel 居中对齐

        QApplication.processEvents()  # 刷新页面

    def __del__(self):
        """清理线程池"""
        self.thread_pool.shutdown(wait=False)
        with self.task_lock:
            for task in self.running_tasks:
                task.cancel()

    def get_video_path(self):
        file_path, _ = QFileDialog.getOpenFileName(None, 'Open Video', './', "Image files (*.avi *.mp4 *.wmv *.mkv)")
        if not file_path:
            return None
        self.org_path = file_path
        return file_path

    def video_start(self):

        self.timer_camera.start(10)

        self.save_results = [
            ['stomata average height (um)', 'stomata average width (um)', 'stomata aspect ratio',
             'aperture average height (um)', 'aperture average width (um)', 'aperture aspect ratio',
             'stomata density (stomata * mm-2)', 'conductance (mol m-2 s-1)', 'stoma count', 'aperture count',
             'image_area']]

        self.timer_camera.timeout.connect(self.open_frame)

    def video_stop(self):
        self.cap.release()
        self.timer_camera.stop()
        # self.timer_info.stop()

    def open_frame(self):
        if self.cap is None:
            return
        ret, now_img = self.cap.read()

        if ret:
            # 目标检测
            # t1 = time.time()
            results = self.model(now_img, conf=self.conf, iou=self.iou)[0]
            # t2 = time.time()
            # take_time_str = '{:.3f} s'.format(t2 - t1)
            # self.ui.time_lb.setText(take_time_str)

            location = results.obb.xywhr

            if len(location) != 0:

                cls_list = results.obb.cls.tolist()
                cls_list = np.array([int(i) for i in cls_list])
                stoma_index = cls_list == 0
                aperture_index = cls_list == 1

                stoma_wh_ratio = location[stoma_index, 3] / location[stoma_index, 2]
                stoma_wh_ratio = stoma_wh_ratio.mean()
                stoma_average_w = location[stoma_index, 2].mean() / 224 * float(self.ui.scale.text())
                stoma_average_h = location[stoma_index, 3].mean() / 224 * float(self.ui.scale.text())

                aperture_wh_ratio = location[aperture_index, 3] / location[aperture_index, 2]
                aperture_wh_ratio = aperture_wh_ratio.mean()
                aperture_average_w = location[aperture_index, 2].mean() / 224 * float(self.ui.scale.text())
                aperture_average_h = location[aperture_index, 3].mean() / 224 * float(self.ui.scale.text())

                # number = max(stoma_index.sum(), aperture_index.sum())
                number = stoma_index.sum()
                aperture_count = aperture_index.sum()
                img_height, img_width = results.orig_shape
                image_area = (img_height / 224) * (img_width / 224) * (float(self.ui.scale.text()) / 1000) ** 2

                stoma_density = number / image_area

                # alpha_mean = (location[aperture_index, 2].mean() / 2) ** 2 * torch.pi
                alpha_mean = (aperture_average_h / 2) ** 2 * torch.pi

                # condance_max = (24.9 * 1e-6) * stoma_density * alpha_mean / (1.6 * (22.4 * 1e-3) *
                #                                                                 (location[aperture_index,
                #                                                                  3].mean() + torch.sqrt(
                #                                                                     alpha_mean * torch.pi / 4))
                #                                                                 )
                condance_mean = (24.9 * 1e-6) * stoma_density * alpha_mean / (1.6 * (22.4 * 1e-3) *
                                                                              (
                                                                                      aperture_average_w + torch.sqrt(
                                                                                  alpha_mean * torch.pi / 4)
                                                                              )
                                                                              )
                self.ui.density.setText('{:.4f}'.format(stoma_density))
                self.ui.conductance.setText('{:.4f}'.format(condance_mean))
                self.ui.stoma_aspect.setText('{:.4f}'.format(stoma_wh_ratio))
                self.ui.stoma_length.setText('{:.4f}'.format(stoma_average_w))
                self.ui.stoma_width.setText('{:.4f}'.format(stoma_average_h))
                self.ui.aperture_aspect.setText('{:.4f}'.format(aperture_wh_ratio))
                self.ui.aperture_length.setText('{:.4f}'.format(aperture_average_w))
                self.ui.aperture_width.setText('{:.4f}'.format(aperture_average_h))

                self.save_results.append(
                    [
                        stoma_average_h.cpu().numpy(),
                        stoma_average_w.cpu().numpy(),
                        stoma_wh_ratio.cpu().numpy(),
                        aperture_average_h.cpu().numpy(),
                        aperture_average_w.cpu().numpy(),
                        aperture_wh_ratio.cpu().numpy(),
                        stoma_density,
                        condance_mean.cpu().numpy(),
                        number.item(),
                        aperture_count.item(),
                        image_area
                    ]
                )
            else:
                self.ui.density.setText('No Detections')
                self.ui.conductance.setText('No Detections')
                self.ui.stoma_aspect.setText('No Detections')
                self.ui.stoma_length.setText('No Detections')
                self.ui.stoma_width.setText('No Detections')
                self.ui.aperture_aspect.setText('No Detections')
                self.ui.aperture_length.setText('No Detections')
                self.ui.aperture_width.setText('No Detections')
                self.save_results.append(
                    [
                        'nan',
                        'nan',
                        'nan',
                        'nan',
                        'nan',
                        'nan',
                        'nan',
                        'nan',
                        'nan',
                        'nan',
                        'nan'
                    ]
                )

            # 绘制窗口2图片
            draw_seg = False
            draw_box = True
            det_img = results.plot(boxes=draw_box, masks=draw_seg, conf=False, labels=False)

            # 获取缩放后的图片尺寸
            img_width, img_height = self.get_resize_size(det_img)
            resize_cvimg = cv2.resize(det_img, (img_width, img_height), interpolation=cv2.INTER_AREA)
            pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
            self.ui.output_data.setPixmap(pix_img)
            self.ui.output_data.setAlignment(Qt.AlignCenter)

            # 绘制窗口1图片
            # win2_img = self.draw_seg_mask(results)
            resize_cvimg = cv2.resize(now_img, (img_width, img_height), interpolation=cv2.INTER_AREA)
            pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
            self.ui.input_data.setPixmap(pix_img)
            self.ui.input_data.setAlignment(Qt.AlignCenter)

            if self.save_camera_flag is True:
                self.count_nums += 1
                self.CameraSave(results, self.count_nums, fps=20)
        else:
            self.cap.release()
            self.timer_camera.stop()
            self.ui.select_videos.setText('Video Start')

    def video_show(self):

        if self.using_ir:
            # warnings.warn('Currently we are not supporting video super-resolution in detection.',
            #               category=None, stacklevel=1, source=None)
            QMessageBox.warning(self, "Warning", "Currently we are not supporting video restoration in detection.")
            return

        if self.is_camera_open:
            self.is_camera_open = False
            self.ui.open_camera.setText('Camera On')
            if self.cap and self.cap.isOpened():
                self.cap.release()
                cv2.destroyAllWindows()

        if self.cap and self.cap.isOpened():
            # 关闭视频
            self.ui.select_videos.setText('Video Start')
            self.ui.output_data.setText('')
            self.ui.input_data.setText('')
            self.cap.release()
            cv2.destroyAllWindows()
            self.ui.output_data.clear()
            self.ui.input_data.clear()
            return

        video_path = self.get_video_path()
        if not video_path:
            return None
        self.ui.select_videos.setText('Video End')
        self.cap = cv2.VideoCapture(video_path)
        self.video_start()
        # self.ui.comboBox.setDisabled(True)

    def camera_show(self):

        if self.using_ir:
            # warnings.warn('Currently we are not supporting video super-resolution in detection.',
            #               category=None, stacklevel=1, source=None)
            QMessageBox.warning(self, "Warning", "Currently we are not supporting real-time restoration detection.")
            return

        self.is_camera_open = not self.is_camera_open
        if self.is_camera_open:
            self.ui.select_videos.setText('Video Start')
            self.ui.open_camera.setText('Camera Off')
            self.cap = cv2.VideoCapture(0)
            self.video_start()
            # self.ui.comboBox.setDisabled(True)
        else:
            self.ui.open_camera.setText('Camera On')
            self.ui.output_data.setText('')
            self.ui.input_data.setText('')
            if self.cap:
                self.cap.release()
                cv2.destroyAllWindows()
            self.cap = None
            self.ui.output_data.clear()
            self.ui.input_data.clear()

    def get_resize_size(self, img):
        """
        计算保持宽高比的缩放尺寸（不裁剪）

        参数:
            img: 输入图像(numpy数组)

        返回:
            (width, height): 保持原图比例的目标尺寸
        """
        img_height, img_width = img.shape[:2]
        target_width, target_height = self.show_width, self.show_height

        # 计算缩放比例
        scale = min(target_width / img_width, target_height / img_height)

        # 计算新尺寸（保持宽高比）
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        return new_width, new_height

    def save_detect_result(self):
        # 保存图片，视频及摄像头检测结果
        if self.cap is None and not self.org_path:
            QMessageBox.about(self, 'Tip',
                              'Currently there is  no results to save. Please open an image or video first.')
            return

        self.save_dir = QFileDialog.getExistingDirectory(self, "Select Saving Directory", "./")

        if not self.save_dir:
            return

        if self.is_camera_open:
            # 保存摄像头检测结果
            res = QMessageBox.information(self, 'Tip',
                                          'Please confirm whether to start saving camera detection results?',
                                          QMessageBox.Yes | QMessageBox.No,
                                          QMessageBox.Yes)
            if res == QMessageBox.Yes:
                # 初始化存储结果
                self.save_camera_flag = True
                self.count_nums = 0
                fps = 20
                width = int(self.cap.get(3))
                height = int(self.cap.get(4))
                size = (width, height)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 为视频编码方式，保存为avi格式
                save_camera_path = os.path.join(self.save_dir, 'camera.avi')
                # mask_save_camera_path = os.path.join(Config.save_path, 'camera_mask.avi')
                # seg_save_camera_path = os.path.join(Config.save_path, 'camera_seg.avi')
                self.out = cv2.VideoWriter(save_camera_path, fourcc, fps, size)
                # self.mask_out = cv2.VideoWriter(mask_save_camera_path, fourcc, fps, size)
                # self.seg_out = cv2.VideoWriter(seg_save_camera_path, fourcc, fps, size)
                if len(self.save_results) == 1:
                    pass
                else:
                    # result = np.array(self.save_results)
                    df = pd.DataFrame(data=self.save_results[1:], columns=self.save_results[0])
                    df.to_csv(f'{self.save_dir}/camera_results.csv', encoding='utf-8', index=False)
                    # np.savetxt(f'{self.save_dir}/results.csv', result, delimiter=',', fmt="%s")
            else:
                return
            return

        if self.cap:
            # 保存视频
            res = QMessageBox.information(self, 'Tip',
                                          'Please confirm whether to start saving the video detection result.',
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if res == QMessageBox.Yes:
                self.video_stop()
                self.ui.select_videos.setText('Video Start')
                # com_text = self.ui.comboBox.currentText()
                self.btn2Thread_object = btn2Thread(self.org_path, self.model, self.conf, self.iou,
                                                    savedir=self.save_dir)
                self.btn2Thread_object.start()
                self.btn2Thread_object.update_ui_signal.connect(self.update_process_bar)
                if len(self.save_results) == 1:
                    pass
                else:
                    # result = np.array(self.save_results)
                    # np.savetxt(f'{self.save_dir}/results.csv', result, delimiter=',', fmt="%s")
                    df = pd.DataFrame(data=self.save_results[1:], columns=self.save_results[0])
                    df.to_csv(f'{self.save_dir}/video_results.csv', encoding='utf-8', index=False)
            else:
                return
        else:
            if len(self.save_results) == 1:
                pass
            else:
                # result = np.array(self.save_results)
                # np.savetxt(f'{self.save_dir}/results.csv', result, delimiter=',', fmt="%s")
                df = pd.DataFrame(data=self.save_results[1:], columns=self.save_results[0])
                df.to_csv(f'{self.save_dir}/results.csv', encoding='utf-8', index=False)

    def update_process_bar(self, cur_num, total):
        # 更新保存的进度条
        if cur_num == 1:
            self.progress_bar = ProgressBar(self)
            self.progress_bar.show()
        if cur_num >= total:
            self.progress_bar.close()
            QMessageBox.about(self, 'Tip',
                              'Video saved successfully!\nThe file is located in the {}.'.format(self.save_dir))
            return
        if self.progress_bar.isVisible() is False:
            # 点击取消保存时，终止进程
            self.btn2Thread_object.stop()
            return
        value = int(cur_num / total * 100)
        self.progress_bar.setValue(cur_num, total, value)
        QApplication.processEvents()

    def update_camera_process(self, cur_num, fps=30):
        # 更新摄像头存储结果的弹窗信息
        if cur_num == 1:
            self.progress = CamProgress(self)
            self.progress.show()
        if self.progress.isVisible() is False:
            # 点击取消保存时，终止进程
            self.save_camera_flag = False
            self.out.release()
            # self.mask_out.release()
            # self.seg_out.release()
            self.progress.close()
            return
        self.progress.setValue(cur_num, fps)
        QApplication.processEvents()

    def CameraSave(self, results, nums, fps=30):
        # 摄像头检测结果存储
        # print('已存储{}帧，时长{}s'.format(nums, round(nums / fps, 2)))
        frame = results.plot(conf=False, labels=False)
        # img_mask_res, img_seg_res = self.get_mask_and_seg(results)
        self.out.write(frame)
        # self.mask_out.write(img_mask_res)
        # self.seg_out.write(img_seg_res)
        self.update_camera_ui_signal.emit(nums)

    @staticmethod
    def check_device(device: str) -> str:
        if device == "cuda":
            if not torch.cuda.is_available():
                # print("CUDA not available because the current PyTorch install was not "
                #       "built with CUDA enabled.")
                device = "cpu"
        else:
            if device == "mps":
                if not torch.backends.mps.is_available():
                    if not torch.backends.mps.is_built():

                        device = "cpu"
                    else:

                        device = "cpu"
        # print(f"using device {device}")
        return device


class btn2Thread(QThread):
    """
    进行检测后的视频保存
    """
    # 声明一个信号
    update_ui_signal = pyqtSignal(int, int)

    def __init__(self, path, model, conf, iou, savedir):
        super(btn2Thread, self).__init__()
        self.org_path = path
        self.model = model
        self.com_text = None
        self.conf = conf
        self.iou = iou
        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()
        self.is_running = True  # 标志位，表示线程是否正在运行
        self.savedir = savedir

    def run(self):
        # VideoCapture方法是cv2库提供的读取视频方法
        cap = cv2.VideoCapture(self.org_path)
        # 设置需要保存视频的格式“xvid”
        # 该参数是MPEG-4编码类型，文件名后缀为.avi
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # 设置视频帧频
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 设置视频大小
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # VideoWriter方法是cv2库提供的保存视频方法
        # 按照设置的格式来out输出
        fileName = os.path.basename(self.org_path)
        name, end_name = fileName.split('.')
        save_name = name + '_detect_result.avi'
        save_video_path = os.path.join(self.savedir, save_name)
        out = cv2.VideoWriter(save_video_path, fourcc, fps, size)

        prop = cv2.CAP_PROP_FRAME_COUNT
        total = int(cap.get(prop))
        # print("[INFO] 视频总帧数：{}".format(total))
        cur_num = 0

        # 确定视频打开并循环读取
        while cap.isOpened() and self.is_running:
            cur_num += 1
            # print('当前第{}帧，总帧数{}'.format(cur_num, total))
            # 逐帧读取，ret返回布尔值
            # 参数ret为True 或者False,代表有没有读取到图片
            # frame表示截取到一帧的图片
            ret, frame = cap.read()
            if ret is True:
                # 检测
                results = self.model(frame, conf=self.conf, iou=self.iou)[0]
                frame = results.plot(conf=False, labels=False)
                # img_mask_res, img_seg_res = self.get_mask_and_seg(results)
                out.write(frame)
                self.update_ui_signal.emit(cur_num, total)
            else:
                break
        # 释放资源
        cap.release()
        out.release()

    def stop(self):
        # 停止保存
        self.is_running = False


if __name__ == "__main__":
    # 对于按钮文字显示不全的，完成高清屏幕自适应设置
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    win = MainWindow()
    # win.show()
    sys.exit(app.exec_())
