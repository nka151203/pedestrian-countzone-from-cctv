import cv2
import numpy as np
import sys
import os
from pathlib import Path
from ultralytics import YOLO
import torch

from tracker.boost_track import BoostTrack

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PersonTracker:
    def __init__(self, model_path, video_path, output_path=None):
        """
        Initialize Person Tracker
        
        Args:
            model_path (str): Đường dẫn đến file model YOLOv11
            video_path (str): Đường dẫn video input
            output_path (str): Đường dẫn video output (optional)
        """
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.output_path = output_path
        self.set_in_roi = set()
        
        # Khởi tạo ByteTracker
        self.tracker = BoostTrack()
        
        # Màu sắc cho các track khác nhau
        self.colors = self._generate_colors(100)
    
    def _generate_colors(self, num_colors):
        """Tạo màu sắc ngẫu nhiên cho các track"""
        colors = []
        for i in range(num_colors):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors.append(color)
        return colors
    
    def _detection_to_boosttrack_format(self, results):
        """
        Chuyển đổi kết quả detection của YOLO sang format ByteTracker
        
        Args:
            results: Kết quả detection từ YOLO
            
        Returns:
            detections: Array numpy với format [x1, y1, x2, y2, score, class]
        """
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Chỉ lấy class "person" (class 0 trong COCO)
                    if int(box.cls) == 0:  # person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()
                        
                        # Format: [x1, y1, x2, y2, score, class]
                        detections.append([x1, y1, x2, y2, conf, cls])
        
        return np.array(detections) if detections else np.empty((0, 6))
    
    def draw_tracks(self, frame, tracks):
        """
        Vẽ bounding box và track ID lên MỘT frame
        
        Args:
            frame: Frame hiện tại
            tracks: Danh sách tracks từ ByteTracker
            
        Returns:
            frame: Frame đã vẽ
        """
        for track in tracks:
            # Lấy thông tin track trong tập tracks (chứa bounding box của tất cả đối tượng thuộc frame này)
            # tracks = (x1,y1,x2,y2,track_id, conf_score)
            x1, y1, x2, y2 = track[:4].astype(int)
            track_id = int(track[4])
            score = float(track[5])
            
            # Chọn màu cho track
            color = self.colors[track_id % len(self.colors)]
            
            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ track ID và confidence
            label = f"ID: {track_id} ({score:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Vẽ background cho text
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Vẽ text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def track_video(self, points):
        """
        points: Region Interesting Points 
        """
        # Mở video
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Không thể mở video: {self.video_path}")
            return
        
        # Lấy thông tin video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Thiết lập video writer nếu cần lưu output
        out = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                print(f"Processing frame {frame_count}/{total_frames}")
                
                # YOLO detection
                results = self.model(frame, verbose=False) #-> (Các object được detect dạng (top, left, width, height))

                # Chuyển đổi detection (top, left, width, height) sang format ByteTracker (x1,y1,x2,y2)
                detections = self._detection_to_boosttrack_format(results)
                
                # Update tracker
                if len(detections) > 0:
                    #Bước Update là bước áp dụng thuật toán, với tập detections vừa detect được, chúng ta thực hiện so sánh
                    #với tập detections ở frame trước -> áp dụng thuật toán để giữ lại id có thể là trùng để ta giữ lại id
                    #cũng như bỏ đi nhưng object đã biết mất
                    img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
                    img_tensor = img_tensor.unsqueeze(0)
                    tracks = self.tracker.update(detections, img_tensor, frame, f"frame_{frame_count}")
                    for track in tracks:
                        x1, y1, x2, y2 = track[:4].astype(int)
                        xb_1 = x1
                        yb_1 = y2
                        xb_2 = x2
                        yb_2 = y2
                        cbx = (xb_1+yb_1)/2 #center bottom X
                        cby = (yb_2+yb_1)/2 #center bottom Y
                        cby = cby - 0.1*cby
                        cb = (cbx, cby) #center bottom
                        track_id = int(track[4])
                        if check_point_int(points, cb):
                            self.set_in_roi.add(track_id)
                else:
                    tracks = []
                
                # Vẽ tracks lên frame hiện tại. Gán frame hiện tại với tập tracks hiện tại
                frame = self.draw_tracks(frame, tracks)
                overlay = frame.copy()
                polygon = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                color = (0, 165, 255)  # Cam (BGR)
                opacity = 0.2

                cv2.fillPoly(overlay, [polygon], color)
                # Trộn overlay với frame
                cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
                # Hiển thị số lượng tracks
                cv2.putText(frame, f"People was in your ROI: {len(self.set_in_roi)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Hiển thị frame
                cv2.imshow('Person Tracking', frame)
                
                # Lưu frame nếu cần
                if out:
                    out.write(frame)
                
                # Thoát nếu nhấn 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            print(f"Đã xử lý xong {frame_count} frames")

def main(model_path, video_path, output_path):
    # Cấu hình đường dẫn
    MODEL_PATH = model_path
    VIDEO_PATH = video_path            # Thay đổi đường dẫn video
    OUTPUT_PATH = output_path
    # Khởi tạo tracker
    tracker = PersonTracker(MODEL_PATH, VIDEO_PATH, OUTPUT_PATH)
    
    # Bắt đầu tracking
    print("Bắt đầu tracking...")
    tracker.track_video()
    print("Hoàn thành!")


def check_point_int(points, pt):
    pt = (float(pt[0]), float(pt[1]))
    # Tính centroid
    cx = np.mean([p[0] for p in points])
    cy = np.mean([p[1] for p in points])
    # Hàm tính góc so với centroid
    def angle(p):
        return np.arctan2(p[1]-cy, p[0]-cx)
    # Sắp xếp các điểm theo góc
    points_sorted = sorted(points, key=angle)
    # Tạo polygon từ list đã sort
    polygon = np.array(points_sorted, dtype=np.int32).reshape((-1, 1, 2))
    # Sử dụng cv2.pointPolygonTest
    result = cv2.pointPolygonTest(polygon, pt, False)
    if result >= 0:
        return True
    else:
        return False


# MODEL_PATH = "D:/python/countzone-from-cctv/tracking/epoch38.pt" # MODEL_PATH = "yolo11m.pt"
# VIDEO_PATH = "D:/python/countzone-from-cctv/tracking/video4.mp4"            
# OUTPUT_PATH = "D:/python/countzone-from-cctv/tracking/video_output.mp4"
# main(MODEL_PATH, VIDEO_PATH, OUTPUT_PATH)
    