import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YOLOInferenceNode(Node):
    def __init__(self):
        super().__init__('yolo_inference_node')
        self.br = CvBridge()
        self.model = YOLO('runs/detect/train4/weights/best.pt')  # load trained model
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        cv_img = self.br.imgmsg_to_cv2(msg, 'bgr8')
        # YOLO expects images as numpy arrays (HWC, BGR)
        results = self.model.predict(cv_img, conf=0.5)  # conf threshold optional
        
        # Extract bounding boxes, classes, and confidence scores
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cls_idx = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls_idx]
                
                # Draw on the cv_img if needed (for debugging)
                cv2.rectangle(cv_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(cv_img, f"{class_name} {conf:.2f}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        # Optionally publish the annotated image or detection results as a new topic
        # E.g., publish annotated image:
        annotated_msg = self.br.cv2_to_imgmsg(cv_img, 'bgr8')
        # You'd need a publisher defined for this:
        # self.publisher_.publish(annotated_msg)
        
def main(args=None):
    rclpy.init(args=args)
    node = YOLOInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    