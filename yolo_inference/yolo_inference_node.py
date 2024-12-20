import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

class YOLOInferenceNode(Node):
    def __init__(self):
        super().__init__('yolo_inference_node')
        self.br = CvBridge()
        self.model = YOLO('data/peacock/runs/detect/train4/weights/best.pt')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(Detection2DArray, '/detections', 10)

    def image_callback(self, msg):
        cv_img = self.br.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model.predict(cv_img, conf=0.5)

        detections_msg = Detection2DArray()
        detections_msg.header = msg.header

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cls_idx = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls_idx]

                detection = Detection2D() 
                detection.bbox.center.position.x = float((x1 + x2) / 2.0)
                detection.bbox.center.position.y = float((y1 + y2) / 2.0)
                detection.bbox.size_x = float(x2 - x1)
                detection.bbox.size_y = float(y2 - y1)

                hypothesisWithPose = ObjectHypothesisWithPose()
            
                # Convert cls_idx to a string
                cls_idx_str = str(cls_idx)
                hypothesisWithPose.hypothesis.class_id = cls_idx_str
                hypothesisWithPose.hypothesis.score = conf
                # additional info can be included here if needed

                detection.results.append(hypothesisWithPose)
                detections_msg.detections.append(detection)

        # Publish the detection results
        self.publisher.publish(detections_msg)

        
def main(args=None):
    rclpy.init(args=args)
    node = YOLOInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    # ros2 run yolo_inference yolo_inference_node
    

if __name__ == '__main__':
    main()
    