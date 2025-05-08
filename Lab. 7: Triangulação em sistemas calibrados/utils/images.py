from is_wire.core import Subscription
from is_msgs.image_pb2 import Image
import numpy as np
import cv2

from .StreamChannel import StreamChannel


def to_np(input_image) -> np.ndarray:
    if isinstance(input_image, np.ndarray):
        output_image = input_image
    elif isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    else:
        output_image = np.array([], dtype=np.uint8)
    return output_image

def get_image(camera_id: int) -> None:
    broker_uri = "amqp://guest:guest@10.10.2.211:30000"
    channel = StreamChannel(broker_uri)
    subscription = Subscription(channel)
    subscription.subscribe(topic='CameraGateway.{}.Frame'.format(camera_id))
    while True:
        msg = channel.consume_last()  
        if type(msg) != bool: 
            img = msg.unpack(Image)
            frame = to_np(img)
            cv2.imshow(f"{camera_id}", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return