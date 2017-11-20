import cv2
import numpy as np


class BirdsEyePerspectiveTxfrm():
    #http://www.coldvision.io/2017/03/02/advanced-lane-finding-using-opencv/
    def compute_perspective_transform(self, binary_image):
        # Define 4 source and 4 destination points = np.float32([[,],[,],[,],[,]])
        shape = binary_image.shape[::-1] # (width,height)
        w = shape[0]
        h = shape[1]
        transform_src = np.float32([ [580,450], [160,h], [1150,h], [740,450]])
        transform_dst = np.float32([ [0,0], [0,h], [w,h], [w,0]])
        M = cv2.getPerspectiveTransform(transform_src, transform_dst)
        return M

    def run(self, img_arr):
        M = self.compute_perspective_transform(img_arr)
        return cv2.warpPerspective(img_arr, M, (img_arr.shape[1], img_arr.shape[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image


class AdaptiveThreshold():

    def __init__(self, high_threshold=255):
        self.high_threshold = high_threshold

    def run(self, img_arr):
        return cv2.adaptiveThreshold(img_arr, self.high_threshold,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 115, 1)


class DrawLine():
    
    def __init__(self, line_start, line_finish):
        self.line_start = line_start
        self.line_finish = line_finish
    
    def run(self, img_arr):
        return cv2.line(img_arr, self.line_start, self.line_finish, (255,0,0), 5)


#Luke's NOTE: Simulator fundamentally doesn't work in a way to support this :(
# class SimulatorLink:
#     """
#     Wrapper around SteeringServer, which allows us to place the Simulator
#     in the Vehicle event loop/pipeline.
#     """

#     def __init__(self, resolution=(120,160), box_size=4, color=(255, 0, 0)):
#         self.sio = socketio.Server()
#         self.timer = FPSTimer()
#         self.top_speed = float(3)
#         #Start websocket server
#         self._go(('0.0.0.0', 9090))

#     def _go(self, address):

#         # wrap Flask application with engineio's middleware
#         self.app = socketio.Middleware(self.sio, self.app)

#         # deploy as an eventlet WSGI server
#         try:
#             eventlet.wsgi.server(eventlet.listen(address), self.app)

#         except KeyboardInterrupt:
#             # unless some hits Ctrl+C and then we get this interrupt
#             print('stopping')

#     def _connect(self, sid, environ):
#         print("connect ", sid)
#         self.timer.reset()
#         self.send_control(0, 0)

#     def _throttle_control(self, last_steering, last_throttle, speed, nn_throttle):
#         '''
#         super basic throttle control, derive from this Server and override as needed
#         '''
#         if speed < self.top_speed:
#             return 0.3

#         return 0.0

#     def _telemetry(self, sid, data):
#         '''
#         Callback when we get new data from Unity simulator.
#         We use it to process the image, do a forward inference,
#         then send controls back to client.
#         Takes sid (?) and data, a dictionary of json elements.
#         '''
#         if data:
#             # The current steering angle of the car
#             last_steering = float(data["steering_angle"])

#             # The current throttle of the car
#             last_throttle = float(data["throttle"])

#             # The current speed of the car
#             speed = float(data["speed"])

#             # The current image from the center camera of the car
#             imgString = data["image"]

#             # decode string based data into bytes, then to Image
#             image = Image.open(BytesIO(base64.b64decode(imgString)))

#             # then as numpy array
#             image_array = np.asarray(image)

#             # optional change to pre-preocess image before NN sees it
#             if self.image_part is not None:
#                 image_array = self.image_part.run(image_array)

#             # forward pass - inference
#             steering, throttle = self.kpart.run(image_array)

#             # filter throttle here, as our NN doesn't always do a greate job
#             throttle = self._throttle_control(last_steering, last_throttle, speed, throttle)

#             # simulator will scale our steering based on it's angle based input.
#             # but we have an opportunity for more adjustment here.
#             steering *= self.steering_scale

#             # send command back to Unity simulator
#             self.send_control(steering, throttle)

#         else:
#             # NOTE: DON'T EDIT THIS.
#             self.sio.emit('manual', data={}, skip_sid=True)

#         self.timer.on_frame()

#     def _send_control(self, steering_angle, throttle):
#         self.sio.emit(
#             "steer",
#             data={
#                 'steering_angle': steering_angle.__str__(),
#                 'throttle': throttle.__str__()
#             },
#             skip_sid=True)

#     def update(self):
#         sio = self.sio
#         @sio.on('telemetry')
#         self._telemetry(sid, data)
#         # def telemetry(sid, data):
#         #     self.telemetry(sid, data)

#         @sio.on('connect')
#         self._connect(sid, environ)
#         # def connect(sid, environ):
#         #     self.connect(sid, environ)

#     def run_threaded(self):
#         return self.data


### Below are official DOnkeycar cv functions


class ImgGreyscale():

    def run(self, img_arr):
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        return img_arr



class ImgCanny():

    def __init__(self, low_threshold=60, high_threshold=110):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
        
    def run(self, img_arr):
        return cv2.Canny(img_arr, 
                         self.low_threshold, 
                         self.high_threshold)

    

class ImgGaussianBlur():

    def __init__(self, kernal_size=5):
        self.kernal_size = kernal_size
        
    def run(self, img_arr):
        return cv2.GaussianBlur(img_arr, 
                                (self.kernel_size, self.kernel_size), 0)



class ImgCrop:
    """
    Crop an image to an area of interest. 
    """
    def __init__(self, top=0, bottom=0, left=0, right=0):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        
    def run(self, img_arr):
        width, height, _ = img_arr.shape
        img_arr = img_arr[self.top:height-self.bottom, 
                          self.left: width-self.right]
        return img_arr
        


class ImgStack:
    """
    Stack N previous images into a single N channel image, after converting each to grayscale.
    The most recent image is the last channel, and pushes previous images towards the front.
    """
    def __init__(self, num_channels=3):
        self.img_arr = None
        self.num_channels = num_channels

    def rgb2gray(self, rgb):
        '''
        take a numpy rgb image return a new single channel image converted to greyscale
        '''
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        
    def run(self, img_arr):
        width, height, _ = img_arr.shape        
        gray = self.rgb2gray(img_arr)
        
        if self.img_arr is None:
            self.img_arr = np.zeros([width, height, self.num_channels], dtype=np.dtype('B'))

        for ch in range(self.num_channels - 1):
            self.img_arr[...,ch] = self.img_arr[...,ch+1]

        self.img_arr[...,self.num_channels - 1:] = np.reshape(gray, (width, height, 1))

        return self.img_arr

        
        
class Pipeline():
    def __init__(self, steps):
        self.steps = steps
    
    def run(self, val):
        for step in self.steps:
            f = step['f']
            args = step['args']
            kwargs = step['kwargs']
            
            val = f(val, *args, **kwargs)
        return val
    