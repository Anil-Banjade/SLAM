# SDL viewer for displaying refernce image or video used for that pose
import sdl2
import sdl2.ext
import cv2
import numpy as np 

class Display:
    def __init__(self, w, h): #w: width, h:height of the window
        try: 
            sdl2.ext.init()

            self.w = w
            self.h = h 
            self.window = sdl2.ext.Window('Reference frames', size = (self.w, self.h))
            self.window.show()
            self.surface = self.window.get_surface()
            self.counter = 0

        except Exception as e:
            print(f"error while Init sdl2: {e}")
            exit(1)

    def display(self, image):
        try:
            # print(f"sent to display {self.counter}")
            self.counter += 1
            # print(image.shape)
            # print(len(image.shape))
            if len(image.shape) == 2:  # Grayscale image
                rgb = np.stack((image,) * 3, axis=-1)  # Convert to 3D by duplicating the grayscale channel
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #surf to update pixels
            surf = sdl2.ext.pixels3d(self.surface)
            rgb_resized = cv2.resize(rgb, (self.w, self.h))
            
            #update surface pixels with iamge
            surf[:,:, 0:3] = rgb_resized.swapaxes(0,1)
            # print("refresh here")
            self.window.refresh()

            self.handle_events()

        except Exception as e:
            print(f"error while displaying sdl2: {e}")
            # exit(1)

    def handle_events(self):

        #handle sdl events: here closing window

        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                sdl2.ext.quit()
                # exit(0)
    
    def close(self):
        #close sdl window and clean up
        sdl2.ext.quit()