import sdl2
import sdl2.ext
import cv2 

class Display(object):
    def __init__(self,W,H):
        sdl2.ext.init()
        self.W,self.H=W,H
        self.window=sdl2.ext.Window('SLAM',size=(W,H))
        self.window.show()

    def point(self,img):
        events=sdl2.ext.get_events()
        for event in events:
            if event.type==sdl2.SDL_QUIT:
                exit(0)
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        surf=sdl2.ext.pixels3d(self.window.get_surface())
        surf[:,:,0:3]=rgb.swapaxes(0,1)
        self.window.refresh()


