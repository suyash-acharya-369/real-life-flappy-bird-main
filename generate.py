import numpy as np
import constants
import cv2

class Generate:
    def __init__(self, height, width):
        # pipe: [[(x: position, y: top, y: bottom)]]
        self.pipes = []
        self.height = height
        self.width = width
        self.points = 0
        # bright color palette for pipes (B, G, R)
        self.colors = [
            (0, 255, 0),      # lime
            (0, 255, 255),    # yellow
            (255, 0, 255),    # magenta
            (255, 128, 0),    # orange
            (0, 128, 255),    # deep sky
            (180, 0, 255),    # pink
            (0, 0, 255),      # red
            (255, 0, 0)       # blue
        ]

    def create(self):
        rand_y_top = np.random.randint(0, self.height - constants.GAP)
        color = self.colors[np.random.randint(0, len(self.colors))]
        # each pipe: [x, y_top, y_bottom, passed, color]
        self.pipes.append([self.width, rand_y_top, rand_y_top + constants.GAP, False, color])
    
    def draw_pipes(self, frm):
        for i in self.pipes:
            if(i[0] <= 0 ):
                continue
            color = i[4] if len(i) > 4 else (0, 255, 0)
            # top pipe (filled)
            cv2.rectangle(frm, (i[0], 0), (i[0] + constants.PIPE_WIDTH, i[1]), color, -1)
            # bottom pipe (filled) — fix height usage
            cv2.rectangle(frm, (i[0], i[2]), (i[0] + constants.PIPE_WIDTH, self.height), color, -1)
            # crisp outlines (slightly darker border)
            border_color = (max(color[0]-40, 0), max(color[1]-40, 0), max(color[2]-40, 0))
            cv2.rectangle(frm, (i[0], 0), (i[0] + constants.PIPE_WIDTH, i[1]), border_color, 3)
            cv2.rectangle(frm, (i[0], i[2]), (i[0] + constants.PIPE_WIDTH, self.height), border_color, 3)
    
    def update(self):
        for i in self.pipes:
            i[0] -= constants.SPEED
            if(i[0] <= 0):
                self.pipes.remove(i)
    
    def check(self, index_pt):
        for i in self.pipes:
            if (index_pt[0] >= i[0] and index_pt[0] <= i[0]+constants.PIPE_WIDTH):
                if ((index_pt[1] <= i[1]) or (index_pt[1] >= i[2])):
                    return True
                else:
                    if not(i[3]):
                        i[3] = True
                        self.points += 1
                        if(self.points % 10 == 0):
                            constants.SPEED += 4
                            constants.GEN_TIME -= 0.2
                    return False
        return False
