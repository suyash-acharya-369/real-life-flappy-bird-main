import mediapipe as mp 
import time
from generate import Generate
import constants 
import cv2
import numpy as np
import os
import ctypes

HS_FILE = "highscore.txt"
BIRD_IMG = "bird.png"  # save the provided image as bird.png in project root
CAM_INDEX = int(os.getenv("CAM_INDEX", "1"))  # external cam default
CAM_WIDTH = int(os.getenv("CAM_WIDTH", "1280"))
CAM_HEIGHT = int(os.getenv("CAM_HEIGHT", "720"))
CAM_FPS = int(os.getenv("CAM_FPS", "60"))

def load_high_score(path):
    if not os.path.exists(path):
        return 0
    try:
        with open(path, 'r') as f:
            return int(f.read().strip() or 0)
    except Exception:
        return 0

def save_high_score(path, score):
    try:
        with open(path, 'w') as f:
            f.write(str(score))
    except Exception:
        pass

def load_bird_sprite(path, target_height=60):
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # ensure BGRA
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    # scale keeping aspect ratio to target height
    h, w = img.shape[:2]
    scale = target_height / float(h)
    new_size = (max(int(w * scale), 1), target_height)
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img

def overlay_bgra_center(dst, sprite_bgra, center):
    if sprite_bgra is None:
        return
    h, w = sprite_bgra.shape[:2]
    x1 = int(center[0] - w / 2)
    y1 = int(center[1] - h / 2)
    x2 = x1 + w
    y2 = y1 + h
    # clip to frame
    H, W = dst.shape[:2]
    if x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H:
        return
    x1c, y1c = max(x1, 0), max(y1, 0)
    x2c, y2c = min(x2, W), min(y2, H)
    sx1, sy1 = x1c - x1, y1c - y1
    sx2, sy2 = sx1 + (x2c - x1c), sy1 + (y2c - y1c)
    roi = dst[y1c:y2c, x1c:x2c]
    spr = sprite_bgra[sy1:sy2, sx1:sx2]
    alpha = spr[:, :, 3:4] / 255.0
    rgb = spr[:, :, :3]
    roi[:] = (alpha * rgb + (1.0 - alpha) * roi).astype(np.uint8)

# open external camera with configurable settings; fallback to index 0 if needed
def open_camera(index, w, h, fps):
    c = cv2.VideoCapture(index)
    c.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    c.set(cv2.CAP_PROP_FPS, fps)
    ok, frame = c.read()
    return c, ok, frame

cap, ok, frm = open_camera(CAM_INDEX, CAM_WIDTH, CAM_HEIGHT, CAM_FPS)
used_cam_index = CAM_INDEX
if not ok or frm is None:
    # fallback to default internal camera
    if cap is not None:
        cap.release()
    cap, ok, frm = open_camera(0, CAM_WIDTH, CAM_HEIGHT, CAM_FPS)
    used_cam_index = 0
    if not ok or frm is None:
        raise RuntimeError("Unable to open any camera. Tried indexes {} and 0.".format(CAM_INDEX))
height_ = frm.shape[0]
width_ = frm.shape[1]
gen = Generate(height_, width_)
s_init = False
s_time = time.time()
is_game_over = False
high_score = load_high_score(HS_FILE)
bird_sprite = load_bird_sprite(BIRD_IMG, target_height=60)

# tracking lock and smoothing for responsiveness and single-person control
lock_active = False
lock_point = None
lock_radius = 160
filtered_pt = None
resp_beta = 0.7

#declarations
hand = mp.solutions.hands
# faster model with lower complexity improves latency
hand_model = hand.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
drawing = mp.solutions.drawing_utils

while True:
    ss = time.time()
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    # ensure fullscreen or large window
    try:
        cv2.namedWindow("window", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        # fallback: resize window to screen resolution
        try:
            screen_w = ctypes.windll.user32.GetSystemMetrics(0)
            screen_h = ctypes.windll.user32.GetSystemMetrics(1)
            cv2.namedWindow("window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("window", screen_w, screen_h)
        except Exception:
            pass

    cv2.putText(frm, "score: "+str(gen.points), (width_ - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (30, 30, 30), 3)
    cv2.putText(frm, "high: "+str(high_score), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (30, 30, 30), 3)
    cv2.putText(frm, "cam:{} {}x{}@{}".format(used_cam_index, CAM_WIDTH, CAM_HEIGHT, CAM_FPS), (50, height_ - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)

    # generate pipe every constants.TIME seconds
    if not(s_init):
        s_init = True 
        s_time = time.time()
    elif(time.time() - s_time) >= constants.GEN_TIME:
        s_init = False 
        gen.create()

    frm.flags.writeable = False
    res = hand_model.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    frm.flags.writeable = True

    #draw pipes && update there positions
    gen.draw_pipes(frm)
    gen.update()

    if res.multi_hand_landmarks:
        # hand is detected
        pts = res.multi_hand_landmarks[0].landmark
        # points = Points(pts)
        # grabbing index finger point
        index_pt = (int(pts[8].x * width_), int(pts[8].y * height_))

        # initialize lock on first detection
        if not lock_active:
            lock_active = True
            lock_point = index_pt
            filtered_pt = index_pt

        # maintain control only if current detection is close to lock
        if lock_active and lock_point is not None:
            dx = index_pt[0] - lock_point[0]
            dy = index_pt[1] - lock_point[1]
            if (dx*dx + dy*dy) <= (lock_radius * lock_radius):
                lock_point = index_pt
                filtered_pt = (
                    int(resp_beta * index_pt[0] + (1 - resp_beta) * filtered_pt[0]),
                    int(resp_beta * index_pt[1] + (1 - resp_beta) * filtered_pt[1])
                )
            # else: leave filtered_pt unchanged (ignore other hands)
        use_pt = filtered_pt if filtered_pt is not None else index_pt

        if gen.check(use_pt): 
            # GAME OVER
            is_game_over = True
            frm = cv2.cvtColor(frm, cv2.COLOR_BGR2HSV)
            frm = cv2.blur(frm, (10, 10))
            cv2.putText(frm, "GAME_OVER!", (100, 100), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 3)
            cv2.putText(frm, "Score : "+str(gen.points), (100, 180), cv2.FONT_HERSHEY_PLAIN, 4, (255,0,0), 3)
            if gen.points > high_score:
                high_score = gen.points
                save_high_score(HS_FILE, high_score)
                cv2.putText(frm, "New High Score!", (100, 260), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            gen.points = 0

        # bird: prefer sprite overlay; fallback to vector drawing if missing
        if bird_sprite is not None:
            overlay_bgra_center(frm, bird_sprite, use_pt)
        else:
            body_color = (40, 180, 255)
            wing_color = (0, 140, 255)
            beak_color = (0, 200, 255)
            eye_color = (255, 255, 255)
            pupil_color = (0, 0, 0)
            cv2.ellipse(frm, use_pt, (24, 18), 0, 0, 360, body_color, -1)
            cv2.ellipse(frm, (use_pt[0]-8, use_pt[1]), (12, 10), -20, 0, 300, wing_color, -1)
            beak = np.array([[use_pt[0]+22, use_pt[1]-2], [use_pt[0]+34, use_pt[1]+2], [use_pt[0]+22, use_pt[1]+6]])
            cv2.fillPoly(frm, [beak], beak_color)
            cv2.circle(frm, (use_pt[0]+6, use_pt[1]-6), 5, eye_color, -1)
            cv2.circle(frm, (use_pt[0]+6, use_pt[1]-6), 2, pupil_color, -1)
        # drawing.draw_landmarks(frm, res.multi_hand_landmarks[0], hand.HAND_CONNECTIONS)


    # cv2.putText(frm, "frame_rate: "+str(int(1/(time.time()-ss))), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("window", frm)
    
    if is_game_over:
        key_inp = cv2.waitKey(0)
        if(key_inp == ord('r')):
            is_game_over = False 
            gen.pipes = []
            constants.SPEED = 16
            constants.GEN_TIME = 1.2
            # reset lock after game over
            lock_active = False
            lock_point = None
            filtered_pt = None
        else :
            cv2.destroyAllWindows()
            cap.release()
            break

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break