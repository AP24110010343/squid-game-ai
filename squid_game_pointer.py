import time, math, random, threading, sys, os, argparse
import cv2
import numpy as np

try:
    import simpleaudio as sa
    _HAS_AUDIO = True
except ImportError:
    sa = None; _HAS_AUDIO = False

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    YOLO = None; _HAS_YOLO = False

try:
    import mediapipe as mp
    _HAS_MP = True
except Exception:
    mp = None; _HAS_MP = False
  
parser = argparse.ArgumentParser(description="Squid Game — Red Light Green Light")
parser.add_argument("--detector", choices=["yolo","mp"], default="yolo")
parser.add_argument("--model", default="yolov8n.mlpackage")
parser.add_argument("--colorblind", action="store_true")
parser.add_argument("--frameskip", type=int, default=1)
parser.add_argument("--mode", choices=["classic","hardcore","training"], default="classic")
parser.add_argument("--no-sound", action="store_true")
args = parser.parse_args()

DETECTOR   = args.detector
MODEL_PATH = args.model
CONF       = 0.35
IMG_SIZE   = 640

GAME_MODES = {
    "classic":  {"time":60,"green_min":1.8,"green_max":5.0,"red_min":2.0,
                 "red_max":4.5,"torso_easy":0.035,"torso_hard":0.012,
                 "grace":0.60,"label":"CLASSIC"},
    "hardcore": {"time":45,"green_min":1.2,"green_max":3.0,"red_min":2.5,
                 "red_max":5.5,"torso_easy":0.025,"torso_hard":0.008,
                 "grace":0.40,"label":"HARDCORE"},
    "training": {"time":90,"green_min":3.0,"green_max":6.0,"red_min":1.5,
                 "red_max":3.0,"torso_easy":0.050,"torso_hard":0.020,
                 "grace":0.80,"label":"TRAINING"},
}

GAME_TIME=60; GREEN_MIN=1.8; GREEN_MAX=5.0; RED_MIN=2.0; RED_MAX=4.5
TORSO_THRESH_EASY=0.035; TORSO_THRESH_HARD=0.012; GRACE_PERIOD=0.60
CURRENT_MODE="classic"

RESTART_DELAY=5.0; HEADER_HEIGHT=90
GREEN_VARIANCE=0.5; RED_VARIANCE=0.6; FREEZE_WARNING_PRE=0.8

DISAPPEAR_TIMEOUT      = 2.5
ELIM_DISAPPEAR_TIMEOUT = 8.0
ASSIGN_DIST_THRESH     = 140   
SMOOTH_ALPHA           = 0.38
BBOX_PAD               = 14
MIN_ASSIGN_DIST        = 45
TORSO_SCALE_MATCH      = 1.4  
IOU_MATCH_THRESH       = 0.10
MERGE_DIST_FACTOR      = 0.6
JUMP_FILTER_THRESH     = 0.12
ELIM_OVERLAY_ALPHA     = 0.40
NMS_IOU_THRESH         = 0.45
SPAWN_GUARD_RATIO      = 0.55
SPAWN_GUARD_MIN        = 50
NEW_TRACKER_GRACE_SEC  = 1.0
MOVE_HISTORY_SIZE      = 6
MOVE_CONFIRM_COUNT     = 3
MERGE_IOU_THRESH       = 0.30
ELIM_ABSORB_FACTOR     = 1.6
PINCH_ON_THRESH   = 0.18
PINCH_OFF_THRESH  = 0.26
CLICK_COOLDOWN    = 0.50
POINTER_SMOOTH    = 0.40
ANGLE_SMOOTH      = 0.30
POINTER_SIZE      = 24
TRAIL_MAX         = 18
TRAIL_DURATION    = 0.30
RIPPLE_DURATION   = 0.50
COLORBLIND_MODE = args.colorblind
FRAME_SKIP      = max(1, args.frameskip)
SOUND_ENABLED   = _HAS_AUDIO and not args.no_sound


def apply_mode(m):
    global GAME_TIME,GREEN_MIN,GREEN_MAX,RED_MIN,RED_MAX
    global TORSO_THRESH_EASY,TORSO_THRESH_HARD,GRACE_PERIOD,CURRENT_MODE
    c = GAME_MODES.get(m, GAME_MODES["classic"])
    GAME_TIME=c["time"]; GREEN_MIN=c["green_min"]; GREEN_MAX=c["green_max"]
    RED_MIN=c["red_min"]; RED_MAX=c["red_max"]
    TORSO_THRESH_EASY=c["torso_easy"]; TORSO_THRESH_HARD=c["torso_hard"]
    GRACE_PERIOD=c["grace"]; CURRENT_MODE=m

apply_mode(args.mode)

if DETECTOR=="yolo" and (not _HAS_YOLO or not os.path.exists(MODEL_PATH)):
    if _HAS_MP: print("[WARN] YOLO unavailable, using MediaPipe."); DETECTOR="mp"
    else: raise SystemExit("[ERROR] No detector.")
if DETECTOR=="mp" and not _HAS_MP:
    raise SystemExit("[ERROR] MediaPipe unavailable.")

SOUNDS = {
    "rules":"../sounds/rules.wav","background":"../sounds/background.wav",
    "red":"../sounds/red.wav","green":"../sounds/green.wav",
    "gunshot":"../sounds/gunshot.wav","eliminated":"../sounds/eliminated.wav",
    "gameover":"../sounds/game_over.wav",
}
for k,p in SOUNDS.items():
    if not os.path.exists(p): print(f"[WARN] Missing: {p}")
def col_green():  return (180,180,0) if COLORBLIND_MODE else (30,200,30)
def col_red():    return (255,100,0) if COLORBLIND_MODE else (0,0,220)
def col_warn():   return (0,165,255)
def col_box_g():  return (180,180,0) if COLORBLIND_MODE else (50,220,50)
def col_box_r():  return (255,100,0) if COLORBLIND_MODE else (50,50,220)
SHAPE_TYPES=("triangle","circle","square")
SHAPES_TOTAL=12; _shapes_line=None

def _pick_shape(prev):
    return random.choice([s for s in SHAPE_TYPES if s!=prev])

def _init_shapes(w, tx, tw, th):
    global _shapes_line; _shapes_line=[]
    gl=max(0,tx-22); gr=min(w,tx+tw+22)
    lw=max(0,gl); rw=max(0,w-gr); tot=lw+rw
    if tot<=50: return
    sz=max(10,int(th*0.95)); cy=HEADER_HEIGHT//2
    step=tot/SHAPES_TOTAL; prev=None; cur=step/2
    for _ in range(SHAPES_TOTAL):
        cx=int(cur) if cur<=lw else int(gr+(cur-lw))
        st=_pick_shape(prev); prev=st
        _shapes_line.append({"type":st,"cx":cx,"cy":cy,"size":sz,
            "dur":random.uniform(1.4,2.4),"phase":random.uniform(0,6.28),
            "offset":random.uniform(0,6.28),
            "base_alpha":random.uniform(0.10,0.38)})
        cur+=step

def _draw_sh(img,st,cx,cy,sz,col=(0,0,0)):
    h2=sz//2
    if st=="circle": cv2.circle(img,(cx,cy),max(4,h2),col,-1,cv2.LINE_AA)
    elif st=="square": cv2.rectangle(img,(cx-h2,cy-h2),(cx+h2,cy+h2),col,-1,cv2.LINE_AA)
    else:
        pts=np.array([[cx,cy-h2],[cx-h2,cy+h2],[cx+h2,cy+h2]],np.int32)
        cv2.fillPoly(img,[pts],col,cv2.LINE_AA)

def _render_shapes(img,tx,tw):
    if not _shapes_line: return
    hdr=img[0:HEADER_HEIGHT,:,:].copy()
    gl=max(0,tx-22); gr=min(img.shape[1],tx+tw+22); now=time.time()
    for s in _shapes_line:
        if gl<=s["cx"]<=gr: continue
        pv=math.sin((now+s["offset"])*(6.28/s["dur"])+s["phase"])
        a=float(np.clip(s["base_alpha"]*(pv+1)/2,0,0.85))
        ly=hdr.copy(); _draw_sh(ly,s["type"],s["cx"],s["cy"],s["size"])
        hdr=cv2.addWeighted(ly,a,hdr,1-a,0)
    img[0:HEADER_HEIGHT,:,:]=hdr

def draw_header(img):
    h,w=img.shape[:2]
    cv2.rectangle(img,(0,0),(w,HEADER_HEIGHT),(203,105,255),-1)
    txt="SQUID GAME"; sc=1.9; th=4
    (tw,tth),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,sc,th)
    tx=w//2-tw//2
    global _shapes_line
    if not _shapes_line: _init_shapes(w,tx,tw,tth)
    _render_shapes(img,tx,tw)
    cv2.putText(img,txt,(tx,58),cv2.FONT_HERSHEY_SIMPLEX,sc,(0,0,0),th,cv2.LINE_AA)
    ml=GAME_MODES.get(CURRENT_MODE,{}).get("label","CLASSIC")
    cv2.putText(img,ml,(10,HEADER_HEIGHT-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
    if COLORBLIND_MODE:
        cv2.putText(img,"[CB]",(w-50,HEADER_HEIGHT-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),1,cv2.LINE_AA)
class Audio:
    def __init__(self):
        self.bg=None; self._lock=threading.Lock()
    def play_once(self,path):
        if not SOUND_ENABLED: return
        try:
            if os.path.exists(path): sa.WaveObject.from_wave_file(path).play()
        except: pass
    def play_blocking(self,path):
        if not SOUND_ENABLED: return
        try:
            if os.path.exists(path): sa.WaveObject.from_wave_file(path).play().wait_done()
        except: pass
    def start_bg(self):
        if not SOUND_ENABLED: return
        with self._lock:
            if not self.bg:
                try:
                    if os.path.exists(SOUNDS["background"]):
                        self.bg=sa.WaveObject.from_wave_file(SOUNDS["background"]).play()
                except: self.bg=None
    def stop_bg(self):
        with self._lock:
            try:
                if self.bg: self.bg.stop()
            except: pass
            self.bg=None

audio=Audio()
def make_cap():
    if sys.platform.startswith("darwin"):
        return cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(0)

cap=make_cap()
if not cap.isOpened(): raise SystemExit("[ERROR] Camera not accessible")

model=None
if DETECTOR=="yolo":
    try:
        print("[INFO] Loading YOLO..."); model=YOLO(MODEL_PATH); print("[INFO] YOLO ready.")
    except:
        model=None
        if _HAS_MP: DETECTOR="mp"; print("[WARN] Fallback to MediaPipe.")
        else: raise

pose=None; mp_pose=None
if DETECTOR=="mp":
    mp_pose=mp.solutions.pose
    pose=mp_pose.Pose(static_image_mode=False,model_complexity=1,
                      min_detection_confidence=0.6,min_tracking_confidence=0.6)
hands_detector=None
if _HAS_MP:
    mp_hands=mp.solutions.hands
    hands_detector=mp_hands.Hands(static_image_mode=False,max_num_hands=1,
        min_detection_confidence=0.7,min_tracking_confidence=0.6)
    print("[INFO] Pinch-pointer hand control enabled.")
else:
    print("[WARN] MediaPipe unavailable — keyboard only.")

# warm camera
print("[INFO] Warming camera...")
for _ in range(6):
    ok,f=cap.read()
    if not ok: break
    cv2.imshow("Squid Game",cv2.flip(f,1))
    if cv2.waitKey(1)&0xFF==ord('q'):
        cap.release(); cv2.destroyAllWindows(); raise SystemExit(0)
print("[INFO] Camera ready.")
class PointerController:
    """Pinch-pointer: index finger controls cursor, thumb+index pinch = click."""

    def __init__(self):
        self.cx=0.0; self.cy=0.0
        self.angle=0.0; self.dir_x=0.0; self.dir_y=-1.0
        self.detected=False
        self.pinching=False; self.just_clicked=False; self._was_pinch=False
        self.pinch_dist=1.0
        self._cooldown=0.0
        self.trail=[]; self.ripples=[]
        self.hover_btn=None
        self.thumb_x=0; self.thumb_y=0
        self.index_x=0; self.index_y=0

    def reset(self):
        self.detected=False; self.pinching=False
        self.just_clicked=False; self._was_pinch=False
        self._cooldown=0.0; self.trail.clear(); self.ripples.clear()
        self.hover_btn=None

    def update(self, frame):
        """Process frame with MediaPipe Hands. Returns True if hand found."""
        self.just_clicked=False
        if hands_detector is None:
            self.detected=False; return False

        h,w=frame.shape[:2]
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=hands_detector.process(rgb)
        if not res.multi_hand_landmarks:
            self.detected=False; return False

        lm=res.multi_hand_landmarks[0].landmark
        # Index tip
        tip=lm[8]; raw_x=tip.x*w; raw_y=tip.y*h
        if self.detected:
            self.cx=POINTER_SMOOTH*raw_x+(1-POINTER_SMOOTH)*self.cx
            self.cy=POINTER_SMOOTH*raw_y+(1-POINTER_SMOOTH)*self.cy
        else:
            self.cx=raw_x; self.cy=raw_y
        self.detected=True
        mcp=lm[5]
        ndx=tip.x-mcp.x; ndy=tip.y-mcp.y
        mag=math.hypot(ndx,ndy)
        if mag>0.001:
            ndx/=mag; ndy/=mag
            self.dir_x=ANGLE_SMOOTH*ndx+(1-ANGLE_SMOOTH)*self.dir_x
            self.dir_y=ANGLE_SMOOTH*ndy+(1-ANGLE_SMOOTH)*self.dir_y
            dm=math.hypot(self.dir_x,self.dir_y)
            if dm>0.001: self.dir_x/=dm; self.dir_y/=dm
        self.angle=math.atan2(self.dir_y,self.dir_x)
        self.thumb_x=int(lm[4].x*w); self.thumb_y=int(lm[4].y*h)
        self.index_x=int(lm[8].x*w); self.index_y=int(lm[8].y*h)
        tx=lm[4].x*w; ty=lm[4].y*h
        ix=tip.x*w; iy=tip.y*h
        dist=math.hypot(tx-ix,ty-iy)
        wrist=lm[0]; mid_mcp=lm[9]
        hand_sz=math.hypot((wrist.x-mid_mcp.x)*w,(wrist.y-mid_mcp.y)*h)
        self.pinch_dist=dist/max(1.0,hand_sz)
        if self.pinching:
            if self.pinch_dist>PINCH_OFF_THRESH: self.pinching=False
        else:
            if self.pinch_dist<PINCH_ON_THRESH: self.pinching=True
        now=time.time()
        if self.pinching and not self._was_pinch:
            if now>self._cooldown:
                self.just_clicked=True
                self._cooldown=now+CLICK_COOLDOWN
                self.ripples.append((int(self.cx),int(self.cy),now))
        self._was_pinch=self.pinching
        self.trail.append((int(self.cx),int(self.cy),now))
        if len(self.trail)>TRAIL_MAX*2:
            self.trail=self.trail[-TRAIL_MAX:]
        return True

    def check_buttons(self, buttons):
        """Returns button value on click, else None."""
        self.hover_btn=None
        if not self.detected:
            for b in buttons: b.is_hovered=False
            return None
        icx=int(self.cx); icy=int(self.cy)
        for b in buttons:
            if b.contains(icx,icy):
                b.is_hovered=True; self.hover_btn=b
                if self.just_clicked:
                    b.flash_time=time.time()
                    return b.value
            else:
                b.is_hovered=False
        return None

    def draw(self, img):
        """Draw pointer arrow, trail, pinch indicator, ripples."""
        if not self.detected: return
        h,w=img.shape[:2]; now=time.time()
        alive=[(x,y,t) for x,y,t in self.trail if now-t<TRAIL_DURATION]
        self.trail[:]=alive
        for x,y,t in alive:
            age=now-t; a=1.0-age/TRAIL_DURATION
            r=max(1,int(3*a))
            cv2.circle(img,(x,y),r,(int(200*a),int(255*a),0),-1,cv2.LINE_AA)
        alive_r=[]
        for rx,ry,rt in self.ripples:
            age=now-rt
            if age>RIPPLE_DURATION: continue
            alive_r.append((rx,ry,rt))
            rad=int(15+age*120); a=max(0,1.0-age/RIPPLE_DURATION)
            tk=max(1,int(3*a))
            cv2.circle(img,(rx,ry),rad,(0,int(255*a),0),tk,cv2.LINE_AA)
        self.ripples[:]=alive_r
        cx=int(self.cx); cy=int(self.cy)
        if self.pinching:      pcol=(0,255,0)
        elif self.hover_btn:   pcol=(0,255,255)
        else:                  pcol=(0,220,255)
        sz=POINTER_SIZE
        bx=cx-self.dir_x*sz; by=cy-self.dir_y*sz
        wing=sz*0.38; px=-self.dir_y; py=self.dir_x
        left=(int(bx+px*wing),int(by+py*wing))
        right=(int(bx-px*wing),int(by-py*wing))
        pts=np.array([(cx,cy),left,right],np.int32)
        cv2.fillPoly(img,[pts+2],(0,0,0),cv2.LINE_AA)
        cv2.fillPoly(img,[pts],pcol,cv2.LINE_AA)
        cv2.polylines(img,[pts],True,(255,255,255),1,cv2.LINE_AA)
        cv2.circle(img,(cx,cy),3,(255,255,255),-1,cv2.LINE_AA)
        pd=self.pinch_dist
        line_col=(0,255,0) if self.pinching else (
            (0,int(min(255,255*(1-pd/0.3))),255) if pd<0.3 else (100,100,100))
        cv2.line(img,(self.thumb_x,self.thumb_y),(self.index_x,self.index_y),
                 line_col,2,cv2.LINE_AA)
        dot_col=(0,255,0) if self.pinching else (0,180,255)
        cv2.circle(img,(self.thumb_x,self.thumb_y),6,dot_col,-1,cv2.LINE_AA)
        cv2.circle(img,(self.index_x,self.index_y),6,dot_col,-1,cv2.LINE_AA)
        if not self.pinching and pd<0.35:
            prog=max(0,1.0-pd/PINCH_ON_THRESH)
            if prog>0:
                angle=int(360*min(1.0,prog))
                arc_r=POINTER_SIZE+10
                arc_col=(0,int(255*prog),255)
                cv2.ellipse(img,(cx,cy),(arc_r,arc_r),-90,0,angle,arc_col,3,cv2.LINE_AA)

        status="PINCH!" if self.pinching else "Point & Pinch"
        scol=(0,255,0) if self.pinching else (180,180,180)
        cv2.putText(img,status,(w-160,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,scol,1,cv2.LINE_AA)
        hand_lbl="Hand: ACTIVE" if self.detected else ""
        cv2.putText(img,hand_lbl,(w-160,h-35),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,200,0),1,cv2.LINE_AA)


pointer = PointerController()

class HoverButton:
    def __init__(self,x,y,w,h,label,value=None):
        self.x=x; self.y=y; self.w=w; self.h=h
        self.label=label; self.value=value
        self.is_hovered=False; self.flash_time=0.0

    def contains(self,cx,cy):
        return self.x<=cx<=self.x+self.w and self.y<=cy<=self.y+self.h

    def reset(self):
        self.is_hovered=False

    def draw(self,img,is_selected=False):
        now=time.time()
        flashing=(now-self.flash_time)<0.4 if self.flash_time else False
        if flashing:        bg=(0,220,0)
        elif is_selected:   bg=(0,130,130)
        elif self.is_hovered: bg=(70,70,70)
        else:               bg=(40,40,40)
        cv2.rectangle(img,(self.x,self.y),(self.x+self.w,self.y+self.h),bg,-1)
        bc=(0,255,255) if is_selected else ((255,255,255) if self.is_hovered else (80,80,80))
        tk=2 if (is_selected or self.is_hovered) else 1
        cv2.rectangle(img,(self.x,self.y),(self.x+self.w,self.y+self.h),bc,tk,cv2.LINE_AA)
        (tw2,th2),_=cv2.getTextSize(self.label,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
        tx=self.x+(self.w-tw2)//2; ty=self.y+(self.h+th2)//2
        tc=(0,0,0) if flashing else (255,255,255)
        cv2.putText(img,self.label,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,0.6,tc,2,cv2.LINE_AA)
        # Hover glow
        if self.is_hovered and not flashing:
            ov=img[self.y:self.y+self.h,self.x:self.x+self.w].copy()
            gl=np.full_like(ov,(0,60,60),np.uint8)
            cv2.addWeighted(gl,0.25,ov,0.75,0,ov)
            img[self.y:self.y+self.h,self.x:self.x+self.w]=ov

menu_buttons=[]; _btn_init=False

def init_buttons(w,h):
    global menu_buttons,_btn_init
    bw=min(340,int(w*0.52)); bh=50
    bx=w//2-bw//2; sy=h//2-70; sp=60
    menu_buttons=[
        HoverButton(bx,sy,bw,bh,
                    f"CLASSIC  ({GAME_MODES['classic']['time']}s)","classic"),
        HoverButton(bx,sy+sp,bw,bh,
                    f"HARDCORE ({GAME_MODES['hardcore']['time']}s)","hardcore"),
        HoverButton(bx,sy+2*sp,bw,bh,
                    f"TRAINING ({GAME_MODES['training']['time']}s)","training"),
        HoverButton(bx,sy+3*sp+20,bw,bh+8,
                    ">> START GAME <<","start"),
    ]
    _btn_init=True

def draw_alert_bar(img,progress):
    h,w=img.shape[:2]; bh=16; m=12
    x1,x2=m,w-m; fl=int((x2-x1)*(1-progress))
    col=col_green() if progress<0.6 else (col_warn() if progress<0.9 else col_red())
    cv2.rectangle(img,(x1,HEADER_HEIGHT-bh-8),(x2,HEADER_HEIGHT-8),(50,50,50),-1)
    cv2.rectangle(img,(x1,HEADER_HEIGHT-bh-8),(x1+fl,HEADER_HEIGHT-8),col,-1)
    cv2.rectangle(img,(x1,HEADER_HEIGHT-bh-8),(x2,HEADER_HEIGHT-8),(0,0,0),1)

def draw_freeze_border(img,intensity):
    h,w=img.shape[:2]; t=int(12*intensity); a=0.5*intensity
    ov=img.copy()
    cv2.rectangle(ov,(0,0),(w,h),(255,100,0) if COLORBLIND_MODE else (0,0,255),t)
    cv2.addWeighted(ov,a,img,1-a,0,img)

def draw_grace_overlay(img,gr):
    if gr<=0: return
    h,w=img.shape[:2]; a=min(0.45,gr/GRACE_PERIOD*0.45)
    ov=img.copy()
    cv2.rectangle(ov,(0,HEADER_HEIGHT),(w,h),(0,120,255),-1)
    cv2.addWeighted(ov,a,img,1-a,0,img)
    lb=f"FREEZE! {gr:.2f}s"
    (lw2,lh2),_=cv2.getTextSize(lb,cv2.FONT_HERSHEY_DUPLEX,1.2,3)
    cv2.putText(img,lb,(w//2-lw2//2,h//2),cv2.FONT_HERSHEY_DUPLEX,1.2,(255,255,255),3,cv2.LINE_AA)

def draw_fps(img,fps):
    cv2.putText(img,f"FPS:{fps:.0f}",(10,img.shape[0]-40),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,150,150),1,cv2.LINE_AA)

def cur_green(ge):
    t=min(1,ge/GAME_TIME); b=GREEN_MAX-t*(GREEN_MAX-GREEN_MIN)
    return max(GREEN_MIN*0.5,b+random.uniform(-GREEN_VARIANCE,GREEN_VARIANCE))
def cur_red(ge):
    t=min(1,ge/GAME_TIME); b=RED_MIN+t*(RED_MAX-RED_MIN)
    return max(RED_MIN*0.5,b+random.uniform(-RED_VARIANCE,RED_VARIANCE))
def cur_thresh(ge):
    t=min(1,ge/GAME_TIME)
    return TORSO_THRESH_EASY-t*(TORSO_THRESH_EASY-TORSO_THRESH_HARD)

def torso_cen(lm,w,h):
    if DETECTOR!="mp" or mp_pose is None: return None,None
    ids=[mp_pose.PoseLandmark.LEFT_SHOULDER,mp_pose.PoseLandmark.RIGHT_SHOULDER,
         mp_pose.PoseLandmark.LEFT_HIP,mp_pose.PoseLandmark.RIGHT_HIP]
    try: pts=[(lm[i.value].x*w,lm[i.value].y*h) for i in ids]
    except: return None,None
    cx=sum(p[0] for p in pts)/4; cy=sum(p[1] for p in pts)/4
    return (cx,cy),max(1,abs(pts[0][1]-pts[2][1]))

def bbox_lm(lm,w,h):
    xs=[int(p.x*w) for p in lm]; ys=[int(p.y*h) for p in lm]
    return (max(0,min(xs)-BBOX_PAD),max(HEADER_HEIGHT,min(ys)-BBOX_PAD),
            min(w,max(xs)+BBOX_PAD),min(h,max(ys)+BBOX_PAD))

trackers={}; next_id=1; player_scores={}

def make_id(i): return f"P{i:03d}"

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[2],b[2]); y2=min(a[3],b[3])
    w2=max(0,x2-x1); h2=max(0,y2-y1); inter=w2*h2
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    ab=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    u=aa+ab-inter
    return inter/u if u>0 else 0

def nms_detections(dets, thresh=NMS_IOU_THRESH):
    """Non-maximum suppression: remove overlapping detections of same person."""
    if not dets: return []
    sd=sorted(dets,key=lambda d:-(d[2][2]-d[2][0])*(d[2][3]-d[2][1]))
    keep=[]
    for d in sd:
        if not any(iou(d[2],k[2])>thresh for k in keep):
            keep.append(d)
    return keep

def assign_all(dets):
    """Global greedy assignment with anti-duplicate protections."""
    global next_id
    now=time.time()
    if not dets: return []

    cands=[]
    for di,(cx,cy,bb,th) in enumerate(dets):
        for tid,tr in trackers.items():
            is_elim=tr.get("eliminated",False)
            dt=min(max(MIN_ASSIGN_DIST,th*TORSO_SCALE_MATCH),ASSIGN_DIST_THRESH)
            if is_elim: dt*=ELIM_ABSORB_FACTOR  # wider radius for eliminated
            d=math.hypot(cx-tr["smooth_cx"],cy-tr["smooth_cy"])
            pr=1 if is_elim else 0
            if d<dt: cands.append((pr,d,di,tid))
    cands.sort()

    ad=set(); at=set(); matched=[]
    for _,_,di,tid in cands:
        if di in ad or tid in at: continue
        ad.add(di); at.add(tid); matched.append((di,tid))

    for di in [i for i in range(len(dets)) if i not in ad]:
        _,_,bb,_=dets[di]
        best_tid=None; best_iou=IOU_MATCH_THRESH
        for tid,tr in trackers.items():
            if tid in at: continue
            s=iou(bb,tr.get("bbox",(0,0,0,0)))
            if s>best_iou:
                best_iou=s; best_tid=tid
        if best_tid:
            ad.add(di); at.add(best_tid); matched.append((di,best_tid))

    res=[]
    for di,tid in matched:
        cx,cy,bb,th=dets[di]; tr=trackers[tid]
        pcx,pcy=tr["smooth_cx"],tr["smooth_cy"]
        tr["smooth_cx"]=SMOOTH_ALPHA*cx+(1-SMOOTH_ALPHA)*pcx
        tr["smooth_cy"]=SMOOTH_ALPHA*cy+(1-SMOOTH_ALPHA)*pcy
        tr["raw_cx"]=cx; tr["raw_cy"]=cy; tr["bbox"]=bb; tr["torso_h"]=th
        tr["last_seen"]=now; tr["age_sec"]=now-tr["born"]
        if not tr.get("eliminated"):
            j=math.hypot(tr["smooth_cx"]-pcx,tr["smooth_cy"]-pcy)
            if (j/max(1,th))>JUMP_FILTER_THRESH:
                tr["prev_cx"]=tr["smooth_cx"]; tr["prev_cy"]=tr["smooth_cy"]
            else:
                tr["prev_cx"]=pcx; tr["prev_cy"]=pcy
        res.append(tid)

    for di in range(len(dets)):
        if di in ad: continue
        cx,cy,bb,th=dets[di]

        best_tid=None; best_d=max(SPAWN_GUARD_MIN, th*SPAWN_GUARD_RATIO)
        for tid,tr in trackers.items():
            if tid in at: continue
            d=math.hypot(cx-tr["smooth_cx"],cy-tr["smooth_cy"])
            if d<best_d:
                best_d=d; best_tid=tid
        if best_tid:
            tr=trackers[best_tid]
            pcx,pcy=tr["smooth_cx"],tr["smooth_cy"]
            tr["smooth_cx"]=SMOOTH_ALPHA*cx+(1-SMOOTH_ALPHA)*pcx
            tr["smooth_cy"]=SMOOTH_ALPHA*cy+(1-SMOOTH_ALPHA)*pcy
            tr["raw_cx"]=cx; tr["raw_cy"]=cy; tr["bbox"]=bb; tr["torso_h"]=th
            tr["last_seen"]=now; tr["age_sec"]=now-tr["born"]
            if not tr.get("eliminated"):
                tr["prev_cx"]=pcx; tr["prev_cy"]=pcy
            ad.add(di); at.add(best_tid); res.append(best_tid)
            continue

        near_elim=False
        for tid,tr in trackers.items():
            if not tr.get("eliminated"): continue
            d=math.hypot(cx-tr["smooth_cx"],cy-tr["smooth_cy"])
            if d < max(SPAWN_GUARD_MIN*1.5, th*SPAWN_GUARD_RATIO*2):
                tr["smooth_cx"]=SMOOTH_ALPHA*cx+(1-SMOOTH_ALPHA)*tr["smooth_cx"]
                tr["smooth_cy"]=SMOOTH_ALPHA*cy+(1-SMOOTH_ALPHA)*tr["smooth_cy"]
                tr["bbox"]=bb; tr["last_seen"]=now
                near_elim=True; res.append(tid); break
        if near_elim: continue

        pid=make_id(next_id); next_id+=1
        trackers[pid]={"smooth_cx":cx,"smooth_cy":cy,"raw_cx":cx,"raw_cy":cy,
            "prev_cx":cx,"prev_cy":cy,"bbox":bb,"torso_h":th,
            "last_seen":now,"born":now,"age_sec":0.0,
            "eliminated":False,"move_history":[]}
        player_scores[pid]=0.0; res.append(pid)
    return res

def merge_trackers():
    """Merge overlapping living trackers by distance AND IoU."""
    ids=[k for k,v in trackers.items() if not v.get("eliminated")]
    rm=set()
    for i in range(len(ids)):
        if ids[i] in rm: continue
        for j in range(i+1,len(ids)):
            if ids[j] in rm: continue
            a,b=ids[i],ids[j]
            ta,tb=trackers[a],trackers[b]
            mt=(ta["torso_h"]+tb["torso_h"])/2
            d=math.hypot(ta["smooth_cx"]-tb["smooth_cx"],ta["smooth_cy"]-tb["smooth_cy"])
            bi=iou(ta["bbox"],tb["bbox"])
            if d<mt*MERGE_DIST_FACTOR or bi>MERGE_IOU_THRESH:
                keep,drop=(a,b) if ta["born"]<tb["born"] else (b,a)
                rm.add(drop)
    for t in rm:
        player_scores.pop(t,None); del trackers[t]

def cull():
    now=time.time(); dl=[]
    for k,v in trackers.items():
        to=ELIM_DISAPPEAR_TIMEOUT if v.get("eliminated") else DISAPPEAR_TIMEOUT
        if now-v["last_seen"]>to: dl.append(k)
    for k in dl:
        player_scores.pop(k,None); del trackers[k]

def detect_persons(frame):
    """Detect persons with NMS applied."""
    h,w=frame.shape[:2]; dets=[]
    if DETECTOR=="yolo" and model:
        for r in model.predict(frame,classes=[0],conf=CONF,imgsz=IMG_SIZE,verbose=False):
            for box in r.boxes:
                x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
                y1=max(y1,HEADER_HEIGHT)
                if x2<=x1 or y2<=y1: continue
                dets.append(((x1+x2)//2,(y1+y2)//2,(x1,y1,x2,y2),max(1,(y2-y1)*0.5)))
    elif DETECTOR=="mp" and pose:
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=pose.process(rgb)
        if res.pose_landmarks:
            lm=res.pose_landmarks.landmark
            cen,th=torso_cen(lm,w,h)
            if cen:
                bb=bbox_lm(lm,w,h)
                dets.append((int(cen[0]),int(cen[1]),bb,th))
    return nms_detections(dets)

# ======================== MOVEMENT CHECK (BUFFERED) ========================
def check_movement(tid, ge):
    """Multi-frame movement buffer. Returns True only if sustained movement."""
    tr=trackers.get(tid)
    if not tr or tr.get("eliminated"): return False
    # Provisional immunity
    if tr["age_sec"]<NEW_TRACKER_GRACE_SEC: return False

    th=max(1,tr["torso_h"])
    d=math.hypot(tr["smooth_cx"]-tr["prev_cx"],tr["smooth_cy"]-tr["prev_cy"])
    moved=(d/th)>cur_thresh(ge)

    # Update history buffer
    hist=tr.get("move_history",[])
    hist.append(moved)
    if len(hist)>MOVE_HISTORY_SIZE: hist=hist[-MOVE_HISTORY_SIZE:]
    tr["move_history"]=hist

    # Need sustained movement
    return sum(hist)>=MOVE_CONFIRM_COUNT

def update_scores(tids,ge):
    for tid in tids:
        tr=trackers.get(tid)
        if tr and not tr.get("eliminated"):
            player_scores.setdefault(tid,0.0)
            if state==STATE_RED: player_scores[tid]+=0.5
            elif state==STATE_GREEN: player_scores[tid]+=0.1

# ======================== GAME STATE ========================
STATE_IDLE="idle"; STATE_GREEN="green"; STATE_GRACE="grace"
STATE_RED="red"; STATE_OVER="over"

state=STATE_IDLE; state_start=0.0; game_start=0.0
green_duration=0.0; red_duration=0.0
eliminated_set=set(); survivors=0

def transition(ns):
    global state,state_start; state=ns; state_start=time.time()

def start_game():
    global eliminated_set,survivors,game_start,next_id
    global green_duration,red_duration,player_scores
    trackers.clear(); eliminated_set.clear(); player_scores.clear()
    survivors=0; next_id=1; green_duration=0; red_duration=0
    game_start=time.time()
    audio.stop_bg(); audio.start_bg()
    transition(STATE_GREEN); audio.play_once(SOUNDS["green"])

# ======================== STATUS OVERLAY ========================
def draw_status(img,ge,gr=0,wa=0):
    h,w=img.shape[:2]; tl=max(0,GAME_TIME-ge)
    if wa>0: draw_freeze_border(img,wa)
    if state==STATE_GRACE and gr>0: draw_grace_overlay(img,gr)
    if state==STATE_GREEN:    bc=col_green(); bt="GREEN LIGHT - MOVE!"
    elif state==STATE_GRACE:  bc=col_warn();  bt="FREEZE NOW!"
    elif state==STATE_RED:    bc=col_red();   bt="RED LIGHT - FREEZE!"
    elif state==STATE_OVER:   bc=(80,80,80);  bt="GAME OVER"
    else:                     bc=(100,100,100);bt="READY"
    (bw2,bh2),_=cv2.getTextSize(bt,cv2.FONT_HERSHEY_DUPLEX,0.75,2)
    bx=w-bw2-20; by=HEADER_HEIGHT+36
    cv2.rectangle(img,(bx-8,by-bh2-6),(bx+bw2+8,by+8),bc,-1)
    cv2.putText(img,bt,(bx,by),cv2.FONT_HERSHEY_DUPLEX,0.75,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(img,f"{tl:.1f}s",(20,HEADER_HEIGHT+42),cv2.FONT_HERSHEY_DUPLEX,1.3,(255,255,255),3,cv2.LINE_AA)
    cv2.putText(img,f"{tl:.1f}s",(20,HEADER_HEIGHT+42),cv2.FONT_HERSHEY_DUPLEX,1.3,(30,30,30),2,cv2.LINE_AA)
    thr=cur_thresh(ge)
    tp=int(100*(thr-TORSO_THRESH_HARD)/max(0.001,TORSO_THRESH_EASY-TORSO_THRESH_HARD))
    sn="Easy" if tp>66 else ("Medium" if tp>33 else "Hard")
    cv2.putText(img,f"Sensitivity: {sn}",(20,h-16),cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),1,cv2.LINE_AA)
    al=len([k for k,v in trackers.items() if not v.get("eliminated")])
    cv2.putText(img,f"Alive: {al}  |  Out: {len(eliminated_set)}",(20,HEADER_HEIGHT+72),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)
    if state in (STATE_RED,STATE_GRACE):
        er=time.time()-state_start
        if state==STATE_GRACE: er=max(0,er-GRACE_PERIOD)
        p=min(1,er/red_duration) if red_duration>0 else 0
        draw_alert_bar(img,p)

# ======================== TRACKER BOXES ========================
def draw_boxes(img):
    h,w=img.shape[:2]
    for tid,tr in trackers.items():
        x1,y1,x2,y2=tr["bbox"]
        x1c=max(0,x1); y1c=max(0,y1); x2c=min(w,x2); y2c=min(h,y2)
        if tr.get("eliminated"):
            if x2c<=x1c or y2c<=y1c: continue
            roi=img[y1c:y2c,x1c:x2c]
            rc=(0,100,255) if COLORBLIND_MODE else (0,0,200)
            rl=np.full_like(roi,rc,np.uint8)
            cv2.addWeighted(rl,ELIM_OVERLAY_ALPHA,roi,1-ELIM_OVERLAY_ALPHA,0,roi)
            img[y1c:y2c,x1c:x2c]=roi
            cv2.rectangle(img,(x1c,y1c),(x2c,y2c),(0,0,220),3,cv2.LINE_AA)
            bw3=x2c-x1c; bh3=y2c-y1c
            fs=max(0.5,min(2.8,bw3/75,bh3/90)); tk=max(2,int(fs*2.5))
            (tw3,th3),_=cv2.getTextSize("OUT",cv2.FONT_HERSHEY_DUPLEX,fs,tk)
            tx3=x1c+(bw3-tw3)//2; ty3=y1c+(bh3+th3)//2
            cv2.putText(img,"OUT",(tx3+2,ty3+2),cv2.FONT_HERSHEY_DUPLEX,fs,(0,0,0),tk+2,cv2.LINE_AA)
            cv2.putText(img,"OUT",(tx3,ty3),cv2.FONT_HERSHEY_DUPLEX,fs,(255,255,255),tk,cv2.LINE_AA)
            sc=player_scores.get(tid,0)
            cv2.putText(img,f"{sc:.0f}pts",(x1c+4,y2c+18),cv2.FONT_HERSHEY_SIMPLEX,0.45,(150,150,150),1,cv2.LINE_AA)
            continue
        cl=col_box_g() if state==STATE_GREEN else col_box_r()
        cv2.rectangle(img,(x1,y1),(x2,y2),cl,2,cv2.LINE_AA)
        ly2=max(y1-6,HEADER_HEIGHT+14)
        cv2.putText(img,tid,(x1+4,ly2),cv2.FONT_HERSHEY_SIMPLEX,0.55,cl,2,cv2.LINE_AA)
        sc=player_scores.get(tid,0)
        if sc>0:
            cv2.putText(img,f"{sc:.0f}pts",(x1+4,min(y2+18,h-4)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(220,220,220),1,cv2.LINE_AA)

# ======================== SCREEN DRAWS ========================
def draw_idle(img):
    h,w=img.shape[:2]
    global _btn_init
    if not _btn_init: init_buttons(w,h)
    draw_header(img)

    # Instruction panel
    ix=14; iy=HEADER_HEIGHT+18
    lines=[
        ("PINCH-POINTER CONTROLS:",True),
        ("",False),
        ("  Move index finger = aim pointer",False),
        ("  Pinch thumb+index = SELECT",False),
        ("  Point at button & pinch to choose",False),
        ("",False),
        ("KEYBOARD: 1/2/3, SPACE, Q, C, F",False),
    ]
    for i,(ln,bold) in enumerate(lines):
        cl=(0,255,255) if bold else (170,170,170)
        sc=0.52 if bold else 0.44
        cv2.putText(img,ln,(ix,iy+i*20),cv2.FONT_HERSHEY_SIMPLEX,sc,cl,1,cv2.LINE_AA)

    # Buttons
    for btn in menu_buttons:
        btn.draw(img, is_selected=(btn.value==CURRENT_MODE))

    # Settings
    cfg=f"Mode: {CURRENT_MODE.upper()} | CB: {'ON' if COLORBLIND_MODE else 'OFF'} | Skip: {FRAME_SKIP}"
    cv2.putText(img,cfg,(10,h-55),cv2.FONT_HERSHEY_SIMPLEX,0.38,(100,100,100),1,cv2.LINE_AA)

    # Hand status
    hs="Hand: DETECTED" if pointer.detected else "Show hand to use pointer"
    hc=(0,200,0) if pointer.detected else (80,80,80)
    cv2.putText(img,hs,(w//2-120,h-55),cv2.FONT_HERSHEY_SIMPLEX,0.45,hc,1,cv2.LINE_AA)

def draw_gameover(img,pe):
    h,w=img.shape[:2]; draw_header(img)
    cv2.putText(img,"GAME OVER",(w//2-160,h//2-100),
                cv2.FONT_HERSHEY_DUPLEX,1.8,col_red(),4,cv2.LINE_AA)
    cv2.putText(img,f"Survivors: {survivors}",(w//2-120,h//2-50),
                cv2.FONT_HERSHEY_DUPLEX,1.2,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(img,f"Eliminated: {len(eliminated_set)}",(w//2-120,h//2-10),
                cv2.FONT_HERSHEY_DUPLEX,1.1,(0,100,255),2,cv2.LINE_AA)
    # Top scores
    ss=sorted(player_scores.items(),key=lambda x:-x[1])[:5]
    if ss:
        cv2.putText(img,"TOP SCORES:",(w//2-80,h//2+35),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),2,cv2.LINE_AA)
        for i,(tid,sc) in enumerate(ss):
            el=tid in eliminated_set
            st=" [OUT]" if el else " [ALIVE]"
            cl=(100,100,255) if el else (100,255,100)
            cv2.putText(img,f"{i+1}. {tid}: {sc:.0f}pts{st}",
                        (w//2-120,h//2+65+i*26),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,cl,1,cv2.LINE_AA)
    rm=max(0,RESTART_DELAY-pe)
    cv2.putText(img,f"Restarting in {rm:.1f}s... (pinch to restart)",
                (w//2-180,h-25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,150,150),1,cv2.LINE_AA)

# ======================== MAIN LOOP ========================
transition(STATE_IDLE); pointer.reset()
print("[INFO] Point & pinch to select. SPACE/Q for keyboard.")
print(f"[INFO] Mode: {CURRENT_MODE.upper()} | Detector: {DETECTOR.upper()}")

frame_counter=0; fps=0.0; fps_timer=time.time(); fps_count=0
last_dets=[]

while True:
    ok,frame=cap.read()
    if not ok: time.sleep(0.05); continue
    frame=cv2.flip(frame,1)
    h,w=frame.shape[:2]
    frame_counter+=1

    fps_count+=1
    if time.time()-fps_timer>=1.0:
        fps=fps_count/(time.time()-fps_timer)
        fps_count=0; fps_timer=time.time()

    ge=time.time()-game_start if state!=STATE_IDLE else 0.0
    gr=0.0; wa=0.0; pe=time.time()-state_start
    run_det=(frame_counter%FRAME_SKIP==0)

    # ==================== IDLE ====================
    if state==STATE_IDLE:
        pointer.update(frame)
        action=pointer.check_buttons(menu_buttons)
        if action:
            if action in GAME_MODES:
                apply_mode(action); _shapes_line=None; _btn_init=False
                print(f"[POINTER] Mode → {action.upper()}")
            elif action=="start":
                pointer.reset(); start_game(); continue
        elif not pointer.detected:
            for b in menu_buttons: b.is_hovered=False

        draw_idle(frame)
        pointer.draw(frame)

    # ==================== GREEN ====================
    elif state==STATE_GREEN:
        if run_det: dets=detect_persons(frame); last_dets=dets
        else: dets=last_dets
        assign_all(dets); merge_trackers(); cull()
        update_scores(list(trackers.keys()),ge)
        if green_duration==0: green_duration=cur_green(ge)
        tue=green_duration-pe
        if tue<=FREEZE_WARNING_PRE: wa=1-(tue/FREEZE_WARNING_PRE)
        if pe>=green_duration:
            green_duration=0; red_duration=cur_red(ge)
            transition(STATE_GRACE); audio.play_once(SOUNDS["red"])
        draw_boxes(frame); draw_header(frame); draw_status(frame,ge,wa=wa)

    # ==================== GRACE ====================
    elif state==STATE_GRACE:
        if run_det: dets=detect_persons(frame); last_dets=dets
        else: dets=last_dets
        assign_all(dets); merge_trackers(); cull()
        update_scores(list(trackers.keys()),ge)
        gr=max(0,GRACE_PERIOD-pe)
        if gr==0: transition(STATE_RED)
        draw_boxes(frame); draw_header(frame); draw_status(frame,ge,gr=gr)

    # ==================== RED ====================
    elif state==STATE_RED:
        if run_det: dets=detect_persons(frame); last_dets=dets
        else: dets=last_dets
        tids=assign_all(dets); merge_trackers(); cull()
        update_scores(tids,ge)

        # Movement check with all anti-duplicate protections
        for tid in tids:
            tr=trackers.get(tid)
            if tr is None or tr.get("eliminated"): continue
            if tr["age_sec"]<NEW_TRACKER_GRACE_SEC: continue  # provisional
            if check_movement(tid,ge) and tid not in eliminated_set:
                eliminated_set.add(tid); tr["eliminated"]=True
                tr["move_history"]=[]  # reset
                audio.play_once(SOUNDS["gunshot"])
                threading.Thread(target=audio.play_blocking,
                                 args=(SOUNDS["eliminated"],),daemon=True).start()

        if pe>=red_duration:
            green_duration=0; transition(STATE_GREEN)
            audio.play_once(SOUNDS["green"])
            # Reset movement histories on phase change
            for tr in trackers.values():
                if not tr.get("eliminated"):
                    tr["move_history"]=[]

        draw_boxes(frame); draw_header(frame); draw_status(frame,ge)

    if state not in (STATE_IDLE,STATE_OVER) and ge>=GAME_TIME:
        survivors=len([k for k,v in trackers.items() if not v.get("eliminated")])
        audio.stop_bg(); audio.play_once(SOUNDS["gameover"])
        transition(STATE_OVER); pointer.reset()

    if state==STATE_OVER:
        pointer.update(frame)
        if pointer.just_clicked:
            pointer.reset(); start_game(); continue
        draw_gameover(frame,pe)
        pointer.draw(frame)
        if pe>=RESTART_DELAY:
            transition(STATE_IDLE); pointer.reset(); _btn_init=False

    draw_fps(frame,fps)
    cv2.imshow("Squid Game",frame)

    key=cv2.waitKey(1)&0xFF
    if key==ord('q'): break
    elif key==ord(' ') and state in (STATE_IDLE,STATE_OVER):
        pointer.reset(); start_game()
    elif key==ord('c'):
        COLORBLIND_MODE=not COLORBLIND_MODE
        print(f"[INFO] Colorblind: {'ON' if COLORBLIND_MODE else 'OFF'}")
    elif key==ord('f'):
        FRAME_SKIP=1 if FRAME_SKIP>1 else 2
        print(f"[INFO] Frame skip: {FRAME_SKIP}")
    elif key==ord('1') and state==STATE_IDLE:
        apply_mode("classic"); _shapes_line=None; _btn_init=False
    elif key==ord('2') and state==STATE_IDLE:
        apply_mode("hardcore"); _shapes_line=None; _btn_init=False
    elif key==ord('3') and state==STATE_IDLE:
        apply_mode("training"); _shapes_line=None; _btn_init=False

print("[INFO] Shutting down...")
cap.release(); cv2.destroyAllWindows(); audio.stop_bg()
if pose: pose.close()
if hands_detector: hands_detector.close()
print("[INFO] Done.")
