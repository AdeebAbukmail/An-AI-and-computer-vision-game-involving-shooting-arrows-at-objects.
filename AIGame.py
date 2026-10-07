
import sys
import json
import time
import threading
import webbrowser
from collections import deque
from math import hypot
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8765
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
               (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
CAM_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# ───────────────────────────── الرؤية الحاسوبية ─────────────────────────────
class Vision(threading.Thread):
    def __init__(self, cam_index):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.cond = threading.Condition()
        self.jpg = None
        self.seq = 0
        self.state = {"g": "none", "n": 0, "cam": False}

    @staticmethod
    def _finger_up(lm, tip, pip):
        w = lm[0]
        d_tip = hypot(lm[tip].x - w.x, lm[tip].y - w.y)
        d_pip = hypot(lm[pip].x - w.x, lm[pip].y - w.y)
        return d_tip > d_pip * 1.05

    def _hand_gesture(self, lm):
        idx = self._finger_up(lm, 8, 6)
        mid = self._finger_up(lm, 12, 10)
        rng = self._finger_up(lm, 16, 14)
        pky = self._finger_up(lm, 20, 18)
        if idx and not (mid or rng or pky):
            return "point"
        if not (idx or mid or rng or pky):
            return "fist"
        return "none"

    def _classify(self, hands, asp):
        if len(hands) == 2:
            a, b = hands[0], hands[1]
            d = hypot((a[9].x - b[9].x) * asp, a[9].y - b[9].y)
            sa = hypot((a[0].x - a[9].x) * asp, a[0].y - a[9].y)
            sb = hypot((b[0].x - b[9].x) * asp, b[0].y - b[9].y)
            if d < (sa + sb) / 2 * 2.1:
                return "bomb"
        found = [self._hand_gesture(h) for h in hands]
        if "point" in found:
            return "point"
        if "fist" in found:
            return "fist"
        return "none"

    def _make_detector(self, mp):
        """يرجع دالة detect(rgb) -> قائمة من قوائم النقاط. يدعم mediapipe القديم والجديد."""
        if hasattr(mp, "solutions"):  # النسخ القديمة
            hands = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=0,
                                             min_detection_confidence=0.6,
                                             min_tracking_confidence=0.5)
            return lambda rgb: [h.landmark for h in (hands.process(rgb).multi_hand_landmarks or [])]

        # النسخ الجديدة (Tasks API) — بتحتاج ملف موديل بينزل مرة وحدة
        import os
        import urllib.request
        from mediapipe.tasks import python as mp_py
        from mediapipe.tasks.python import vision
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
        if not os.path.exists(path):
            print("[…] بنزّل موديل اليد (مرة وحدة، ~8MB)...")
            try:
                urllib.request.urlretrieve(MODEL_URL, path)
            except Exception as e:
                print("[!] فشل تنزيل الموديل:", e)
                print("    نزّله يدوياً من:", MODEL_URL)
                print("    وحطه جنب هذا الملف باسم hand_landmarker.task")
                return None
        opts = vision.HandLandmarkerOptions(
            base_options=mp_py.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.6, min_tracking_confidence=0.5)
        lmk = vision.HandLandmarker.create_from_options(opts)
        t0 = time.time()

        def detect(rgb):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = lmk.detect_for_video(img, int((time.time() - t0) * 1000))
            return list(res.hand_landmarks or [])
        return detect

    def run(self):
        try:
            import cv2
            import mediapipe as mp
        except ImportError:
            print("\n[!] لازم تثبت المكتبات:  pip install opencv-python mediapipe\n")
            return

        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print("\n[!] ما قدرت أفتح الكاميرا. جرب رقم كاميرا ثاني:  python space_gesture_game.py 1\n")
            return

        detect = self._make_detector(mp)
        if detect is None:
            return
        hist = deque(maxlen=3)
        stable = "none"
        print("[✓] الكاميرا شغالة")

        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            frame = cv2.flip(frame, 1)  # مرآة
            h, w = frame.shape[:2]
            lms = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            raw = self._classify(lms, w / h)
            hist.append(raw)
            if len(hist) >= 2 and hist[-1] == hist[-2]:
                stable = raw

            for lm in lms:
                pts = [(int(p.x * w), int(p.y * h)) for p in lm]
                for a, b in CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (255, 0, 200), 2)
                for pt in pts:
                    cv2.circle(frame, pt, 4, (0, 255, 255), -1)

            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok2:
                continue
            with self.cond:
                self.jpg = buf.tobytes()
                self.state = {"g": stable, "n": len(lms), "cam": True}
                self.seq += 1
                self.cond.notify_all()


VISION = Vision(CAM_INDEX)

# ───────────────────────────── السيرفر ─────────────────────────────
class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass  # المتصفح قطع الاتصال — عادي


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                body = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            elif self.path.startswith("/video"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last = -1
                while True:
                    with VISION.cond:
                        VISION.cond.wait_for(lambda: VISION.seq != last, timeout=3)
                        last, jpg = VISION.seq, VISION.jpg
                    if jpg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")

            elif self.path.startswith("/events"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last = -1
                while True:
                    with VISION.cond:
                        VISION.cond.wait_for(lambda: VISION.seq != last, timeout=1)
                        last, st = VISION.seq, dict(VISION.state)
                    self.wfile.write(("data: " + json.dumps(st) + "\n\n").encode())
                    self.wfile.flush()
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def main():
    VISION.start()
    srv = QuietServer(("127.0.0.1", PORT), Handler)
    srv.daemon_threads = True
    url = "http://127.0.0.1:%d" % PORT
    print("[✓] اللعبة شغالة على:", url, "   (Ctrl+C للإيقاف)")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nباي!")


# ───────────────────────────── اللعبة (HTML + CSS + JS) ─────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<title>حرب الفضاء بالكاميرا</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{--cyan:#35e0ff;--pink:#ff3fa4;--gold:#ffc447;--red:#ff4d5e;--ink:#050818;}
  *{box-sizing:border-box;margin:0}
  html,body{height:100%;overflow:hidden;background:var(--ink);color:#e9f4ff;
    font-family:"Segoe UI",Tahoma,"Noto Sans Arabic",Arial,sans-serif}
  canvas{position:fixed;inset:0;display:block}

  /* نافذة الكاميرا — أعلى يسار */
  #camBox{position:fixed;top:14px;left:14px;width:280px;border-radius:16px;overflow:hidden;
    border:2px solid var(--cyan);box-shadow:0 0 22px rgba(53,224,255,.45);background:#000;
    direction:ltr;transition:border-color .12s,box-shadow .12s;z-index:5}
  #camBox img{display:block;width:100%;aspect-ratio:4/3;object-fit:cover;background:#0a1030}
  #camLabel{position:absolute;left:0;right:0;bottom:0;padding:7px 10px;font-size:14px;font-weight:600;
    background:linear-gradient(transparent,rgba(0,0,0,.85));text-align:center;direction:rtl}
  #camBox.fist{border-color:#ffe14d;box-shadow:0 0 26px rgba(255,225,77,.7)}
  #camBox.point{border-color:#6dff8a;box-shadow:0 0 26px rgba(109,255,138,.7)}
  #camBox.bomb{border-color:var(--red);box-shadow:0 0 30px rgba(255,77,94,.85)}

  /* HUD */
  #hud{position:fixed;top:14px;right:18px;width:min(320px,40vw);z-index:5;display:none}
  .row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px}
  .row span{opacity:.75;font-size:14px}
  .row b{font-size:30px;font-variant-numeric:tabular-nums;text-shadow:0 0 14px var(--cyan)}
  .bar{height:12px;border-radius:8px;background:rgba(255,255,255,.1);overflow:hidden;margin:3px 0 10px;
    border:1px solid rgba(255,255,255,.18)}
  .bar i{display:block;height:100%;width:100%;border-radius:8px;transition:width .1s}
  #hp{background:linear-gradient(90deg,#ff3f5e,#ffd34d,#46ff9b)}
  #bb{background:linear-gradient(90deg,#7a5cff,#ff3fa4)}
  .small{font-size:12px;opacity:.7;margin-top:-6px;margin-bottom:6px}

  /* الشاشات */
  .screen{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;z-index:10;
    background:radial-gradient(ellipse at center,rgba(5,8,24,.6),rgba(5,8,24,.93))}
  .panel{max-width:760px;width:92vw;padding:30px 34px;border-radius:22px;
    background:rgba(10,16,48,.82);border:1px solid rgba(53,224,255,.35);
    box-shadow:0 0 60px rgba(53,224,255,.18),inset 0 0 40px rgba(255,63,164,.06)}
  h1{font-size:clamp(30px,5vw,52px);line-height:1.15;margin-bottom:6px;
    background:linear-gradient(90deg,var(--cyan),var(--pink));-webkit-background-clip:text;
    background-clip:text;color:transparent}
  .sub{opacity:.75;margin-bottom:20px}
  .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:18px}
  .g{padding:14px 10px;border-radius:14px;text-align:center;background:rgba(255,255,255,.05);
    border:1px solid rgba(255,255,255,.12);font-size:14px}
  .g .e{font-size:38px;display:block;margin-bottom:4px}
  .g b{display:block;margin-bottom:2px}
  ul.rules{list-style:none;padding:0;font-size:14px;line-height:1.9;margin-bottom:20px;opacity:.92}
  ul.rules li::before{content:"◆";color:var(--pink);margin-left:8px;font-size:10px}
  button{font:inherit;font-weight:700;font-size:18px;padding:13px 34px;border-radius:14px;cursor:pointer;
    color:#04101e;border:0;background:linear-gradient(90deg,var(--cyan),#7af0ff);
    box-shadow:0 0 24px rgba(53,224,255,.55)}
  button:hover{filter:brightness(1.1)}
  button:focus-visible{outline:3px solid #fff;outline-offset:3px}
  #over{display:none;text-align:center}
  #over .big{font-size:64px;font-weight:800;color:var(--gold);text-shadow:0 0 24px var(--gold)}
  @media (max-width:640px){.grid{grid-template-columns:1fr}#camBox{width:170px}}
</style>
</head>
<body>
<canvas id="game"></canvas>

<div id="camBox">
  <img id="cam" src="/video" alt="الكاميرا" onerror="this.style.opacity=.2">
  <div id="camLabel">⏳ جاري الاتصال بالكاميرا...</div>
</div>

<div id="hud">
  <div class="row"><span>النتيجة</span><b id="sc">0</b></div>
  <div class="row"><span>الموجة</span><b id="wv">1</b></div>
  <div class="small">الصحة</div><div class="bar"><i id="hp"></i></div>
  <div class="small">شحن القنبلة</div><div class="bar"><i id="bb"></i></div>
</div>

<div id="menu" class="screen"><div class="panel">
  <h1>حرب الفضاء بالكاميرا</h1>
  <p class="sub">إيدك هي السلاح، والكيبورد بيحرّك السفينة.</p>
  <div class="grid">
    <div class="g"><span class="e">✊</span><b>اقفل إيدك</b>طلقة وحدة</div>
    <div class="g"><span class="e">☝️</span><b>ارفع السبابة بس</b>5 طلقات باتجاهات مختلفة</div>
    <div class="g"><span class="e">🙏</span><b>ضم إيديك ببعض</b>قنبلة</div>
  </div>
  <ul class="rules">
    <li>الحركة: W A S D أو الأسهم — بدائل: Space طلقة، F خمس طلقات، B قنبلة.</li>
    <li>الطلقة ضررها 1: الاستكشافي (1 حياة)، المقاتل (3)، الدبابة (8)، الصخرة (2).</li>
    <li>طلقتك لما تصطدم بطلقة عدو بتلغيها وبتختفي الاثنتين.</li>
    <li>القنبلة: 8 ضرر لكل الأعداء، بتمسح طلقاتهم وبتفجّر الصخور. تحتاج 6 ثواني شحن.</li>
    <li>طلقة العدو تنقص 10 من صحتك، الصخرة 15، وتصادم عدو 25.</li>
  </ul>
  <button id="startBtn">ابدأ (Enter)</button>
</div></div>

<div id="over" class="screen"><div class="panel">
  <h1>انتهت المعركة</h1>
  <div class="big" id="finalScore">0</div>
  <p class="sub" id="finalWave"></p>
  <button id="againBtn">العب من جديد (Enter)</button>
</div></div>

<script>
const $ = id => document.getElementById(id);
const cv = $('game'), ctx = cv.getContext('2d');
let W = 0, H = 0;
function resize(){ W = cv.width = innerWidth; H = cv.height = innerHeight; }
addEventListener('resize', resize); resize();
const rnd = (a,b) => a + Math.random()*(b-a);
const clamp = (v,a,b) => Math.max(a, Math.min(b, v));
const d2 = (a,b) => (a.x-b.x)**2 + (a.y-b.y)**2;

/* ───── المدخلات ───── */
const keys = {};
let mode = 'menu';
addEventListener('keydown', e => {
  keys[e.code] = true;
  if (['Space','ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.code)) e.preventDefault();
  if (e.code === 'Enter' && mode !== 'play') start();
  if (mode === 'play' && !e.repeat) { if (e.code === 'KeyF') fireSpread(); if (e.code === 'KeyB') useBomb(); }
});
addEventListener('keyup', e => keys[e.code] = false);
$('startBtn').onclick = start; $('againBtn').onclick = start;

/* ───── اتصال حركات اليد من بايثون ───── */
const gest = {g:'none', n:0, cam:false};
const GNAME = {none:'✋ جاهز', fist:'✊ طلقة', point:'☝️ 5 طلقات', bomb:'🙏 قنبلة'};
function setLabel(){
  const box = $('camBox');
  box.className = gest.g === 'none' ? '' : gest.g;
  $('camLabel').textContent = gest.cam ? (gest.n ? GNAME[gest.g] : '🖐️ ورّيني إيدك') : '⌨️ الكاميرا مش متصلة — استخدم الكيبورد';
}
(function connect(){
  const es = new EventSource('/events');
  es.onmessage = e => { const d = JSON.parse(e.data); gest.g = d.g; gest.n = d.n; gest.cam = d.cam; setLabel(); };
  es.onerror = () => { gest.g = 'none'; gest.cam = false; setLabel(); };
})();

/* ───── حالة اللعبة ───── */
let P, bullets, ebullets, enemies, rocks, parts, rings;
let score = 0, wave = 1, shake = 0, flash = 0, spawnT = 1, rockT = 5, tAll = 0;
const stars = Array.from({length:150}, () => ({x:Math.random(), y:Math.random(), z:rnd(.2,1)}));
const TYPES = {
  scout:  {hp:1, r:16, score:10, col:'#4ff0ff', shoot:2.8, spd:70},
  fighter:{hp:3, r:23, score:30, col:'#ff4fa8', shoot:1.8, spd:55},
  tank:   {hp:8, r:34, score:80, col:'#ffc447', shoot:2.4, spd:35}
};

function start(){
  P = {x:W/2, y:H-110, vx:0, vy:0, hp:100, inv:0, fire:0, spread:0, bombCd:0};
  bullets=[]; ebullets=[]; enemies=[]; rocks=[]; parts=[]; rings=[];
  score=0; wave=1; spawnT=1; rockT=5; shake=0; flash=0;
  mode='play';
  $('menu').style.display='none'; $('over').style.display='none'; $('hud').style.display='block';
}
function gameOver(){
  mode='over';
  $('finalScore').textContent = score;
  $('finalWave').textContent = 'وصلت للموجة ' + wave;
  $('over').style.display='flex'; $('hud').style.display='none';
}

/* ───── الأسلحة ───── */
function shoot(angleDeg, spd){
  const a = angleDeg*Math.PI/180;
  bullets.push({x:P.x, y:P.y-24, vx:Math.cos(a)*spd, vy:Math.sin(a)*spd, r:5});
}
function fire1(){ shoot(-90, 950); P.fire = .2; }
function fireSpread(){
  if (P.spread > 0) return;
  [-120,-105,-90,-75,-60].forEach(a => shoot(a, 820));
  P.spread = .55;
}
function useBomb(){
  if (P.bombCd > 0) return;
  P.bombCd = 6; shake = 28; flash = 1;
  rings.push({x:P.x, y:P.y, r:20, max:Math.hypot(W,H), a:1});
  for (const e of enemies) hurt(e, 8);
  for (const b of ebullets){ b.dead = true; spark(b.x,b.y,'#ff9a5a',3); }
  for (const r of rocks){ r.dead = true; explode(r.x,r.y,'#b9a48c',r.r*.9); score += 5; }
}

/* ───── ضرر وانفجارات ───── */
function hurt(e, d){
  if (e.dead) return;
  e.hp -= d; e.hit = .08;
  if (e.hp <= 0){ e.dead = true; score += TYPES[e.k].score; explode(e.x, e.y, TYPES[e.k].col, e.r*1.4); }
}
function hurtPlayer(n){
  if (P.inv > 0) return;
  P.hp -= n; P.inv = .7; shake = Math.max(shake, 14);
  spark(P.x, P.y, '#ff5a6a', 14);
  if (P.hp <= 0){ explode(P.x,P.y,'#35e0ff',50); gameOver(); }
}
function spark(x,y,col,n){
  for (let i=0;i<n;i++){ const a=rnd(0,6.28), s=rnd(60,300);
    parts.push({x,y,vx:Math.cos(a)*s,vy:Math.sin(a)*s,life:rnd(.25,.6),max:.6,col,sz:rnd(1.5,3.5)}); }
}
function explode(x,y,col,size){
  const n = Math.min(70, 14+size|0);
  for (let i=0;i<n;i++){ const a=rnd(0,6.28), s=rnd(40,size*9);
    parts.push({x,y,vx:Math.cos(a)*s,vy:Math.sin(a)*s,life:rnd(.4,1),max:1,col:Math.random()<.35?'#fff':col,sz:rnd(2,5)}); }
  rings.push({x,y,r:4,max:size*2.2,a:.9,small:true,col});
  shake = Math.max(shake, size*.25);
}

/* ───── توليد الأعداء ───── */
function spawnEnemy(){
  const pool = ['scout']; if (wave>=2) pool.push('fighter','fighter'); if (wave>=3) pool.push('tank');
  const k = pool[Math.floor(Math.random()*pool.length)], T = TYPES[k];
  enemies.push({k, x:rnd(70,W-70), y:-50, hp:T.hp, max:T.hp, r:T.r, t:rnd(0,6), ty:rnd(120,H*.42), sh:rnd(.6,T.shoot), ph:rnd(0,6), hit:0});
}
function spawnRock(){
  rocks.push({x:rnd(40,W-40), y:-40, vx:rnd(-50,50), vy:rnd(110,190), r:rnd(18,30), hp:2, rot:0, vr:rnd(-2,2), hit:0,
    pts:Array.from({length:9},(_,i)=>[i/9*6.283, rnd(.78,1.1)])});
}
function enemyShoot(e){
  const T = TYPES[e.k];
  const aim = Math.atan2(P.y-e.y, P.x-e.x);
  const mk = (a,s) => ebullets.push({x:e.x,y:e.y+e.r*.6,vx:Math.cos(a)*s,vy:Math.sin(a)*s,r:6});
  if (e.k==='scout') mk(Math.PI/2, 250);
  else if (e.k==='fighter') mk(aim, 320);
  else { mk(aim-.3, 260); mk(aim, 280); mk(aim+.3, 260); }
  e.sh = T.shoot * rnd(.8,1.2) / (1 + wave*.06);
}

/* ───── التحديث ───── */
function update(dt){
  tAll += dt;
  const ax = (keys.KeyD||keys.ArrowRight?1:0) - (keys.KeyA||keys.ArrowLeft?1:0);
  const ay = (keys.KeyS||keys.ArrowDown?1:0) - (keys.KeyW||keys.ArrowUp?1:0);
  const k = Math.min(1, dt*10);
  P.vx += (ax*470 - P.vx)*k; P.vy += (ay*470 - P.vy)*k;
  P.x = clamp(P.x + P.vx*dt, 30, W-30); P.y = clamp(P.y + P.vy*dt, 50, H-40);
  P.fire -= dt; P.spread -= dt; P.bombCd -= dt; P.inv -= dt;

  if ((gest.g==='fist' || keys.Space) && P.fire<=0) fire1();
  if (gest.g==='point') fireSpread();
  if (gest.g==='bomb') useBomb();

  wave = 1 + Math.floor(score/300);
  spawnT -= dt; if (spawnT<=0){ spawnEnemy(); spawnT = Math.max(.55, 1.8 - wave*.12); }
  rockT -= dt;  if (rockT<=0){ spawnRock(); rockT = Math.max(2, 5.5 - wave*.3) * rnd(.7,1.2); }

  for (const b of bullets){ b.x+=b.vx*dt; b.y+=b.vy*dt; if (b.y<-30||b.x<-30||b.x>W+30) b.dead=true; }
  for (const b of ebullets){ b.x+=b.vx*dt; b.y+=b.vy*dt; if (b.y>H+30||b.y<-60||b.x<-30||b.x>W+30) b.dead=true; }
  for (const e of enemies){
    const T = TYPES[e.k]; e.t += dt; e.hit -= dt;
    if (e.y < e.ty) e.y += 110*dt;
    else { e.x += Math.cos(e.t*1.1+e.ph)*T.spd*dt*1.6; e.y += Math.sin(e.t*.8+e.ph)*12*dt; }
    e.x = clamp(e.x, 40, W-40);
    if (e.y > 0){ e.sh -= dt; if (e.sh<=0) enemyShoot(e); }
  }
  for (const r of rocks){ r.x+=r.vx*dt; r.y+=r.vy*dt; r.rot+=r.vr*dt; r.hit-=dt; if (r.y>H+60) r.dead=true; }

  /* تصادمات */
  for (const b of bullets){
    for (const e of enemies) if (!e.dead && d2(b,e) < (e.r+b.r)**2){ hurt(e,1); b.dead=true; spark(b.x,b.y,'#9ff',6); break; }
    if (b.dead) continue;
    for (const r of rocks) if (!r.dead && d2(b,r) < (r.r+b.r)**2){
      b.dead=true; r.hp--; r.hit=.08; spark(b.x,b.y,'#d8c2a6',6);
      if (r.hp<=0){ r.dead=true; score+=5; explode(r.x,r.y,'#b9a48c',r.r*1.1); } break; }
    if (b.dead) continue;
    for (const eb of ebullets) if (!eb.dead && d2(b,eb) < (b.r+eb.r+3)**2){ eb.dead=b.dead=true; spark(b.x,b.y,'#ffb36b',8); break; }
  }
  for (const eb of ebullets) if (!eb.dead && d2(eb,P) < (eb.r+16)**2){ eb.dead=true; hurtPlayer(10); }
  for (const e of enemies) if (!e.dead && d2(e,P) < (e.r+18)**2){ e.hp=0; hurt(e,1); hurtPlayer(25); }
  for (const r of rocks) if (!r.dead && d2(r,P) < (r.r+18)**2){ r.dead=true; explode(r.x,r.y,'#b9a48c',r.r); hurtPlayer(15); }

  bullets=bullets.filter(o=>!o.dead); ebullets=ebullets.filter(o=>!o.dead);
  enemies=enemies.filter(o=>!o.dead); rocks=rocks.filter(o=>!o.dead);

  for (const p of parts){ p.x+=p.vx*dt; p.y+=p.vy*dt; p.vx*=.985; p.vy*=.985; p.life-=dt; }
  parts = parts.filter(p=>p.life>0);
  for (const r of rings){ r.r += (r.small ? 320 : 1500)*dt; r.a = Math.max(0, 1 - r.r/r.max); }
  rings = rings.filter(r=>r.r<r.max);
  shake *= Math.pow(.02, dt); flash = Math.max(0, flash - dt*2.2);

  $('sc').textContent = score; $('wv').textContent = wave;
  $('hp').style.width = Math.max(0,P.hp) + '%';
  $('bb').style.width = (100 - clamp(P.bombCd/6,0,1)*100) + '%';
}

/* ───── الرسم ───── */
function drawBG(dt){
  const g = ctx.createLinearGradient(0,0,0,H);
  g.addColorStop(0,'#04061a'); g.addColorStop(.6,'#0a0f35'); g.addColorStop(1,'#1a0b36');
  ctx.fillStyle = g; ctx.fillRect(0,0,W,H);
  const neb = (x,y,r,c) => { const n = ctx.createRadialGradient(x,y,0,x,y,r); n.addColorStop(0,c); n.addColorStop(1,'transparent'); ctx.fillStyle=n; ctx.fillRect(0,0,W,H); };
  neb(W*.2 + Math.sin(tAll*.1)*40, H*.3, H*.6, 'rgba(255,63,164,.13)');
  neb(W*.8, H*.7 + Math.cos(tAll*.12)*40, H*.7, 'rgba(53,224,255,.11)');
  for (const s of stars){
    s.y += s.z*60*dt*(mode==='play'?1:.4); if (s.y>1){ s.y=0; s.x=Math.random(); }
    const tw = .6 + .4*Math.sin(tAll*3 + s.x*40);
    ctx.fillStyle = 'rgba(255,255,255,' + (s.z*tw) + ')';
    ctx.fillRect(s.x*W, s.y*H, s.z*2.2, s.z*2.2 + s.z*3);
  }
}
function drawPlayer(){
  if (P.inv>0 && Math.floor(P.inv*20)%2===0) return;
  ctx.save(); ctx.translate(P.x,P.y);
  ctx.globalCompositeOperation='lighter';
  const fl = 14 + Math.random()*14;
  const fg = ctx.createLinearGradient(0,18,0,18+fl+14);
  fg.addColorStop(0,'rgba(255,255,255,.95)'); fg.addColorStop(.4,'rgba(80,200,255,.8)'); fg.addColorStop(1,'transparent');
  ctx.fillStyle=fg; ctx.beginPath(); ctx.moveTo(-8,20); ctx.lineTo(0,20+fl+14); ctx.lineTo(8,20); ctx.fill();
  ctx.globalCompositeOperation='source-over';
  ctx.shadowColor='#35e0ff'; ctx.shadowBlur=18;
  const body = ctx.createLinearGradient(-30,0,30,0);
  body.addColorStop(0,'#2a5cff'); body.addColorStop(.5,'#9fe9ff'); body.addColorStop(1,'#2a5cff');
  ctx.fillStyle=body; ctx.beginPath();
  ctx.moveTo(0,-30); ctx.lineTo(10,-6); ctx.lineTo(32,18); ctx.lineTo(14,13); ctx.lineTo(9,24);
  ctx.lineTo(-9,24); ctx.lineTo(-14,13); ctx.lineTo(-32,18); ctx.lineTo(-10,-6); ctx.closePath(); ctx.fill();
  ctx.shadowBlur=0;
  ctx.fillStyle='#ff3fa4'; ctx.fillRect(-26,12,5,8); ctx.fillRect(21,12,5,8);
  ctx.fillStyle='#07123a'; ctx.beginPath(); ctx.ellipse(0,-6,5,10,0,0,6.3); ctx.fill();
  ctx.fillStyle='rgba(120,240,255,.85)'; ctx.beginPath(); ctx.ellipse(0,-8,2.5,6,0,0,6.3); ctx.fill();
  ctx.restore();
}
function drawEnemy(e){
  const T = TYPES[e.k], c = e.hit>0 ? '#fff' : T.col;
  ctx.save(); ctx.translate(e.x,e.y);
  ctx.shadowColor = T.col; ctx.shadowBlur = 16; ctx.fillStyle = c;
  ctx.beginPath();
  if (e.k==='scout'){ ctx.moveTo(0,18); ctx.lineTo(16,-12); ctx.lineTo(0,-4); ctx.lineTo(-16,-12); }
  else if (e.k==='fighter'){ ctx.moveTo(0,22); ctx.lineTo(10,2); ctx.lineTo(28,-14); ctx.lineTo(12,-16); ctx.lineTo(0,-8); ctx.lineTo(-12,-16); ctx.lineTo(-28,-14); ctx.lineTo(-10,2); }
  else { for (let i=0;i<6;i++){ const a=i/6*6.283+.52; ctx.lineTo(Math.cos(a)*e.r, Math.sin(a)*e.r); } }
  ctx.closePath(); ctx.fill(); ctx.shadowBlur = 0;
  ctx.fillStyle = 'rgba(5,8,30,.8)';
  if (e.k==='tank'){ ctx.beginPath(); ctx.arc(0,0,e.r*.55,0,6.3); ctx.fill();
    ctx.fillStyle = c; ctx.beginPath(); ctx.arc(0,0,e.r*.28 + Math.sin(tAll*6)*2,0,6.3); ctx.fill(); }
  else { ctx.beginPath(); ctx.arc(0,-2,5,0,6.3); ctx.fill(); ctx.fillStyle=c; ctx.beginPath(); ctx.arc(0,-2,2.2,0,6.3); ctx.fill(); }
  if (e.max>1){ ctx.fillStyle='rgba(0,0,0,.55)'; ctx.fillRect(-e.r,-e.r-14,e.r*2,5);
    ctx.fillStyle=T.col; ctx.fillRect(-e.r,-e.r-14,e.r*2*e.hp/e.max,5); }
  ctx.restore();
}
function drawRock(r){
  ctx.save(); ctx.translate(r.x,r.y); ctx.rotate(r.rot);
  ctx.fillStyle = r.hit>0 ? '#fff' : '#8a7a6a'; ctx.strokeStyle='#cbb89f'; ctx.lineWidth=2;
  ctx.beginPath(); r.pts.forEach(([a,m],i)=>{ const x=Math.cos(a)*r.r*m, y=Math.sin(a)*r.r*m; i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
  ctx.closePath(); ctx.fill(); ctx.stroke();
  ctx.fillStyle='rgba(0,0,0,.25)'; ctx.beginPath(); ctx.arc(-r.r*.25,-r.r*.2,r.r*.22,0,6.3); ctx.fill();
  ctx.restore();
}
function render(dt){
  ctx.save();
  if (shake>.5) ctx.translate(rnd(-shake,shake), rnd(-shake,shake));
  drawBG(dt);
  if (mode==='play' || mode==='over'){
    rocks.forEach(drawRock); enemies.forEach(drawEnemy);
    ctx.globalCompositeOperation='lighter';
    for (const b of bullets){
      const ang = Math.atan2(b.vy,b.vx);
      ctx.save(); ctx.translate(b.x,b.y); ctx.rotate(ang);
      const g = ctx.createLinearGradient(-22,0,8,0); g.addColorStop(0,'transparent'); g.addColorStop(1,'#bff8ff');
      ctx.fillStyle=g; ctx.shadowColor='#35e0ff'; ctx.shadowBlur=14;
      ctx.beginPath(); ctx.roundRect(-22,-3,30,6,3); ctx.fill(); ctx.restore();
    }
    for (const b of ebullets){
      const g = ctx.createRadialGradient(b.x,b.y,0,b.x,b.y,b.r*2.6);
      g.addColorStop(0,'#fff'); g.addColorStop(.35,'#ff7a5a'); g.addColorStop(1,'transparent');
      ctx.fillStyle=g; ctx.beginPath(); ctx.arc(b.x,b.y,b.r*2.6,0,6.3); ctx.fill();
    }
    for (const p of parts){
      ctx.globalAlpha = Math.max(0,p.life/p.max); ctx.fillStyle=p.col;
      ctx.fillRect(p.x-p.sz/2,p.y-p.sz/2,p.sz,p.sz);
    }
    ctx.globalAlpha = 1;
    for (const r of rings){
      ctx.lineWidth = r.small ? 3 : 14; ctx.strokeStyle = r.small ? r.col : '#ff8ae0';
      ctx.globalAlpha = r.a; ctx.beginPath(); ctx.arc(r.x,r.y,r.r,0,6.3); ctx.stroke();
      if (!r.small){ ctx.fillStyle='rgba(255,140,230,.08)'; ctx.fill(); }
    }
    ctx.globalAlpha = 1; ctx.globalCompositeOperation='source-over';
    if (mode==='play') drawPlayer();
  }
  ctx.restore();
  if (flash>0){ ctx.fillStyle='rgba(255,255,255,'+(flash*.45)+')'; ctx.fillRect(0,0,W,H); }
}

/* ───── الحلقة الرئيسية ───── */
let last = performance.now();
function loop(now){
  const dt = Math.min(.05, (now-last)/1000); last = now;
  if (mode==='play') update(dt);
  else {
    tAll += dt; if (parts) { for (const p of parts){ p.x+=p.vx*dt; p.y+=p.vy*dt; p.life-=dt; } parts=parts.filter(p=>p.life>0); }
    shake *= Math.pow(.02, dt);
  }
  render(dt);
  requestAnimationFrame(loop);
}
parts = []; bullets=[]; ebullets=[]; enemies=[]; rocks=[]; rings=[]; P={x:0,y:0,inv:0};
setLabel();
requestAnimationFrame(loop);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
