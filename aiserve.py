import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import os
import re
import subprocess
import shutil
import glob
from datetime import datetime
import time

# --- MATERIAL DESIGN 3 DARK PALETTE ---
MD_BG = "#111318"
MD_SURFACE = "#1E2129"
MD_SURFACE_VARIANT = "#2B2D31"
MD_PRIMARY = "#A8C7FA"
MD_ON_PRIMARY = "#062E6F"
MD_ERROR = "#F2B8B5"
MD_TEXT_MAIN = "#E3E3E3"
MD_TEXT_SUB = "#C4C7C5"

FONT_HEADER = ("Segoe UI", 16, "bold")
FONT_BODY = ("Segoe UI", 10)
FONT_BTN = ("Segoe UI", 10, "bold")

# --- CUSTOM ROUNDED BUTTON ENGINE ---
class RoundedButton(tk.Canvas):
    def __init__(self, parent, width, height, corner_radius, padding, color, fg, text, command=None, click_color=None):
        tk.Canvas.__init__(self, parent, borderwidth=0,
            relief="flat", highlightthickness=0, bg=parent["bg"])
        self.command = command
        self.default_color = color
        self.fg = fg
        self.text_content = text
        self.click_color = click_color if click_color else "#444746"

        self.hover_color = "#35383F"
        if color == MD_PRIMARY: self.hover_color = "#82B4FF"
        if color == MD_SURFACE_VARIANT: self.hover_color = "#3C4043"

        self.corner_radius = corner_radius
        self.width = width
        self.height = height

        self.configure(width=width, height=height)
        self.draw(self.default_color)

        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def draw(self, color):
        self.delete("all")
        self._draw_rounded_rect(0, 0, self.width, self.height, self.corner_radius, color)
        self.text_id = self.create_text(self.width/2, self.height/2, text=self.text_content, fill=self.fg, font=FONT_BTN)

    def _draw_rounded_rect(self, x, y, w, h, r, color):
        points = (x+r, y, x+r, y, x+w-r, y, x+w-r, y, x+w, y, x+w, y+r, x+w, y+r, x+w, y+h-r, x+w, y+h-r, x+w, y+h, x+w-r, y+h, x+w-r, y+h, x+r, y+h, x+r, y+h, x, y+h, x, y+h-r, x, y+h-r, x, y+r, x, y+r, x, y)
        return self.create_polygon(points, smooth=True, fill=color)

    def _on_press(self, event):
        self.draw(self.click_color)
        if self.command: self.command()

    def _on_release(self, event):
        self.draw(self.hover_color)

    def _on_enter(self, event):
        self.draw(self.hover_color)

    def _on_leave(self, event):
        self.draw(self.default_color)

    def config_text(self, text):
        self.itemconfig(self.text_id, text=text)

# --- MAIN APP ---
class AiServeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AiServe SL")
        self.root.geometry("1280x850")
        self.root.minsize(1000, 700)
        self.root.configure(bg=MD_BG)

        # --- Variables ---
        self.video_path = ""
        self.srt_path = ""
        self.output_folder = ""
        self.report_path = ""

        self.cap = None
        self.flight_data = []
        self.is_playing = False
        self.total_frames = 0
        self.fps = 30
        self.current_frame = 0

        self.current_gps = None
        self.last_valid_gps = None
        self.last_raw_frame = None

        self.ai_active = tk.BooleanVar(value=False)
        self.debug_view = tk.BooleanVar(value=False)
        self.ai_sensitivity = 1000
        self.last_auto_snap_time = 0
        self.COOLDOWN_SECONDS = 2.0
        self.frame_skip_var = tk.IntVar(value=2)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Horizontal.TScale", background=MD_BG, troughcolor=MD_SURFACE_VARIANT, borderwidth=0)

        self.setup_ui()
        self.update_video()

    def setup_ui(self):
        # 1. LEFT SIDEBAR (Fixed Width)
        sidebar = tk.Frame(self.root, bg=MD_SURFACE, width=320, padx=20, pady=20)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="AiServe SL", bg=MD_SURFACE, fg=MD_TEXT_MAIN, font=("Helvetica", 24, "bold")).pack(anchor="w")
        tk.Label(sidebar, text="Autonomous Inspection", bg=MD_SURFACE, fg=MD_PRIMARY, font=("Helvetica", 10)).pack(anchor="w", pady=(0, 30))

        # Sidebar Components
        self.create_header(sidebar, "Mission Setup")
        self.btn_vid = RoundedButton(sidebar, 280, 45, 20, 0, MD_SURFACE_VARIANT, MD_TEXT_MAIN, "📂 Load Video", self.load_video)
        self.btn_vid.pack(pady=6)
        self.btn_srt = RoundedButton(sidebar, 280, 45, 20, 0, MD_SURFACE_VARIANT, MD_TEXT_MAIN, "📄 Load SRT (Auto)", self.load_srt)
        self.btn_srt.pack(pady=6)
        self.btn_fld = RoundedButton(sidebar, 280, 45, 20, 0, MD_SURFACE_VARIANT, MD_TEXT_MAIN, "📂 Select Output", self.select_folder)
        self.btn_fld.pack(pady=6)
        self.lbl_folder = tk.Label(sidebar, text="No folder selected", bg=MD_SURFACE, fg=MD_TEXT_SUB, font=("Helvetica", 9))
        self.lbl_folder.pack(pady=5)

        tk.Frame(sidebar, height=20, bg=MD_SURFACE).pack()
        self.create_header(sidebar, "AI Neural Core")
        ai_box = tk.Frame(sidebar, bg=MD_SURFACE)
        ai_box.pack(fill="x", pady=5)
        self.chk_ai = tk.Checkbutton(ai_box, text=" Enable Veg Detection", variable=self.ai_active,
                                     bg=MD_SURFACE, fg=MD_TEXT_MAIN, selectcolor=MD_BG,
                                     activebackground=MD_SURFACE, activeforeground=MD_PRIMARY,
                                     font=FONT_BODY, command=self.toggle_ai_msg)
        self.chk_ai.pack(anchor="w")
        tk.Label(ai_box, text="Sensitivity", bg=MD_SURFACE, fg=MD_TEXT_SUB, font=("Helvetica", 9)).pack(anchor="w", pady=(10,0))
        self.slider_sens = ttk.Scale(ai_box, from_=500, to=10000, orient="horizontal", command=self.update_sensitivity)
        self.slider_sens.set(1000)
        self.slider_sens.pack(fill="x", pady=5)
        self.lbl_ai_status = tk.Label(sidebar, text="System Ready", bg=MD_SURFACE, fg=MD_TEXT_SUB, font=("Helvetica", 9))
        self.lbl_ai_status.pack(pady=5)

        tk.Frame(sidebar, height=20, bg=MD_SURFACE).pack()
        self.create_header(sidebar, "Export")
        self.btn_gallery = RoundedButton(sidebar, 280, 45, 20, 0, MD_SURFACE_VARIANT, MD_TEXT_MAIN, "Collections", self.open_gallery)
        self.btn_gallery.pack(pady=6)
        self.btn_kml = RoundedButton(sidebar, 280, 45, 20, 0, MD_SURFACE_VARIANT, MD_TEXT_MAIN, "Export KML Map", self.generate_kml)
        self.btn_kml.pack(pady=6)

        # 2. MAIN AREA (Right Side Container)
        main_area = tk.Frame(self.root, bg=MD_BG)
        main_area.pack(side="right", expand=True, fill="both", padx=20, pady=20)

        # 3. BOTTOM BAR (Packed FIRST inside Main Area so it sticks to bottom)
        bottom_bar = tk.Frame(main_area, bg=MD_BG)
        bottom_bar.pack(side="bottom", fill="x", pady=(10, 0))

        # Bottom Bar Components
        # A. Controls Row
        ctrl_frame = tk.Frame(bottom_bar, bg=MD_BG)
        ctrl_frame.pack(side="bottom", fill="x", pady=10)

        self.btn_play = RoundedButton(ctrl_frame, 120, 50, 25, 0, MD_PRIMARY, MD_ON_PRIMARY, "▶ PLAY", self.toggle_play)
        self.btn_play.pack(side="left", padx=20)

        self.btn_snap = RoundedButton(ctrl_frame, 160, 50, 25, 0, MD_SURFACE_VARIANT, MD_TEXT_MAIN, "📷 SNAPSHOT", self.take_snapshot, click_color=MD_ERROR)
        self.btn_snap.pack(side="right")

        spd_frm = tk.Frame(ctrl_frame, bg=MD_BG)
        spd_frm.pack(side="left", padx=20)
        tk.Label(spd_frm, text="Speed", bg=MD_BG, fg=MD_TEXT_SUB).pack(side="left")
        self.slider_speed = ttk.Scale(spd_frm, from_=1, to=30, orient="horizontal", variable=self.frame_skip_var)
        self.slider_speed.pack(side="left", padx=10)

        # B. Timeline
        self.slider = ttk.Scale(bottom_bar, from_=0, to=100, orient="horizontal", command=self.on_slider, style="Horizontal.TScale")
        self.slider.pack(side="bottom", fill="x", pady=5)

        # C. GPS Text
        self.lbl_coords = tk.Label(bottom_bar, text="Lat: --.------   Long: --.------", bg=MD_BG, fg=MD_PRIMARY, font=("Courier", 12))
        self.lbl_coords.pack(side="bottom", pady=5)

        # 4. VIDEO CONTAINER (Packed LAST to fill remaining space)
        video_border = tk.Frame(main_area, bg=MD_SURFACE_VARIANT, padx=1, pady=1)
        video_border.pack(side="top", expand=True, fill="both")

        self.video_container = tk.Label(video_border, bg="black", text="No Signal", fg="#333", font=("Helvetica", 16))
        self.video_container.pack(expand=True, fill="both")

    def create_header(self, parent, text):
        tk.Label(parent, text=text, bg=MD_SURFACE, fg=MD_PRIMARY, font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(0, 10))

    # --- LOGIC ---
    def load_video(self):
        f = filedialog.askopenfilename(filetypes=[("Video", "*.MP4 *.MOV")])
        if f:
            self.video_path = f
            self.cap = cv2.VideoCapture(f)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.slider.config(to=self.total_frames)
            self.btn_vid.config_text("✔ Video Ready")

            base = os.path.splitext(f)[0]
            found = False
            for ext in [".SRT", ".srt", ".Srt"]:
                if os.path.exists(base + ext):
                    self.load_srt(base + ext)
                    found = True
                    break
            if not found: self.extract_embedded_telemetry(f)
            self.show_frame(0)

    def extract_embedded_telemetry(self, video_file):
        self.lbl_coords.config(text="Extracting GPS...", fg=MD_ERROR)
        self.root.update()
        if shutil.which("ffmpeg") is None:
            messagebox.showwarning("Info", "Install FFmpeg for embedded GPS.")
            return
        try:
            temp = os.path.join(os.path.dirname(video_file), "temp_gps.srt")
            subprocess.run(["ffmpeg", "-i", video_file, "-map", "0:s:0", temp, "-y"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(temp) and os.path.getsize(temp)>0: self.load_srt(temp)
        except: pass

    def load_srt(self, path=None):
        if not path: path = filedialog.askopenfilename()
        if not path: return
        self.srt_path = path
        self.btn_srt.config_text("✔ GPS Linked")
        self.parse_srt(path)

    def parse_srt(self, path):
        self.flight_data = []
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
            blocks = content.split('\n\n')
            for block in blocks:
                lines = block.split('\n')
                if len(lines) < 3: continue
                idx = -1
                for i, l in enumerate(lines):
                    if '-->' in l: idx = i; break
                if idx == -1: continue
                try:
                    s = lines[idx].split(' --> ')[0].strip()
                    h, m, sec = s.split(':')
                    sec, ms = sec.split(',')
                    total_sec = int(h)*3600 + int(m)*60 + int(sec) + int(ms)/1000.0
                except: continue
                text = " ".join(lines[idx+1:])
                lat = re.search(r'(?:lat|latitude)[\D]*?([-+]?\d+\.\d+)', text, re.IGNORECASE)
                lon = re.search(r'(?:lon|long|longitude|lng)[\D]*?([-+]?\d+\.\d+)', text, re.IGNORECASE)
                if lat and lon:
                    self.flight_data.append({'seconds': total_sec, 'lat': float(lat.group(1)), 'lon': float(lon.group(1))})
            if self.flight_data: self.lbl_coords.config(text=f"GPS Linked ({len(self.flight_data)} pts)", fg=MD_PRIMARY)
        except: pass

    def detect_vegetation_on_panel(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        disp = frame_bgr.copy()

        mask_p = cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,120]))
        kernel = np.ones((5,5), np.uint8)
        mask_p = cv2.morphologyEx(mask_p, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_p = cv2.erode(mask_p, kernel, iterations=1)
        cnts_p, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_panels = []
        for c in cnts_p:
            if cv2.contourArea(c) > 5000:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04*peri, True)
                if 3 <= len(approx) <= 8:
                    valid_panels.append(c)
                    cv2.drawContours(disp, [c], -1, (255, 0, 0), 2)

        mask_v = cv2.inRange(hsv, np.array([30,40,40]), np.array([90,255,255]))
        cnts_v, _ = cv2.findContours(mask_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = False
        for c in cnts_v:
            if cv2.contourArea(c) > self.ai_sensitivity:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    for p in valid_panels:
                        if cv2.pointPolygonTest(p, (cx, cy), False) >= 0:
                            detected = True
                            cv2.drawContours(disp, [c], -1, (0, 255, 0), 2)
                            x,y,w,h = cv2.boundingRect(c)
                            cv2.rectangle(disp, (x,y), (x+w,y+h), (0,0,255), 2)
                            break
        return detected, disp

    def update_sensitivity(self, val): self.ai_sensitivity = int(float(val))
    def toggle_ai_msg(self): self.lbl_ai_status.config(text="AI Active" if self.ai_active.get() else "Ready", fg=MD_PRIMARY)

    def select_folder(self):
        f = filedialog.askdirectory()
        if f:
            self.output_folder = os.path.abspath(f)
            self.lbl_folder.config(text=f"...{self.output_folder[-25:]}")
            self.btn_fld.config_text("✔ Output Set")
            ts = datetime.now().strftime("%Y%m%d")
            self.report_path = os.path.join(self.output_folder, f"Report_{ts}.xlsx")
            self.kml_path = os.path.join(self.output_folder, f"Map_{ts}.kml")

    def take_snapshot(self, auto=False):
        if not self.cap or not self.output_folder: return
        if self.last_raw_frame is None: return

        lat, lon = 0.0, 0.0
        if self.current_gps: lat, lon = self.current_gps['lat'], self.current_gps['lon']
        elif self.last_valid_gps: lat, lon = self.last_valid_gps['lat'], self.last_valid_gps['lon']

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        frame = self.last_raw_frame.copy()
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, h-60), (w, h), (0,0,0), -1)

        scale = w / 1400.0 if w > 1000 else 0.6
        cv2.putText(frame, f"{ts} | {lat:.6f}, {lon:.6f}", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,255,255), 2)

        fname = f"Snap_{datetime.now().strftime('%H%M%S_%f')[:9]}.jpg"
        cv2.imwrite(os.path.join(self.output_folder, fname), frame)

        data = {'Timestamp': ts, 'Image': fname, 'Lat': lat, 'Lon': lon, 'Type': "AI" if auto else "Manual"}
        try:
            if os.path.exists(self.report_path): pd.concat([pd.read_excel(self.report_path), pd.DataFrame([data])]).to_excel(self.report_path, index=False)
            else: pd.DataFrame([data]).to_excel(self.report_path, index=False)
            if auto:
                self.lbl_ai_status.config(text="Auto-Saved", fg=MD_ERROR)
                self.root.after(1000, self.toggle_ai_msg)
        except: pass

    def open_gallery(self):
        if not self.output_folder: return
        gal = tk.Toplevel(self.root)
        gal.title("Gallery")
        gal.geometry("1000x700")
        gal.configure(bg=MD_BG)

        files = glob.glob(os.path.join(self.output_folder, "*.jpg"))
        files.sort(key=os.path.getmtime, reverse=True)

        lb = tk.Listbox(gal, bg=MD_SURFACE, fg="white", bd=0)
        lb.pack(side="left", fill="y")
        cv = tk.Canvas(gal, bg="black", bd=0)
        cv.pack(side="right", expand=True, fill="both")

        for f in files: lb.insert(tk.END, os.path.basename(f))

        self.z_scale = 1.0
        self.g_img = None

        def show(e):
            if not lb.curselection(): return
            path = os.path.join(self.output_folder, lb.get(lb.curselection()[0]))
            self.g_img = Image.open(path)
            self.z_scale = 1.0
            redraw()

        def redraw():
            if not self.g_img: return
            w, h = self.g_img.size
            nw, nh = int(w*self.z_scale), int(h*self.z_scale)
            i = ImageTk.PhotoImage(self.g_img.resize((nw, nh), Image.Resampling.LANCZOS))
            cv.delete("all")
            cv.create_image(cv.winfo_width()//2, cv.winfo_height()//2, image=i, anchor="center")
            cv.img = i

        def zoom(e):
            if not self.g_img: return
            if e.num==5 or e.delta<0: self.z_scale = max(0.2, self.z_scale-0.1)
            else: self.z_scale = min(5.0, self.z_scale+0.1)
            redraw()

        lb.bind("<<ListboxSelect>>", show)
        gal.bind("<MouseWheel>", zoom)
        gal.bind("<Button-4>", zoom)
        gal.bind("<Button-5>", zoom)

        if files:
            lb.selection_set(0)
            show(None)

    def generate_kml(self):
        if not hasattr(self, 'kml_path') or not os.path.exists(self.report_path): return
        try:
            df = pd.read_excel(self.report_path)
            kml = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2">', '<Document>']
            for _, r in df.iterrows():
                path = os.path.join(self.output_folder, str(r['Image']))
                kml.append(f"<Placemark><name>{r['Type']}</name><description><![CDATA[<img src='file://{path}' width='400'/>]]></description><Point><coordinates>{r['Lon']},{r['Lat']},0</coordinates></Point></Placemark>")
            kml.append('</Document></kml>')
            with open(self.kml_path, "w") as f: f.write("".join(kml))
            messagebox.showinfo("Done", "KML Generated")
        except: pass

    def show_frame(self, frame_num):
        self.current_frame = frame_num
        self.process_frame(frame_num)

    def update_video(self):
        if self.is_playing and self.cap:
            self.current_frame += self.frame_skip_var.get()
            if self.current_frame >= self.total_frames:
                self.is_playing = False
                self.btn_play.config_text("▶ PLAY")
            else:
                self.slider.set(self.current_frame)
                self.process_frame(self.current_frame)
        self.root.after(30, self.update_video)

    def process_frame(self, frame_num):
        if not self.cap: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if ret:
            self.last_raw_frame = frame.copy()
            disp = frame

            if self.ai_active.get() or self.debug_view.get():
                is_veg, ai_disp = self.detect_vegetation_on_panel(frame)
                disp = ai_disp
                if self.ai_active.get() and is_veg and (time.time()-self.last_auto_snap_time > self.COOLDOWN_SECONDS):
                    self.take_snapshot(auto=True)
                    self.last_auto_snap_time = time.time()

            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

            # --- AUTO RESIZING LOGIC ---
            w_gui = self.video_container.winfo_width()
            h_gui = self.video_container.winfo_height()

            if w_gui < 100: w_gui, h_gui = 800, 600

            w_gui -= 4
            h_gui -= 4

            h, w = rgb.shape[:2]
            scale = min(w_gui/w, h_gui/h)
            new_w, new_h = int(w*scale), int(h*scale)

            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_container.config(image=img, text="")
            self.video_container.image = img
            self.update_gps(frame_num)

    def update_gps(self, frame_num):
        if not self.flight_data: return
        sec = frame_num / self.fps
        c = min(self.flight_data, key=lambda x: abs(x['seconds'] - sec))
        if abs(c['seconds'] - sec) < 1.5:
            self.current_gps = c
            self.last_valid_gps = c
            self.lbl_coords.config(text=f"Lat: {c['lat']:.6f}   Long: {c['lon']:.6f}", fg=MD_PRIMARY)
        else:
            self.current_gps = None
            if self.last_valid_gps: self.lbl_coords.config(text=f"Last: {self.last_valid_gps['lat']:.6f}   {self.last_valid_gps['lon']:.6f}", fg=MD_TEXT_SUB)
            else: self.lbl_coords.config(text="Searching GPS...", fg=MD_ERROR)

    def toggle_play(self):
        if self.cap:
            self.is_playing = not self.is_playing
            self.btn_play.config_text("⏸ PAUSE" if self.is_playing else "▶ PLAY")

    def seek(self, d): pass
    def on_slider(self, v):
        if not self.is_playing:
            self.current_frame = int(float(v))
            self.process_frame(self.current_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = AiServeApp(root)
    root.mainloop()
