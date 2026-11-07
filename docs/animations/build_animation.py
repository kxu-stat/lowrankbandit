"""Turn an overlay PDF into a README animation.

    pdflatex two_track_animation.tex          # one PDF page per overlay
    python3 build_animation.py two_track_animation
    pdflatex greedy_flow_animation.tex
    python3 build_animation.py greedy_flow_animation

Writes NAME.gif and NAME.mp4.  Needs pdftoppm (poppler) and ffmpeg.

SECONDS gives how long each page is shown, one number per page, grouped by caption.  The numbers follow one
rule: a caption stays 0.8 s plus 1 s per 5 words (a formula counts as one word per 12 characters of its
LaTeX), and each animation step lasts at least 0.7-1.3 s; the extra time goes to the last page of the
caption.  SCALE multiplies every number: 1.0 gives 67 s and 53 s, 0.9 gives 60 s and 48 s.
"""

import glob
import os
import subprocess
import sys

SCALE = 0.9
DPI = 150

SECONDS = {
    "two_track_animation": [
        4.2,                                          # 1      overview (goal)
        0.8, 0.8, 0.8, 1.5,                           # 2-5    Step 1: exploration
        0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,            # 6-12   Step 1: two tracks
        0.8, 4.0,                                     # 13-14  Step 1: independence
        0.8, 5.6,                                     # 15-16  Step 2: preliminary fit
        0.7, 0.7, 0.7, 0.7, 3.6,                      # 17-21  Step 2: predictable refit
        0.8, 0.8, 0.8, 3.5,                           # 22-25  Step 2: truncation; Track 2
        0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 1.4,  # 26-34  Step 3: calibration along Track 2
        5.4,                                          # 35     Step 3: fold 1 formulas
        0.8, 0.7, 0.7, 0.7, 0.7, 0.8,                 # 36-41  Step 3: roles exchanged
        1.3, 1.3, 3.4,                                # 42-44  cross-fitting
        3.8,                                          # 45     Step 4: aggregation
        2.6,                                          # 46     complete figure
    ],
    "greedy_flow_animation": [
        4.8,                                          # 1      overview
        2.6, 3.0, 1.0, 3.0, 2.2,                      # 2-6    exploration, round 1
        1.2, 2.0, 3.2,                                # 7-9    rounds 2 and tau; first fit
        2.6, 4.2, 4.0, 1.0, 1.4, 1.0, 3.0, 3.0,       # 10-17  greedy, round tau+1
        1.2, 2.0, 2.0,                                # 18-20  later rounds; round n
        5.0,                                          # 21     complete figure
    ],
}

name = sys.argv[1]
folder = "pages_" + name
os.makedirs(folder, exist_ok=True)
for old in glob.glob(folder + "/*"):
    os.remove(old)
subprocess.run(["pdftoppm", "-r", str(DPI), "-png", name + ".pdf", folder + "/p"], check=True)
pages = sorted(glob.glob(folder + "/p-*.png"))
seconds = [SCALE * s for s in SECONDS[name]]
if len(seconds) != len(pages):
    sys.exit("%s: %d pages but %d durations" % (name, len(pages), len(seconds)))
print("%s: %d pages, %.1f s" % (name, len(pages), sum(seconds)))

# ffmpeg concat list: each page with its display time (the last page is listed twice, as ffmpeg requires)
with open(folder + "/list.txt", "w") as f:
    for page, s in zip(pages, seconds):
        f.write("file '%s'\nduration %.2f\n" % (os.path.basename(page), s))
    f.write("file '%s'\n" % os.path.basename(pages[-1]))

concat = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", folder + "/list.txt"]
subprocess.run(concat + ["-vf", "palettegen=max_colors=128:stats_mode=full", folder + "/palette.png"], check=True)
subprocess.run(concat + ["-i", folder + "/palette.png", "-lavfi", "paletteuse=dither=none",
                         "-fps_mode", "vfr", name + ".gif"], check=True)
subprocess.run(concat + ["-vf", "fps=10,scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
                         "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", name + ".mp4"], check=True)
