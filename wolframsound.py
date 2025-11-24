import numpy as np
from PIL import Image
from scipy.io.wavfile import write
from moviepy import ImageSequenceClip, AudioFileClip

# -----------------------------
# 1) CA generation
# -----------------------------
def wolfram_ca(rule_num, width=128, steps=600, wrap=True):
    rule = np.array([(rule_num >> i) & 1 for i in range(8)], dtype=np.uint8)
    state = np.zeros(width, dtype=np.uint8)
    state[width // 2] = 1
    rows = [state.copy()]
    for _ in range(steps - 1):
        if wrap:
            left  = np.roll(state, 1)
            right = np.roll(state, -1)
        else:
            left  = np.concatenate([[0], state[:-1]])
            right = np.concatenate([state[1:], [0]])
        neighborhood = (left << 2) | (state << 1) | right
        state = rule[neighborhood]
        rows.append(state.copy())
    return np.array(rows)  # (steps, width)


# -----------------------------
# 2) Sound rendering (pure tones)
# -----------------------------
def sine_segment(freq, dur, sr=44100, amp=0.10):
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)

    # 10ms fade in/out to avoid clicks
    env = np.ones_like(t)
    a = int(0.01 * sr)
    r = int(0.01 * sr)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if r > 0:
        env[-r:] = np.linspace(1, 0, r)

    return amp * env * np.sin(2 * np.pi * freq * t)

def render_sequence(freqs, seg_dur=0.05, sr=44100, amp=0.10):
    audio = np.concatenate([sine_segment(f, seg_dur, sr, amp) for f in freqs])
    return np.clip(audio, -0.9, 0.9)

def ca_to_freqs(rows, fmin=180, fmax=1200):
    width = rows.shape[1]
    density = rows.sum(axis=1) / width  # 0..1
    return fmin + density * (fmax - fmin)


# -----------------------------
# 3) Frame rendering (16:9, white bg, black cells)
# -----------------------------
def make_frames(rows, out_size=(1920, 1080), margin=60):
    """
    rows: (steps, width) uint8
    returns list of PIL Images, one per time step,
    showing columns 0..t growing left->right.
    """
    steps, width = rows.shape
    H, Wpix = out_size[1], out_size[0]

    # We'll display a matrix of shape (space, time) = (width, steps)
    spacetime = rows.T  # now shape (width(space), steps(time))

    # Determine cell size so height fits nicely
    usable_h = H - 2 * margin
    usable_w = Wpix - 2 * margin

    cell_sz = min(usable_h // width, usable_w // steps)
    cell_sz = max(1, int(cell_sz))

    img_h = width * cell_sz
    img_w = steps * cell_sz

    frames = []
    for t in range(steps):
        # reveal only up to column t
        mat = spacetime[:, :t+1]  # (width, t+1)

        # scale up to pixels (crisp nearest-neighbor)
        small = (1 - mat) * 255  # invert so 1-cells become black later
        small = small.astype(np.uint8)
        im_small = Image.fromarray(small, mode="L")
        im_scaled = im_small.resize(( (t+1)*cell_sz, width*cell_sz ), resample=Image.NEAREST)

        # convert to RGB and threshold to pure black/white
        im_scaled = im_scaled.point(lambda p: 0 if p < 128 else 255).convert("RGB")

        # white canvas
        canvas = Image.new("RGB", out_size, (255, 255, 255))

        # paste starting at left margin, vertically centered
        x0 = margin
        y0 = (H - im_scaled.size[1]) // 2
        canvas.paste(im_scaled, (x0, y0))

        frames.append(canvas)
    return frames


# -----------------------------
# 4) Assemble video with duplicated frames per tone
# -----------------------------
def frames_for_audio(frames, seg_dur, fps):
    """
    Duplicate each still so it lasts seg_dur seconds at fps.
    """
    k = int(round(seg_dur * fps))
    k = max(1, k)
    expanded = []
    for fr in frames:
        expanded.extend([fr] * k)
    return expanded


# -----------------------------
# 5) Full pipeline for one rule
# -----------------------------
def make_ca_av(rule_num=110,
               width=128,
               seg_dur=0.05,
               total_seconds=30,
               fps=30,
               out_base="ca_rule110",
               out_size=(1920, 1080)):
    steps = int(total_seconds / seg_dur)

    # CA
    rows = wolfram_ca(rule_num, width=width, steps=steps, wrap=True)

    # Audio
    freqs = ca_to_freqs(rows)
    audio = render_sequence(freqs, seg_dur=seg_dur)
    sr = 44100
    wav_path = f"{out_base}.wav"
    write(wav_path, sr, (audio * 32767).astype(np.int16))

    # Frames
    stills = make_frames(rows, out_size=out_size)
    vid_frames = frames_for_audio(stills, seg_dur=seg_dur, fps=fps)

    # Convert PIL -> numpy for moviepy
    vid_frames_np = [np.array(im) for im in vid_frames]

    # Video clip
    clip = ImageSequenceClip(vid_frames_np, fps=fps).with_audio(AudioFileClip(wav_path))

    mp4_path = f"{out_base}.mp4"
    clip.write_videofile(mp4_path, codec="libx264", audio_codec="aac")
    print("Wrote:", mp4_path, wav_path)


if __name__ == "__main__":
    make_ca_av(rule_num=110, out_base="ca_rule110")
