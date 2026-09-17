import glob, scipy.io as sio, cv2

root = "data/extracted/UTD-MHAD"
for mod in ["Inertial", "Skeleton", "Depth"]:
    f = sorted(glob.glob(f"{root}/{mod}/**/*.mat", recursive=True))[0]
    m = sio.loadmat(f)
    print(mod, f.split("/")[-1],
          {k: v.shape for k, v in m.items() if not k.startswith("__")})

v = sorted(glob.glob(f"{root}/RGB/**/*.avi", recursive=True))[0]
cap = cv2.VideoCapture(v)
print("RGB", v.split("/")[-1],
      "frames", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
      "fps", cap.get(cv2.CAP_PROP_FPS),
      "size", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
