import imageio
import glob
image_list = []
img_src = "plots"
videos="videos"
print('\nCompiling image list...', end='', flush=True)
for img in sorted(glob.glob(img_src + '/*.png')):
    im = imageio.imread(img)
    # im = cv2.resize(im, (900, 900)) #manually resize if needed
    image_list.append(im)

print('Creating mp4 video...', end='', flush=True)
imageio.mimwrite(videos + '/frp.mp4', image_list, fps=2)
print('Done.\n')