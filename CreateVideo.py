import imageio
import glob
image_list = []
img_src = "RunTimeIncoming_results"
videos="Videos"
print('\nCompiling image list...', end='', flush=True)
for img in sorted(glob.glob(img_src + '/*.png')):
    im = imageio.imread(img)
    # print(im.size)
    # im = cv2.resize(im, (900, 900)) #manually resize if needed
    # if(im.size==3571232):
    image_list.append(im)

print('Creating mp4 video...', end='', flush=True)
imageio.mimwrite(videos + '/GOES_superResolution.mp4', image_list, fps=3)
print('Done.\n')