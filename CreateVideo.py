"""
Video creating utility

Created on Sun Jul 23 11:17:09 2022

@author: mukul
"""
import glob
import os

import cv2
import imageio

from GlobalValues import RealTimeIncoming_results, videos

if not os.path.exists(videos):
    os.mkdir(videos)

def create_video(img_src, out_put, fps):
    image_list = []
    image_list2 = []
    image_list3 = []
    print(img_src)
    dir = {}
    print('\nCompiling image list...', end='', flush=True)
    count = 0
    for img in sorted(glob.glob(img_src + '/*.png'),reverse=True)[::-1]:
        # print(img)
        # coverage = float((img.split('/')[1]).split('_')[0])


        try:
            im = imageio.imread(img)
            dir[im.size] = dir.get(im.size, -1) + 1
        
            # im = cv2.resize(im, (912, 912)) #manually resize if needed
            if (im.size == 36941080):
                image_list.append(im)
            # if coverage >0.1:
            #     image_list.append(im)
            # elif coverage >0.01:
            #     image_list2.append(im)
            # else:
            #     image_list3.append(im)
        except:
            count += 1
            print(img)
            continue
    print(dir, count)
    print('Creating mp4 video...', end='', flush=True)
    os.makedirs(videos, exist_ok=True)
    
    imageio.mimwrite(videos + '/' + out_put, image_list, fps=fps)
    # imageio.mimwrite(videos + '/2' + out_put, image_list2, fps=fps)
    # imageio.mimwrite(videos + '/3' + out_put, image_list3, fps=fps)
    print('Done.\n')
    print(videos + '/' + out_put)
# DataRepository/RealTimeIncoming_results/ParkFire_burnt/results
# DataRepository/RealTimeIncoming_results/ParkFire/results
# DataRepository/RealTimeIncoming_results/GalenaFire/results
site = 'DavisCreekFire'
GOES_OR_RESULT = 'results_zoom'
create_video(img_src=f'DataRepository/RealTimeIncoming_results/DavisCreekFire/results'
             , out_put=f'{site}_{GOES_OR_RESULT}.mp4'
             , fps=4)

# create_video(img_src=f'DataRepository/reference_data_sNPP_REQ/compare_all_SNPP_REQ'
#              , out_put=f'VIIRS_SNPP_Req.mp4'
#              , fps=4)

# create_video(img_src=f'DataRepository/reference_data_NOAA/compare_all_NOAA'
#              , out_put=f'VIIRS_NOAA_Req.mp4'
#              , fps=4)


# create_video(img_src='results/goodsamples'
#              , out_put='good_sample_predction.mp4'
#              , fps=4)

# create_video(img_src='results/avgsamples'
#              , out_put='avg_sample_predction.mp4'
#              , fps=4)

# create_video(img_src='results/badsamples'
#              , out_put='bad_sample_predction.mp4'
#              , fps=4)

#
