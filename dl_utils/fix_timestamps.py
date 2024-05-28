import os
import sys
import glob
import time

""" 06/02/2020 data collection:
# 05:01:12.213381700
# 18000000000000 + 60000000000 + 12000000000 + 213381700 = 18072213381700

## calculate actual blackfly timestamp from modified time:
# calculated = os.stat(file).st_mtime_ns - 18072213381700

## calculate actual lidar timestamp from filename:
# calculated = 1589137899623081000 + int(os.path.splitext(os.path.basename(file))[0].split('_')[-1]) * 1000000
"""

""" 09/11/2020 data collection:
## calculate blackfly timestamp in seconds from filename:
# calculated = (time.mktime(time.strptime(os.path.splitext(os.path.basename(file))[0].split('-')[-2], "%m%d%Y%H%M%S")) * 1000 - 43230) * 1e6

## calculate blackfly timestamp from modified time:
# calculated = os.stat(file).st_mtime_ns - (43230 * 1e6)

## calculate actual ouster timestamp from filename:
# calculated = (1597727834112 + int(os.path.splitext(os.path.basename(file))[0].split('_')[-1])) * 1e6

## calculate actual velodyne timestamp from filename:
# calculated = int(os.path.splitext(os.path.basename(file))[0].split('_')[-1]) * 1e6
"""

if len(sys.argv) != 2:
    print('Usage: python {} scans_folder'.format(sys.argv[0]))
    sys.exit(1)

files = sorted(glob.glob(os.path.join(sys.argv[1], '*.pcd')))

# file = os.path.join(sys.argv[1], 'os1_2123216168.pcd')
# if True:
for file in files:
    actual = os.stat(file).st_mtime_ns
    # calculated = int(os.path.splitext(os.path.basename(file))[0].split('_')[-1]) * 1e6
    # difference = abs(actual - calculated) / 1000000.0
    # print(actual)
    # print(calculated)
    # print(difference)
    # if difference > 90:
    #     print(difference, 'ms')
    # os.utime(file, ns=(calculated, calculated))
    # sys.stdout.write('.')
    # sys.stdout.flush()
    print('{},{}'.format(os.path.basename(file), round(actual / 1000000.0)))
# print()
