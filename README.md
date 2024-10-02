
To work with rosbag files in images format instead
1. Run rosbag extraction
2. Then rectify aux images
3. Sync rosfiles to associate frames correctly

Then the dataset is ready... for further processing

Prepare the scene for 3DGS by running colmap.py

At this point I could run a traditional 3DGS by pointing it to the folder.


If I want to use nerfstudio's splatfactor implementation I also have to convert the colmap .bin files to .txt and then run colmap2nerf.py to make the dataset folder compatible with splatgfacto
