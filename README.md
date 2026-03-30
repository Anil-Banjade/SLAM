In comparisoin to our main branch this branch does not support nerf currently but is a reabased version of main to mitigate some errors and experiment with various parameters and combinations.

useful flags:
--source path to video or images
--use_images True # if you are using image frames rather than video
--optimize True# To enable bundle adjustment, consumes lot of time (recommended except in scenes that have really good features and dont overlap very much)
-- fps_divider 1 # HIghly recommended, divides current fps by itself resulting in fps of 1 frame per second, ensures parallax if scene is continous
--show_tests True # to enable verbose mode on logs and also enable sdl2 viewer
--display {pose/map_points/all} options to view one of three, sometimes open3d gives very bad scaling so removing one of parameter helps to visualize the output

![Library building of Thapathali campus: camera takes straight line and returns back to original spot local BA barely makes a diff since pnp does it's work pretty well (drift is present, requires graph optimization or global BA( graph optimization should do it](video_frames/images/library.png)
![local bA results when going around a desk, 2 outlier poses at the end, again loop closure should solve it) note: when local BA is off results are very bad](images/desk.png)
