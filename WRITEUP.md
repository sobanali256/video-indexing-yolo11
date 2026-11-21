# Write-up

## 1. Your timings (100 words)
How long did your first working version take on the full 50 files, and how long did your final version take? Which single change saved the most time?
My initial version was ran on 3 Parquet files (train-0000, train-0024, train-0049) to project the overall runtime. 3 files took over ~3.5hrs and was projected that all 50 files if ran using brute force will take ~33hrs that is more than a day given there are no errors or session issues during the run. The final version on all 50 files took ~2hrs. Sampling that is sending few evenly spaced frames from each second to the model instead of every frame provided the largest speedup which was measured to be 7.9x.


## 2. Your pipeline (150 words)
Describe what your script does, step by step, from opening a Parquet file to writing an output line.
At the very start, the model is loaded once and the GPU is set to half precision (FP16). After this the parquet files are read from the input folder where each row is one video and one row is read at a time. Anything broken/corrupt is recorded as an error line. The video bytes are opened in memory and checked for rotation. After this, every frame is decoded and sampled keeping only 3 frames while rest are discarded. The selected frames are then prepared for YOLO converting their color, rotating if required and matching their resolution. The GPU receives the frames in batch of 32 and we received 5 numbers as output which tell us about how many of each object was present. Once the video is completed its result is written and marked as done so in case of a crash we know where to start.


## 3. Where the time goes (150 words)
Which part of your final version takes the most time? How did you measure this?
In the final version, majority of the time is consumed on GPU inference. This was measured by using the built in timers (run.py, enabled with --stats-json). The timing points recorded were every GPU batch call, each decode thread where only time spend decoding and preparing frames was counted. Time for the startup and the entire loop was also recorded. To further strengthen the credibility of the measured time bench.py checked the machine every 30 seconds to see if there is any download running or any other processor using the GPU which may affect our results. If any of these was found it would mark the run as clean = no along with the reason. This provided us with evidence whether our measured time could be trusted and accurate or not.


## 4. Checking correctness (100 words)
How did you check that your results stayed correct after you made the script faster?
The very first brute force run (train-0000, train-0024, train-0049) was done to create a baseline with which we could compared our speedup version. Score.py uses the grading rules and compares every run against the reference calculating the F1 / people exact / people within +-1. The only limitation is that the brute force on all 50 files were not done so the first 3 parquet were used as reference and scoring.


## 5. Working with AI (100 words)
Describe one suggestion from an AI tool that was wrong or that you decided not to use. How did you find out it was wrong, or why did you reject it?
AI had suggested of using adaptive sampling, starting from a few frames and increasing the number to look at more if the frames disagreed. However this idea was not implemented and rejected. Both fixed and adaptive strategy was tested and the evidence stated that adaptive was indeed more accurate but provided lower speedup, using fixed frame number (K=3) provided the maximum speedup but had lower accuracy (F1 / people exact / people within +-1) however it was well within the acceptable range. So fixed frame was selected over the adaptive sampling prioritizing speedup.


## 6. Next steps (100 words)
If you had one more week, what would you change, and why?
The every first step would be to run the brute force on all 50 files to create a larger baseline and then test the current speedup version and test how it hold. Besides this, implement TensorRT as currently all the time is being consumed in GPU inference and TensorRT would have helped us reduced that giving us expected speedup of ~1.5-3x.

