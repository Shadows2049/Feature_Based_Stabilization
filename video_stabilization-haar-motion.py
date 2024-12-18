import numpy as np
import cv2
import matplotlib.pyplot as plt

def movingAverage(curve, radius): 
    window_size = 2 * radius + 1
    # Define the filter 
    f = np.ones(window_size) / window_size 
    # Add padding to the boundaries 
    curve_pad = np.pad(curve, (radius, radius), 'edge') 
    # Apply convolution 
    curve_smoothed = np.convolve(curve_pad, f, mode='same') 
    curve_smoothed = curve_smoothed[radius:-radius]

    return curve_smoothed 

def smooth(trajectory): 
    smoothed_trajectory = np.copy(trajectory) 
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory

def fixBorder(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

def calculate_mse(original, stabilized):
    return np.mean((original - stabilized) ** 2)

def calculate_psnr(mse):
    return 10 * np.log10(255 ** 2 / mse)

def plot_trajectories(original_trajectory, smoothed_trajectory):
    plt.figure()
    plt.plot(original_trajectory[:, 0], label='Original Trajectory')
    plt.plot(smoothed_trajectory[:, 0], label='Smoothed Trajectory')
    plt.xlabel('Frame')
    plt.ylabel('X Translation')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(original_trajectory[:, 1], label='Original Trajectory')
    plt.plot(smoothed_trajectory[:, 1], label='Smoothed Trajectory')
    plt.xlabel('Frame')
    plt.ylabel('Y Translation')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(original_trajectory[:, 2], label='Original Trajectory')
    plt.plot(smoothed_trajectory[:, 2], label='Smoothed Trajectory')
    plt.xlabel('Frame')
    plt.ylabel('Rotation Angle')
    plt.legend()
    plt.show()

def plot_motion_vectors(prev_pts, curr_pts):
    plt.figure()
    for i in range(len(prev_pts)):
        if prev_pts[i].shape[0] == 2 and curr_pts[i].shape[0] == 2:
            plt.arrow(prev_pts[i, 0], prev_pts[i, 1], curr_pts[i, 0] - prev_pts[i, 0], curr_pts[i, 1] - prev_pts[i, 1], head_width=5, head_length=5, fc='r', ec='r')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Motion Vectors')
    plt.show()

# Optimal 50
SMOOTHING_RADIUS = 50 

# Load the Haar Cascade Classifier for detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read input video
cap = cv2.VideoCapture('video.mp4') 
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('video_out.avi', fourcc, fps, (2 * w, h))

# Read first frame
_, prev = cap.read() 
# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) 
transforms = np.zeros((n_frames - 1, 3), np.float32) 

for i in range(n_frames - 2):
    # Detect faces in the previous frame
    faces = face_cascade.detectMultiScale(prev_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no faces are detected, use goodFeaturesToTrack
    if len(faces) == 0:
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=200,
                                           qualityLevel=0.01,
                                           minDistance=30,
                                           blockSize=3)
    else:
        # Use the center of the detected face as the feature point
        x, y, w, h = faces[0]
        prev_pts = np.array([[x + w / 2, y + h / 2]], dtype=np.float32).reshape(-1, 1, 2)
   
    # Read next frame
    success, curr = cap.read() 
    if not success: 
        break 

    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 

    # track feature points
    if prev_pts is not None and len(prev_pts) > 0:
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        if prev_pts.shape != curr_pts.shape: 
            print(f"Shape mismatch: prev_pts shape: {prev_pts.shape}, curr_pts shape: {curr_pts.shape}")
         # Sanity check
        assert prev_pts.shape == curr_pts.shape

        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts) 
        
        if m is not None:
            dx = m[0, 2] 
            dy = m[1, 2] 
            # Extract rotation angle 
            da = np.arctan2(m[1, 0], m[0, 0]) 
            # Store transformation
            transforms[i] = [dx, dy, da]

        prev_gray = curr_gray
        print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))
    else:
        print("No valid points found in prev_pts.")



# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0) 
smoothed_trajectory = smooth(trajectory) 
difference = smoothed_trajectory - trajectory
transforms_smooth = transforms + difference

plot_trajectories(trajectory, smoothed_trajectory)

# Reset stream to first frame 
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 

# Write n_frames-1 transformed frames
mse_values = []
psnr_values = []

for i in range(n_frames - 2):
    success, frame = cap.read() 
    if not success:
        break

    # Extract transformations from the new transformation array
    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]

    # Reconstruct transformation matrix accordingly to new values
    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy

    # Apply affine wrapping to the given frame
    frame_stabilized = cv2.warpAffine(frame, m, (w, h))

    # Fix border artifacts
    frame_stabilized = fixBorder(frame_stabilized) 
    frame_stabilized = cv2.resize(frame_stabilized, (frame.shape[1], frame.shape[0]))
    # Calculate MSE and PSNR
    mse = calculate_mse(frame, frame_stabilized)
    psnr = calculate_psnr(mse)
    mse_values.append(mse)
    psnr_values.append(psnr)

    # Write the frame to the file
    frame_out = cv2.hconcat([frame, frame_stabilized])

    # If the image is too big, resize it.
    if frame_out.shape[1] > 1920: 
        frame_out = cv2.resize(frame_out, (frame_out.shape[1] // 2, frame_out.shape[0] // 2))
  
    cv2.imshow("Before and After", frame_out)
    cv2.waitKey(10)
    out.write(frame_out)

# Release video
cap.release()
out.release()
# Close windows
cv2.destroyAllWindows()

# Plot MSE and PSNR values
plt.figure()
plt.plot(mse_values, label='MSE')
plt.xlabel('Frame')
plt.ylabel('MSE')
plt.legend()
plt.show()

plt.figure()
plt.plot(psnr_values, label='PSNR')
plt.xlabel('Frame')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.show()