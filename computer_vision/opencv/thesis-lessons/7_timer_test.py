import time

# Simulates the thesis config values
CONSISTENCY_FRAMES = 3
consistency_count = 0
active_event = False

# A simulated stream of 6 incoming video frames
# True means the pixel area changed; False means it is normal
simulated_video_stream = [True, True, False, True, True, True]

print("="*50)
print("🚀 STARTING TEMPORAL STATE MACHINE SIMULATION")
print("="*50)

# loops the frames while keeping track of the index
for frame_idx, changed in enumerate(simulated_video_stream, start = 1):
    time.sleep(1.0) # simulate frame delay

    # Execute thesis frame logic
    if changed:
        consistency_count += 1
    else:
        consistency_count = 0
        active_event = False
    
    print(f"Frame: {frame_idx} Changed={changed} | Buffer count={consistency_count}")

    # check event trigger

    if consistency_count >= CONSISTENCY_FRAMES and not active_event:
        active_event = True
        print(f"🚨 [ALERT] Zone reached stability limit! Posting event to API database.")

print("="*50)