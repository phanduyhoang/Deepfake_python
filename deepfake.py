import cv2
import numpy as np
import mediapipe as mp

# -------------------- MediaPipe Setup --------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ---------------------------------------------------------
# 1) Helpers to detect landmarks and extract face points
# ---------------------------------------------------------
def mediapipe_detection(image, model):
    """
    Convert image color from BGR to RGB for MediaPipe,
    then run the model for detection, and finally
    convert back to BGR.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr, results

def extract_face_landmarks(results, image_shape):
    """
    Extract face landmarks in pixel coordinates (x,y)
    from MediaPipe Holistic results.
    Returns a list of (x, y) tuples in integer pixel coordinates.
    """
    landmark_list = []
    if results.face_landmarks:
        h, w = image_shape[:2]
        for lm in results.face_landmarks.landmark:
            px = int(lm.x * w)
            py = int(lm.y * h)
            landmark_list.append((px, py))
    return landmark_list

def clamp_points(points, width, height):
    """
    Ensure x and y coordinates are within [0, width-1] and [0, height-1].
    This prevents OpenCV subdivision errors when points go out of bounds.
    """
    clamped = []
    for (x, y) in points:
        # Clamp the coordinates
        x_c = max(0, min(x, width - 1))
        y_c = max(0, min(y, height - 1))
        clamped.append((x_c, y_c))
    return clamped

# ---------------------------------------------------------
# 2) Delaunay Triangulation & Warping
# ---------------------------------------------------------
def rect_contains(rect, point):
    """
    Check if a point is inside a rectangle.
    """
    x, y, w, h = rect
    px, py = point
    if px < x or px > x + w or py < y or py > y + h:
        return False
    return True

def calculate_delaunay_triangles(rect, points):
    """
    Calculate Delaunay triangles for a set of points within a rect.
    Returns a list of triangle indices: (p1, p2, p3).
    """
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        # Insert each point into the subdiv
        subdiv.insert(p)
    
    triangle_list = subdiv.getTriangleList()
    delaunay_triangles = []
    pt_dict = { (p[0], p[1]): i for i, p in enumerate(points) }
    
    for t in triangle_list:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        
        if pt1 in pt_dict and pt2 in pt_dict and pt3 in pt_dict:
            idx1 = pt_dict[pt1]
            idx2 = pt_dict[pt2]
            idx3 = pt_dict[pt3]
            delaunay_triangles.append((idx1, idx2, idx3))
    return delaunay_triangles

def apply_affine_transform(src, src_tri, dst_tri, size):
    """
    Applies an affine transform to warp the triangle from src to match the shape in dst.
    """
    # Calculate the transformation matrix
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    # Apply the transformation
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst

def warp_triangle(img1, img2, t1, t2):
    """
    Warp the triangular region t1 in img1 to t2 in img2 and seamlessly blend.
    """
    # Find bounding rectangles for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    
    # Offset points by the top-left corner of the respective bounding rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []
    for i in range(3):
        t1_rect.append((t1[i][0] - r1[0], t1[i][1] - r1[1]))
        t2_rect.append((t2[i][0] - r2[0], t2[i][1] - r2[1]))
        t2_rect_int.append((t2[i][0] - r2[0], t2[i][1] - r2[1]))
    
    # Extract patches
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    
    # Warp the triangle
    img1_crop = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    warped_triangle = apply_affine_transform(
        img1_crop,
        t1_rect,
        t2_rect,
        (r2[2], r2[3])
    )
    
    # Create mask for triangle
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)
    
    # Convert image to float for seamless blending
    img2_crop = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    img2_crop = img2_crop * (1 - mask)
    img2_crop += warped_triangle * mask
    
    # Put the patched region back into the image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2_crop


# ---------------------------------------------------------
# 3) Main "DeepFake" Replacement Pipeline
# ---------------------------------------------------------
# ---------------------------------------------------------
# 3) Main "DeepFake" Replacement Pipeline
# ---------------------------------------------------------
def main():
    # ----------------------- Load Source Face -----------------------
    source_img_path = 'joe.jpg'  # <-- Replace with your path
    source_img = cv2.imread(source_img_path)
    if source_img is None:
        print(f"Error: Could not load '{source_img_path}'. Please check the path.")
        return
    
    # ----------------------- Extract Source Face Landmarks -----------------------
    with mp_holistic.Holistic(min_detection_confidence=0.5, 
                              min_tracking_confidence=0.5) as holistic:
        _, source_results = mediapipe_detection(source_img, holistic)
    
    source_face_points = extract_face_landmarks(source_results, source_img.shape)
    
    # We need at least a few face points to proceed
    if len(source_face_points) < 3:
        print("Not enough face landmarks detected in source image!")
        return
    
    # Prepare for Delaunay triangulation
    h_s, w_s = source_img.shape[:2]
    rect_s = (0, 0, w_s, h_s)
    
    # Triangulate the source face
    source_face_points = clamp_points(source_face_points, w_s, h_s)
    source_triangles = calculate_delaunay_triangles(rect_s, source_face_points)
    
    # ----------------------- Initialize Webcam -----------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Video Writers
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    
    # Writer for original video
    original_out = cv2.VideoWriter('original_output.avi', fourcc, 20.0, (frame_width, frame_height))
    
    # Writer for masked video
    masked_out = cv2.VideoWriter('masked_output.avi', fourcc, 20.0, (frame_width, frame_height))
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, 
                              min_tracking_confidence=0.5) as holistic:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam. Exiting...")
                break
            
            # Save the original frame
            original_out.write(frame)
            
            # 1) Detect your (live) face landmarks
            _, live_results = mediapipe_detection(frame, holistic)
            live_face_points = extract_face_landmarks(live_results, frame.shape)
            
            # 2) Clamp points to avoid out-of-range insertion in Subdiv2D
            h_l, w_l = frame.shape[:2]
            live_face_points = clamp_points(live_face_points, w_l, h_l)
            
            if len(live_face_points) < 3:
                # If no face or too few landmarks, just show the original frame
                cv2.imshow("DeepFake Demo", frame)
                masked_out.write(frame)  # Save the unprocessed frame
            else:
                # 3) Triangulate on your face
                rect_l = (0, 0, w_l, h_l)
                
                # Recalculate the Delaunay each frame (simple but not optimal)
                live_triangles = calculate_delaunay_triangles(rect_l, live_face_points)
                
                # 4) Create a copy of the webcam frame to place the warped face
                output_frame = frame.copy()
                
                # 5) Warp each triangle from source -> live
                for tri_idx in source_triangles:
                    # Source triangle (3 points)
                    t1 = [ source_face_points[tri_idx[0]],
                           source_face_points[tri_idx[1]],
                           source_face_points[tri_idx[2]] ]
                    
                    # Live triangle (3 points) at the same indices
                    t2 = [ live_face_points[tri_idx[0]],
                           live_face_points[tri_idx[1]],
                           live_face_points[tri_idx[2]] ]
                    
                    # Warp
                    warp_triangle(source_img, output_frame, t1, t2)
                
                # 6) Show the result
                cv2.imshow("DeepFake Demo", output_frame)
                masked_out.write(output_frame)  # Save the processed frame
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    original_out.release()
    masked_out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
