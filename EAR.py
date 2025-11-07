from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    # 수직 눈 좌표
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # 수평 눈 좌표
    C = dist.euclidean(eye[0], eye[3])
    
    # 종횡비 계산
    ear = (A + B) / (2.0 * C)
    return ear