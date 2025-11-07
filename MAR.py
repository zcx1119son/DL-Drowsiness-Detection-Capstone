from scipy.spatial import distance as dist

def mouth_aspect_ratio(mouth):
    # 수직 입의 좌표
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])  # 53, 57

    # 수평 입의 좌표
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55

    # 입 비율 계산
    mar = (A + B) / (2.0 * C)
    return mar