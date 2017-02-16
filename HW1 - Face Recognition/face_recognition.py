# Julian Velez
# 2/15/2017
# an ai program for recognizing faces by comparing seven key features:

import numpy as np
import math

# reads data into a 3d list (face_list[i][j][k])
# i -> the i+1 image
# j -> the j point in the image (starting from 0)
# k -> 0 is x and 1 is y
def init_data():
    face_list=[]
    
    for i in range(1,6):
        face_list.append(np.genfromtxt("m-00"+str(i)+"/m-00"+str(i)+"-01.pts", 
                            skip_header=3, skip_footer=1))
        face_list.append(np.genfromtxt("m-00"+str(i)+"/m-00"+str(i)+"-05.pts", 
                            skip_header=3, skip_footer=1))
    for i in range(1,6):
        face_list.append(np.genfromtxt("w-00"+str(i)+"/w-00"+str(i)+"-01.pts", 
                            skip_header=3, skip_footer=1))
        face_list.append(np.genfromtxt("w-00"+str(i)+"/w-00"+str(i)+"-05.pts", 
                            skip_header=3, skip_footer=1))

    return face_list

# compute the distance between two points
def find_dist(x1, y1, x2, y2):
    dist = math.sqrt((x2-x1)**2+(y2-y1)**2)

    return dist

# compute the eye length ratio of each image
# length of eye (maximum of two) over distance between points 8 and 13
def eye_length_ratio(face_list):
    list = face_list
    eye_length_ratios = []

    for i in range(0, len(list)):
        # find length of eye 1
        eye_1_x1 = list[i][9][0]
        eye_1_y1 = list[i][9][1]
        eye_1_x2 = list[i][10][0]
        eye_1_y2 = list[i][10][1]
        eye_1 = find_dist(eye_1_x1, eye_1_y1, eye_1_x2, eye_1_y2)

        # find length of eye 2
        eye_2_x1 = list[i][11][0]
        eye_2_y1 = list[i][11][1]
        eye_2_x2 = list[i][12][0]
        eye_2_y2 = list[i][12][1]
        eye_2 = find_dist(eye_2_x1, eye_2_y1, eye_2_x2, eye_2_y2)

        # find distance between points 8 and 13
        eight_x = list[i][8][0]
        eight_y = list[i][8][1]
        thirteen_x = list[i][13][0]
        thirteen_y = list[i][13][1]
        dist = find_dist(eight_x, eight_y, thirteen_x, thirteen_y)

        if eye_1 > eye_2:
            ratio = eye_1/dist
        else:
            ratio = eye_2/dist
        
        eye_length_ratios.append(ratio)
    
    return eye_length_ratios

# compute the eye distance ratio of each image
# distance between center of two eyes over distance between points 8 and 13
def eye_dist_ratio(face_list):
    list = face_list
    eye_dist_ratios = []

    for i in range(0, len(list)):
        left_eye_x = list[i][0][0]
        left_eye_y = list[i][0][1]
        right_eye_x = list[i][1][0]
        right_eye_y = list[i][1][1]
        eye_center_dist = find_dist(left_eye_x, left_eye_y, right_eye_x, right_eye_y)

        eight_x = list[i][8][0]
        eight_y = list[i][8][1]
        thirteen_x = list[i][13][0]
        thirteen_y = list[i][13][1]
        dist = find_dist(eight_x, eight_y, thirteen_x, thirteen_y)

        ratio = eye_center_dist/dist

        eye_dist_ratios.append(ratio)

    return eye_dist_ratios

# compute the nose ratio of each image
# distance between points 15 and 16 over distance between 20 and 21
def nose_ratio(face_list):
    list = face_list
    nose_ratios = []

    for i in range(0, len(list)):
        fifteen_x = list[i][15][0]
        fifteen_y = list[i][15][1]
        sixteen_x = list[i][16][0]
        sixteen_y = list[i][16][1]
        nostril_dist = find_dist(fifteen_x, fifteen_y, sixteen_x, sixteen_y)

        twenty_x = list[i][20][0]
        twenty_y = list[i][20][1]
        twentyone_x = list[i][21][0]
        twentyone_y = list[i][21][1]
        dist = find_dist(twenty_x, twenty_y, twentyone_x, twentyone_y)

        ratio = nostril_dist/dist

        nose_ratios.append(ratio)

    return nose_ratios

# compute the lip size ratio of each image
# distance between points 2 and 3 over distance between 17 and 18
def lip_size_ratio(face_list):
    list = face_list
    lip_size_ratios = []

    for i in range(0, len(list)):
        two_x = list[i][2][0]
        two_y = list[i][2][1]
        three_x = list[i][3][0]
        three_y = list[i][3][1]
        length = find_dist(two_x, two_y, three_x, three_y)

        seventeen_x = list[i][17][0]
        seventeen_y = list[i][17][1]
        eighteen_x = list[i][18][0]
        eighteen_y = list[i][18][1]
        size = find_dist(seventeen_x, seventeen_y, eighteen_x, eighteen_y)

        ratio = length/size

        lip_size_ratios.append(ratio)

    return lip_size_ratios

# compute the lip length ratio of each image
# distance between points 2 and 3 over distance between 20 and 21
def lip_length_ratio(face_list):
    list = face_list
    lip_length_ratios = []

    for i in range(0, len(list)):
        two_x = list[i][2][0]
        two_y = list[i][2][1]
        three_x = list[i][3][0]
        three_y = list[i][3][1]
        length = find_dist(two_x, two_y, three_x, three_y)

        twenty_x = list[i][20][0]
        twenty_y = list[i][20][1]
        twentyone_x = list[i][21][0]
        twentyone_y = list[i][21][1]
        dist = find_dist(twenty_x, twenty_y, twentyone_x, twentyone_y)

        ratio = length/dist

        lip_length_ratios.append(ratio)

    return lip_length_ratios

# compute the eye-brow length ratio of each image
# distance between points 4 and 5 (or distance between points 6 and 7 whichever is larger)
# over distance between 8 and 13
def brow_length_ratio(face_list):
    list = face_list
    brow_length_ratios = []

    for i in range(0, len(list)):
        four_x = list[i][4][0]
        four_y = list[i][4][1]
        five_x = list[i][5][0]
        five_y = list[i][5][1]
        left_brow = find_dist(four_x, four_y, five_x, five_y)

        six_x = list[i][6][0]
        six_y = list[i][6][1]
        seven_x = list[i][7][0]
        seven_y = list[i][7][1]
        right_brow = find_dist(six_x, six_y, seven_x, seven_y)

        eight_x = list[i][8][0]
        eight_y = list[i][8][1]
        thirteen_x = list[i][13][0]
        thirteen_y = list[i][13][1]
        dist = find_dist(eight_x, eight_y, thirteen_x, thirteen_y)

        if left_brow > right_brow:
            ratio = left_brow/dist
        else:
            ratio = right_brow/dist
        
        brow_length_ratios.append(ratio)

    return brow_length_ratios

# compute the aggressive ratio of each image
# distance between points 10 and 19 over distance between 20 and 21
def aggressive_ratio(face_list):
    list = face_list
    aggressive_ratios = []

    for i in range(0, len(list)):
        ten_x = list[i][10][0]
        ten_y = list[i][10][1]
        nineteen_x = list[i][19][0]
        nineteen_y = list[i][19][1]
        dist_1 = find_dist(ten_x, ten_y, nineteen_x, nineteen_y)

        twenty_x = list[i][20][0]
        twenty_y = list[i][20][1]
        twentyone_x = list[i][21][0]
        twentyone_y = list[i][21][1]
        dist_2 = find_dist(twenty_x, twenty_y, twentyone_x, twentyone_y)

        ratio = dist_1/dist_2

        aggressive_ratios.append(ratio)

    return aggressive_ratios

# print the feature values for each image
def print_features(ratio_1, ratio_2, ratio_3, ratio_4, ratio_5, ratio_6, ratio_7):
    i = 0
    while i < 10:
        if i % 2 == 0:
            print("m-00" + str((i+2)/2) + "-01 feature values:")
        else:
            print("m-00" + str((i+1)/2) + "-05 feature values:")
        print("Eye Length Ratio: " + str(ratio_1[i]))
        print("Eye Distance Ratio: " + str(ratio_2[i]))
        print("Nose Ratio: " + str(ratio_3[i]))
        print("Lip Size Ratio: " + str(ratio_4[i]))
        print("Lip Length Ratio: " + str(ratio_5[i]))
        print("Eye-Brow Length Ratio: " + str(ratio_6[i]))
        print("Aggressive Ratio: " + str(ratio_7[i]) + "\n")
        i = i + 1
    while i < 20:
        if i % 2 == 0:
            print("w-00" + str((i/2)-4) + "-01 feature values:")
        else:
            print("w-00" + str(((i-1)/2)-4) + "-05 feature values:")
        print("Eye Length Ratio: " + str(ratio_1[i]))
        print("Eye Distance Ratio: " + str(ratio_2[i]))
        print("Nose Ratio: " + str(ratio_3[i]))
        print("Lip Size Ratio: " + str(ratio_4[i]))
        print("Lip Length Ratio: " + str(ratio_5[i]))
        print("Eye-Brow Length Ratio: " + str(ratio_6[i]))
        print("Aggressive Ratio: " + str(ratio_7[i]) + "\n")
        i = i + 1

# calculate distances between each image 
# use feature values to find the closest match
def find_match(ratio_1, ratio_2, ratio_3, ratio_4, ratio_5, ratio_6, ratio_7):
    min_dist = float('inf')
    min_index = 0

    for i in range(1, len(ratio_1),2):
        for j in range(0, len(ratio_1),2):
            if i == j:
                continue
            else:
                dif_1 = ratio_1[i]-ratio_1[j]
                dif_2 = ratio_2[i]-ratio_2[j]
                dif_3 = ratio_3[i]-ratio_3[j]
                dif_4 = ratio_4[i]-ratio_4[j]
                dif_5 = ratio_5[i]-ratio_5[j]
                dif_6 = ratio_6[i]-ratio_6[j]
                dif_7 = ratio_7[i]-ratio_7[j]

                dist = math.sqrt(dif_1**2+dif_2**2+dif_3**2+dif_4**2+dif_5**2+dif_6**2+dif_7**2)

                if (dist < min_dist):
                    min_dist = dist
                    min_index = j
        if (i < 11):
            if (min_index < 10):
                print("Image m-00" + str((i+1)/2) + "-05 is closest to image m-00" + 
                        str((min_index+1)) + "-01")
            else:
                print("Image m-00" + str((i+1)/2) + "-05 is closest to image w-00" +
                        str(((min_index+1)/2)-4) + "-01")
        else:
            if (min_index < 10):
                print("Image w-00" + str(((i+1)/2)-5) + "-05 is closest to image m-00" + 
                        str((min_index+1)) + "-01")
            else:
                print("Image w-00" + str(((i+1)/2)-5) + "-05 is closest to image w-00" +
                        str(((min_index+1)/2)-4) + "-01")
        min_dist = float('inf')

# the main function
def main():
    # initialize data
    face_list = init_data() 
    
    # calculate the seven features for each image
    eye_length_ratios = eye_length_ratio(face_list)
    eye_dist_ratios = eye_dist_ratio(face_list)
    nose_ratios = nose_ratio(face_list)
    lip_size_ratios = lip_size_ratio(face_list)
    lip_length_ratios = lip_length_ratio(face_list)
    brow_length_ratios = brow_length_ratio(face_list)
    aggressive_ratios = aggressive_ratio(face_list)
    
    # print the features for each image
    print_features(eye_length_ratios, eye_dist_ratios, nose_ratios, lip_size_ratios, lip_length_ratios, 
                brow_length_ratios, aggressive_ratios)
    
    # find the matches for each image using the above features
    find_match(eye_length_ratios, eye_dist_ratios, nose_ratios, lip_size_ratios, lip_length_ratios, 
                brow_length_ratios, aggressive_ratios)
    print("\n")

main()