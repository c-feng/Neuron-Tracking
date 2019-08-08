
def img_crop(img, coord, size = [80, 80, 80]):
    img_s = img.shape
    size_ = np.array(size)
    l_s = size_ // 2
    r_s = size_ - l_s
    x, y, z = coord
    
    x_l = x - l_s[0] if x - l_s[0] >= 0 else 0
    y_l = y - l_s[1] if y - l_s[1] >= 0 else 0
    z_l = z - l_s[2] if z - l_s[2] >= 0 else 0

    x_p_l = l_s[0] - (x - x_l)
    y_p_l = l_s[1] - (y - y_l)
    z_p_l = l_s[2] - (z - z_l)


    x_r = x + r_s[0] if x + r_s[0] <= img_s[0] else img_s[0]
    y_r = y + r_s[1] if y + r_s[1] <= img_s[1] else img_s[1]
    z_r = z + r_s[2] if z + r_s[2] <= img_s[2] else img_s[2]

    x_p_r = r_s[0] - (x_r - x)
    y_p_r = r_s[1] - (y_r - y)
    z_p_r = r_s[2] - (z_r - z)

    pad_length = np.array([[x_p_l, x_p_r], [y_p_l, y_p_r], [z_p_l, z_p_r]])
    
    #print(pad_length, coord) 
    img = pad(img, pad_length, mode = "constant")
    
     
    img_c = img[x_l:x_r + x_p_l + x_p_r, y_l:y_r + y_p_l + y_p_r, z_l:z_r + z_p_l + z_p_r]


    return img_c

