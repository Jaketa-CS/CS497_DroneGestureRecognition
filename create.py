import cv2 as cv
import os
import random
def augment_images(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for dir in dirs:
            subdir = os.path.join(output_dir, os.path.relpath(os.path.join(root, dir), input_dir))
            if not os.path.exists(subdir):
                os.makedirs(subdir)

        # Loop over all the image files in the current subdirectory
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                img = cv.imread(os.path.join(root, file))
                output_subdir = os.path.join(output_dir, os.path.relpath(root, input_dir))

                for i in range(4):
                    ran =  random.randint(0,1000)
                    #ran1 = random.randint(0,100)
                    name = str(ran) #+ str(ran1)
                    if i ==0:
                        print("c: ",i)
                        bw_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                        output_file = os.path.join(output_subdir,  "gray" +name+file)
                        cv.imwrite(output_file, bw_img)
                      #  del img, bw_img
                    if i ==1:
                        for e in range(5):
                            m = e*20
                            print("c: ",e)
                            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                            ret, thresh = cv.threshold(gray, 0 ,255-m,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
                            output_file = os.path.join(output_subdir,  "edge"+str(e) +name+file)
                            cv.imwrite(output_file, thresh)
                        # del img, thresh
                    if i ==2:
                        for e in range(0,20,2):
                            print("c: ",e)
                            blur = cv.GaussianBlur(img, (7,7), e)
                            output_file = os.path.join(output_subdir,  "blur"+str(e) +name+file)
                            cv.imwrite(output_file, blur)
                        # del img, thresh

                    if i ==3:
                        print("c: ",i)
                        for j in range(70):
                            print(j)
                            alpha = random.uniform(1.00, 3.00)
                            beta =  round(random.uniform(0, 100))
                            print("alpha: ", alpha)
                            print("beta : ", beta) 
                            bright = cv.convertScaleAbs(img, alpha = alpha, beta = beta)
                            output_file = os.path.join(output_subdir, "bright_"+str(j) + name+file)
                            cv.imwrite(output_file, bright)
                    #if i ==4:
                    #if i ==5:
                    #if i ==6:
                    #if i ==7:
                                

                #rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

                # Save the rotated image to the output directory with the same subdirectory structure
                #output_subdir = os.path.join(output_dir, os.path.relpath(root, input_dir))
                #output_file = os.path.join(output_subdir, file)
                #cv2.imwrite(output_file, rotated_img)

                # Release memory
                #del img, rotated_img


            
            #save the rotated image and rotate the image by 90 degrees
            """
            rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            rotated_img_path = os.path.join(output_dir, path+"rotated2_" + file)
            cv2.imwrite(rotated_img_path, rotated_img)

            rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            rotated_img_path = os.path.join(output_dir, path+"rotated3_" + file)
            cv2.imwrite(rotated_img_path, rotated_img)
            
            bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bw_img_path = os.path.join(output_dir, path+"bw_" + file)
            cv2.imwrite(rotated_img_path, bw_img)
"""

    
input_dir = "/home/madjack/data"
output_dir = "/home/madjack/mygestures"
augment_images(input_dir, output_dir)
