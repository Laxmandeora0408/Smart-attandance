import cv2
import numpy as npy
import face_recognition as face_rec
# function
def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)


# img declaration
laxman_picture = face_rec.load_image_file('sample_images\laxman_picture.jpg')
laxman_picture = cv2.cvtColor(laxman_picture, cv2.COLOR_BGR2RGB)
laxman_picture = resize(laxman_picture, 0.50)
harshita_picture = face_rec.load_image_file('sample_images\harshita_picture.jpg')
harshita_picture = resize(harshita_picture, 0.50)
harshita_picture = cv2.cvtColor(harshita_picture, cv2.COLOR_BGR2RGB)

# finding face location

faceLocation_laxman_picture = face_rec.face_locations(laxman_picture)[0]
encode_laxman_picture = face_rec.face_encodings(laxman_picture)[0]
cv2.rectangle(laxman_picture, (faceLocation_laxman_picture[3], faceLocation_laxman_picture[0]), (faceLocation_laxman_picture[1], faceLocation_laxman_picture[2]), (255, 0, 255), 3)


faceLocation_harshita_picture = face_rec.face_locations(harshita_picture)[0]
encode_harshita_picture = face_rec.face_encodings(harshita_picture)[0]
cv2.rectangle(harshita_picture, (harshita_picture[3], harshita_picture[0]), (harshita_picture[1], harshita_picture[2]), (255, 0, 255), 3)

results = face_rec.compare_faces([encode_laxman_picture], encode_harshita_picture)
print(results)
cv2.putText(harshita_picture, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2 )

cv2.imshow('main_img', laxman_picture)
cv2.imshow('test_img', harshita_picture)
cv2.waitKey(0)
cv2.destroyAllWindows()