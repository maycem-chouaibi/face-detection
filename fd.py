# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:49:08 2021

@author: mayssem
"""

import face_recognition as fr
from PIL import Image, ImageDraw
import numpy as np

#Upload group image
image = fr.load_image_file('test2.jpg', 'RGB')

#upload each face
harry_image = fr.load_image_file("daniel.jpg")
harry_encoding = fr.face_encodings(harry_image)[0]

ron_image = fr.load_image_file("rupert.jpg")
ron_encoding = fr.face_encodings(ron_image)[0]

hermione_image = fr.load_image_file("emma.jpg")
hermione_encoding = fr.face_encodings(hermione_image)[0]

#match face encoding to desired label
known_encodings = [
    harry_encoding,
    ron_encoding,
    hermione_encoding
]

face_names = [
    "Harry Potter",
    "Ron Wiesley",
    "Hermione Granger"
]

#find faces on group image
face_locations = fr.face_locations(image)
face_encodings = fr.face_encodings(image, face_locations)

#pillow
pillow_image = Image.fromarray(image)
draw = ImageDraw.Draw(pillow_image)
fr.face_landmarks(image)
#loop though detected faces
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    #set default name value
    name = "Unknown"
    #compare current face to declared 'known' faces
    matches = fr.compare_faces(known_encodings, face_encoding)
    face_distances = fr.face_distance(known_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    #set name to face (will be set to 'Unknown' if face is not in the declared faces)
    if matches[best_match_index]:
        name = face_names[best_match_index]
    #draw rectangle on face and add label
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 0))
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height), (right, bottom)), fill=(0, 0, 0), outline=(0, 0, 0))
    draw.text((left, bottom - text_height), name, fill=(255, 255, 255, 255))

#show result
pillow_image.show()

