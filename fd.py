# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:49:08 2021

@author: mayssem
"""

import face_recognition as fr
from PIL import Image, ImageDraw

image = fr.load_image_file('test2.jpg', 'RGB')

face_locations = fr.face_locations(image)
face_encodings = fr.face_encodings(image, face_locations)

pil_image = Image.fromarray(image)
draw = ImageDraw.Draw(pil_image)
fr.face_landmarks(image)
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    name = "Unknown"
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 0))
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height), (right, bottom)), fill=(0, 0, 0), outline=(0, 0, 0))
    draw.text((left, bottom - text_height), name, fill=(255, 255, 255, 255))

pil_image.show()
