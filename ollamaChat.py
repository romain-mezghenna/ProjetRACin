import ollama 
import videoEditor as ve
import os
# Get a random image from the video video.mp4 at 15 seconds

image = ve.extract_image("videomp4_out", 13) 

# the path of the image is ./images/video-15.png

response = ollama.chat(model="llava", messages= [ 
    {
        'role' : 'system',
        'content': """You're about to analyze an image that comes from a movie and has been preprocessed by a object detection model. You need to describe the scene and gives details about all the alchol related objects detected. You're response should be as short as possible because it will be encoded in a vectorial space. Do not describe what is the image, describe only the content."""
    },
    {
      'role': 'user',
      'content': "Analyze the image and make the shortest description possible. Do nott describe the image, describe the content. Do no make assumptions or approximations.",
      'images' : [image]
    },
  ]
)



print(response['message']['content'])



# Delete the image
#os.remove(image)