# Import required modules
from PIL import Image
import numpy as np
  
# Load image
image = Image.open('screen.png')
  
# Convert image to array
image_arr = np.array(image)
  
# Crop image
image_arr = image_arr[
    # y
    # top - bottom
    262:335, 
    # x
    # left - right
    127:200]
  
# Convert array to image
image = Image.fromarray(image_arr)
  
# Display image
image.show()
image.save('crop.png')