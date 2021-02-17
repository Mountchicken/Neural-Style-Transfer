# Neural-Style-Transfer
New Year, New Style For Your Image

## Structure
### File

- `main.py`: models and codes for NST,run it
- `content`: where you stored the content image
- `style`  : where you stored the style image
- `generated`:The genearted images



## Which hyperparameters Can be changed in the main.py

### line33--image_size
- this parameter determines the size of the geneareted image and the size which the content image will
- be resize to. The larger it is, the better the quality of the geneareted image will be with the cost
- of more training time

### line45-line46 --steps and lr
- you can use a decreasing learning-rate and less or more steps depends on the training result

### line47-line48 --alpha and beta
- The larger the alpha, the more the generated image looks like the content iamge.The larger the beta,
- the more the generated image looks like the style image

## contact information
- mountchicken@outlook.com
