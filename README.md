# Lego_pattern_recognition
This project is for recognizing colors and special patterns of Lego Blocks. Mainly program is made by Numpy and OpenCV library. Firstly It loads photos to detect edges of patterns by Sobel filters. Then it compares with example patterns to recognise specific. After that elements are analized in term of colors. Final result of image is how many specific patterns and colors of blocks are in the picture. Program takes 2 arguments: path to files and path to result.

Program is recognizing 5 patterns and 6 colors:
### Colors:

-white,

-yellow,

-red,

-green,

-blue,


### Patterns:

-square,

-rectangle,

-tetris shape,

-Z shape,

-L shape,

#### Input image
![img_003_scaled](https://user-images.githubusercontent.com/50676292/228884629-69fe9d0e-36d2-424b-bc53-f9b5fd200aa1.png)

#### Sobel edge detection
![img_003_binary](https://user-images.githubusercontent.com/50676292/228884775-691cd090-0732-4a30-8c96-6d0f88f0df3c.png)

#### Example masks of elements

![l_shape_mask](https://user-images.githubusercontent.com/50676292/228884929-4ec766d7-aa56-403b-a1e5-36592ba9daf9.png)
![tetris_mask](https://user-images.githubusercontent.com/50676292/228884954-6f24171a-e8bf-4d69-8324-0e02da1f8a4f.png)

#### Example results

"img_003.jpg": 

        [
        {
            "square": 0,
            "rectangle": 0,
            "tetris": 3,
            "z_shape": 8,
            "L_shape": 0
        },
        {
            "blue": 0,
            "red": 1,
            "green": 1,
            "white": 0,
            "yellow": 1,
            "mix": 8
        }
        ]
