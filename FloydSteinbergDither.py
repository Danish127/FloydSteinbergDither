#pip install Pillow, numpy

from PIL import Image
import numpy as np
import math, sys
import random

#Color palette of Black, White, Green, Blue, Red, Yellow, Orange, Gray
Palette = [(0,0,0), (255,255,255), (0,255,0), (0,0,255), (255,0,0), (255,255,0), (255,165,0), (128,128,128)]

#Will be populated with the HunterLab values of the above Palette
LABPalette = []

#Allows multiple files in sequence to be dithered at once
def FloydSteinbergDither(*argv):
    #Foreach file
    for arg in argv:
        #Do the thing
        Dither(arg)


#Ratio of Error to distribute to East Pixel
EastRatio = (7.00 / 16.00)
#Ratio of Error to distribute to South-East Pixel
SouthEastRatio = (1.00 / 16.00)
#Ratio of Error to distribute to South Pixel
SouthRatio = (5.00 / 16.00)
#Ratio of Error to distribute to South-West Pixel
SouthWestRatio = (3.00 / 16.00)

#Floyd-Steinberg Dithering, using HunterLab intermediary
def Dither(fname):
    #global Palette
    global LABPalette
    #If the HunterLab palette has not been populated
    if len(LABPalette) == 0:
        for tone in Palette:
            #Transform the Palette entries to HunterLab
            LABPalette.append(toHunterLab(tone[0], tone[1], tone[2]))
    #Pillow open the image file
    img = Image.open(fname)
    #Numpy array it
    ary = np.array(img)
    # Split the three channels
    if img.mode == "RGB":
        r,g,b = np.split(ary,3,axis=2)
        r=r.reshape(-1)
        g=g.reshape(-1)
        b=b.reshape(-1)
        a=[]
    #Split the four channels (png)
    elif img.mode == "RGBA":
        r,g,b,a = np.split(ary,4,axis=2)
        r=r.reshape(-1)
        g=g.reshape(-1)
        b=b.reshape(-1)
        #a=a.reshape(-1)
        #We choose to convert transparent pixels to White later on

    #Convert the imagepixel by pixel into HunterLab
    HunterLab = list(map(lambda x: toHunterLab(x[0],x[1],x[2]), zip(r,g,b)))
    #Reshape above array into image with original dimentions
    HunterLab = np.array(HunterLab).reshape([ary.shape[0], ary.shape[1]])
    

    #Will be a list of Palette indecies
    ms = []
    
    #Iterate through the image, x first
    for y in range(ary.shape[0]):
        for x in range(ary.shape[1]):
            #Grab the current value from the array
            Old = HunterLab[y, x]
            #Get the index of the closest palette color
            index = FindClosestLAB(Old)

            #If the pixel is supposed to be transparent, replace it with white, but continue to distribute the error outward
            if len(a) > 1 and a[y, x] == 0:
                index = 1
                #Toggle to true if you want to skip transparent pixels for error distribution
                if True:
                    ms.append(index)
                    continue
                    #Possibly use a heuristic to skip large groups of transparent pixels, but preserve for small groups

            #Grab the HunterLab value for the index
            New = LABPalette[index]
            
            #Calculate deltas
            Delta_L = Old["L"] - New["L"]
            Delta_A = Old["A"] - New["A"]
            Delta_B = Old["B"] - New["B"]

            #Bounds check and distribute the error from defined ratios

            #East Pixel
            if x < (ary.shape[1] - 1):
            
                NewEast = HunterLab[y, x+1]
                NewEast["L"] += Delta_L * EastRatio
                NewEast["A"] += Delta_A * EastRatio
                NewEast["B"] += Delta_B * EastRatio

                HunterLab[y, x+1] = NewEast
            

            if y < (ary.shape[0] - 1):
            

                #Southwest Pixel
                if (x > 0):
                
                    NewSouthwest = HunterLab[y+1, x-1]
                    NewSouthwest["L"] += Delta_L * SouthWestRatio
                    NewSouthwest["A"] += Delta_A * SouthWestRatio
                    NewSouthwest["B"] += Delta_B * SouthWestRatio

                    HunterLab[y+1, x-1] = NewSouthwest
                

                #South Pixel
                
                NewSouth = HunterLab[y+1, x]
                NewSouth["L"] += Delta_L * SouthRatio
                NewSouth["A"] += Delta_A * SouthRatio
                NewSouth["B"] += Delta_B * SouthRatio

                HunterLab[y+1, x] = NewSouth
                

                #Southeast Pixel
                if (x < (ary.shape[1] - 1)):
                
                    NewSoutheast = HunterLab[y+1, x+1]
                    NewSoutheast["L"] += Delta_L * SouthEastRatio
                    NewSoutheast["A"] += Delta_A * SouthEastRatio
                    NewSoutheast["B"] += Delta_B * SouthEastRatio

                    HunterLab[y+1, x+1] = NewSoutheast
            #Add the index to the list
            ms.append(index)
    #Uncomment for generating a palette-shuffled image
    #random.shuffle(Palette)

    #Once all done, convert the array of indecies back into the original palette
    ms = list(map(lambda x: Palette[x], ms))

    #Shape ms back into an image, with a bitdepth of 3 bytes per pixel
    bitmap = np.array(ms).reshape([ary.shape[0], ary.shape[1], 3])

    #create a new image
    im = Image.fromarray(bitmap.astype(np.uint8))

    #show and save
    im.show()
    im.save(fname + '_Dithered.bmp', )


#Code converted from ColorMine math to get RGB -> HunterLab
def toHunterLab(R, G, B):
    #Attribute https://github.com/colormine/colormine
    r = pivotRgb(R / 255.0)
    g = pivotRgb(G / 255.0)
    b = pivotRgb(B / 255.0)

    # Observer. = 2°, Illuminant = D65
    X = r * 0.4124 + g * 0.3576 + b * 0.1805
    Y = r * 0.2126 + g * 0.7152 + b * 0.0722
    Z = r * 0.0193 + g * 0.1192 + b * 0.9505

    LAB_L = 10.0 * math.sqrt(Y)
    LAB_A =  17.5 * ((1.02 * X - Y) / math.sqrt(Y)) if Y != 0.0 else 0.0
    LAB_B = 7.0 * ((Y - 0.847 * Z) / math.sqrt(Y)) if Y != 0.0 else 0.0

    return {'L' : LAB_L, 'A' : LAB_A, 'B' : LAB_B }


def pivotRgb(n):
    return (math.pow((n + 0.055) / 1.055, 2.4) if n > 0.04045 else n / 12.92) * 100

#Find the closest HunterLab to input
def FindClosestLAB(OldPixel):
    #Set lowest to maximum value
    lowest = sys.float_info.max
    #start with Black as closest
    index = 0
    #Iterate through palette
    for LAB in LABPalette:
        #Calculate deltas
        LDelta = LAB["L"] - OldPixel["L"]
        ADelta = LAB["A"] - OldPixel["A"]
        BDelta = LAB["B"] - OldPixel["B"]
        #Magnitude of the difference
        mag = math.sqrt((LDelta * LDelta) + (ADelta * ADelta) + (BDelta * BDelta))

        #If magnitude is lower, update index and lowest
        if mag <= lowest:
            lowest = mag
            index = LABPalette.index(LAB)
    #return index of lowest
    return index


FloydSteinbergDither("C:\\{path to image to dither}")