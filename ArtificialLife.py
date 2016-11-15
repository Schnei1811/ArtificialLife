import pygame
import random
import sys
from math import sqrt,fabs,atan2, degrees, pi
from timeit import default_timer as timer
from sklearn.preprocessing import OneHotEncoder
import time
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

sys.setrecursionlimit(10000)
rfclf = pd.read_pickle('data/RFpickle.pickle') 

oldfile = open("data/16VarALDataAIInput.txt","r")
olddata = oldfile.read()
oldfile.close()

n_features = 16
n_classes = 10
poly = PolynomialFeatures(2)

pygame.init()
#pygame.mixer.music.load("data/Corridors_of_Time_Piano_.wav")
#pygame.mixer.music.play(-1)

display_width = 1200
display_height = 800

black = (0, 0, 0)
white = (255, 255, 255)
red = (200, 0, 0)
green = (0, 200, 0)
bright_red = (255, 0, 0)
bright_green = (0, 255, 0)

animal_width = 73
animal_height = 73

gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('Artificial Life')
clock = pygame.time.Clock()

AnimalImg = pygame.image.load('data/animalimg.png')
ShelterImg = pygame.image.load('data/Shelter.png')
Icon = pygame.image.load('data/animalimg.png')
pygame.display.set_icon(Icon)

def deep_neural_network_model(data, hidden_size):

    n_nodes_hl1 = int(hidden_size)
    n_nodes_hl2 = int(hidden_size)
    hidden_1_layer = {'f_fum':n_nodes_hl1,
                      'weight':tf.Variable(tf.random_normal([n_features, n_nodes_hl1])),
                      'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'f_fum':n_nodes_hl2,
                      'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    output_layer = {'f_fum':None,
                    'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                    'bias':tf.Variable(tf.random_normal([n_classes])),}

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']
    return output

def use_deep_neural_network(input_data):

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess,"data/model.ckpt")
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[input_data]}),1)))
    return result

def food_counter(count):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Food Eaten: " + str(count), True, black)
    gameDisplay.blit(text, (0, 0))

def water_counter(count):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Water Consumed: " + str(count), True, black)
    gameDisplay.blit(text, (0, display_height*0.04))

def energy_counter(count):
    font = pygame.font.SysFont("comicsansms", 25)
    count = int(count)
    text = font.render("Energy Regained: " + str(count), True, black)
    gameDisplay.blit(text, (0, display_height*0.08))

def hunger(hungertimer, initialtime, forage,scorefood,scorewater,scoreenergy):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Hunger: " + str(round(initialtime - hungertimer) + forage), True, black)
    gameDisplay.blit(text, (display_width-175, 0))
    if (initialtime - hungertimer+forage) < 0:
        edge_contact(scorefood,scorewater,scoreenergy)
    return

def thirst(thirsttimer, initialtime, forage,scorefood,scorewater,scoreenergy):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Thirst: " + str(round(initialtime - thirsttimer)+forage), True, black)
    gameDisplay.blit(text, (display_width-175, display_height*0.04))
    if (initialtime - thirsttimer+forage) < 0:
        edge_contact(scorefood,scorewater,scoreenergy)
    return

def energy(energytimer, initialtime, forage,scorefood,scorewater,scoreenergy):
    font = pygame.font.SysFont("comicsansms", 25)
    energytimer = int(energytimer)
    initialtime = int(initialtime)
    forage = int(forage)
    text = font.render("Energy: " + str(round(initialtime - energytimer)+forage), True, black)
    gameDisplay.blit(text, (display_width-175, display_height*0.08))
    if (initialtime - energytimer+forage) < 0:
        edge_contact(scorefood,scorewater,scoreenergy)
    return 

def currenttime(timer):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Time: " + str(round(timer)), True, black)
    gameDisplay.blit(text, (display_width/2-50, display_height*0.01))
    return

def edge_contact(scorefood,scorewater,scoreenergy):

    prevscores = np.loadtxt("data/scores.txt",delimiter=",")

    if prevscores[0] + prevscores[1] <= scorefood + scorewater:
        score = open("data/scores.txt","w")
        score.write(str(scorefood) + "," + str(scorewater) + "," + str(scoreenergy))
        score.flush()
        score.close()
        oldfile = open("data/16VarALDataAIInput.txt","r")
        olddata = oldfile.read()
        oldfile.close()
        runfile = open("data/singlerun.txt","r")
        readrunfile = runfile.read()
        runfile.close()
        file = open("data/16VarALDataAIInput.txt", "w")
        file.write(olddata)
        file.write(readrunfile)
        file.flush()
    else:
        oldfile = open("data/16VarALDataAIInput.txt","r")
        olddata = oldfile.read()
        oldfile.close()
        file = open("data/16VarALDataAIInput.txt", "w")
        file.write(olddata)
        file.flush()
    game_loop()

def food(x, y, width, height, color):
    pygame.draw.rect(gameDisplay, color, [x, y, width, height])

def water(x, y, width, height, color):
    pygame.draw.rect(gameDisplay, color, [x, y, width, height])

def animal(x, y):
    gameDisplay.blit(AnimalImg, (x, y))

def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()

def button(msg, x, y, w, h, ic, ac, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    # print(click)
    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        pygame.draw.rect(gameDisplay, ac, (x, y, w, h))
        if click[0] == 1 and action != None:
            action()
    else:
        pygame.draw.rect(gameDisplay, ic, (x, y, w, h))
    smallText = pygame.font.SysFont("comicsansms", 20)
    textSurf, textRect = text_objects(msg, smallText)
    textRect.center = ((x + (w / 2)), (y + (h / 2)))
    gameDisplay.blit(textSurf, textRect)

def quit():
    pygame.quit()
    quit()

def game_intro():
    intro = True

    while intro:
        for event in pygame.event.get():
            # print(event)
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        gameDisplay.fill(white)
        largeText = pygame.font.SysFont("comicsansms", 115)
        TextSurf, TextRect = text_objects("Artificial Life", largeText)
        TextRect.center = ((display_width / 2), (display_height / 2))
        gameDisplay.blit(TextSurf, TextRect)

        button("GO!", display_width*.3, display_height*(5/8), 100, 50, green, bright_green, game_loop)
        button("Quit", display_width*.6, display_height*(5/8), 100, 50, red, bright_red, quit)

        pygame.display.update()
        clock.tick(15)

def game_loop():

    singlerunfile = open("data/singlerun.txt","w")

    global animal_width
    global animal_height
    global file

    x = (display_width * 0.45)
    y = (display_height * 0.8)
    
    x_change = 0
    y_change = 0

    food_startx = random.randrange(0+display_width*0.1, display_width-display_width*.1)
    food_starty = random.randrange(0+display_height*0.1, display_height-display_height*.1)
    food_width = 50
    food_height = 50
    food_speed = 2
    fooddirectioniter = 50
    foodx_change = 0
    foody_change = 0

    water_startx = random.randrange(0+display_width*0.1, display_width-display_width*.1)
    water_starty = random.randrange(0+display_height*0.1, display_height-display_height*.1)
    water_width = 50
    water_height = 50

    shelter_startx = (display_width * 0.8)
    shelter_starty = (display_height * 0.8)
    shelter_width = 115
    shelter_height = 115
    safe = True

    food_green = (0, 170, 0)
    water_blue = (53, 115, 255)
    morning_color = (242, 240, 142)
    noon_color = (64, 187, 247)
    evening_color = (23, 79, 162)
    night_color = (29, 35, 95)

    scorefood = 0
    scorewater = 0
    scoreenergy = 0
    energyregained = 0
    foragefood = 0
    foragewater = 0
    foodtimemultiplier = 2
    watertimemultiplier = 2.5
    energytimemultiplier = 3
    daytimemultiplier = 0.2
    initialfoodtime = 100
    initialwatertime = 100
    initialenergytime = 100
    keypress = 5
    daytimercount = False
    daycount = 0
    starttime = time.time()
    gameExit = False

    while not gameExit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        if fooddirectioniter == 50:
            fooddirectioniter = 0
            fooddirection = random.randrange(1,9,1)
            if fooddirection == 1:
                foodx_change = -5
            if fooddirection == 2:
                foodx_change = 5
            if fooddirection == 3:
                foody_change = 5
            if fooddirection == 4:
                foody_change = 5
            if fooddirection == 5:
                foodx_change = 0
                foody_change = 0
            if fooddirection == 6:
                foodx_change = 5
                foody_change = 5
            if fooddirection == 7:
                foodx_change = -5
                foody_change = 5
            if fooddirection == 8:
                foodx_change = 5
                foody_change = -5
            if fooddirection == 9:
                foodx_change = -5
                foody_change = -5

        fooddirectioniter = fooddirectioniter +1
        food_startx +=foodx_change
        food_starty +=foody_change

        if food_startx - 20 > display_width - food_width:
            foodx_change = -1
        if food_startx < -5:
            foodx_change = 1
        if food_starty-50 > display_height - food_height:
            foody_change = -1
        if food_starty < -5:
            foody_change = 1

        daytimer = round(((time.time()-starttime)*daytimemultiplier),2)
        totaltimer = time.time()-starttime

        if daytimercount == True:
            daytimer = round((daytimer - 4*daycount),2)

        if daytimer < 1:
            safe = True
            gameDisplay.fill(morning_color)
            food_color = food_green
            water_color = water_blue
        elif daytimer < 2:
            gameDisplay.fill(noon_color)
            food_color = food_green
            water_color = water_blue
        elif daytimer < 3:
            gameDisplay.fill(evening_color)
            food_color = food_green
            water_color = water_blue
        elif daytimer < 4:
            safe = False
            gameDisplay.fill(night_color)
            food_color = black
            water_color = black
        elif daytimer < 5:
            daytimercount = True
            daycount = daycount+ 1
            food_color = food_green
            water_color = water_blue

        food(food_startx, food_starty, food_width, food_height, food_color)
        water(water_startx, water_starty, water_width, water_height, water_color)

        animal(x, y)
        food_counter(scorefood)
        water_counter(scorewater)
        energy_counter(energyregained)
        hunger(foodtimemultiplier*totaltimer, initialfoodtime, foragefood,scorefood,scorewater,scoreenergy)
        thirst(watertimemultiplier*totaltimer, initialwatertime, foragewater,scorefood,scorewater,scoreenergy)
        energy(energytimemultiplier*totaltimer, initialwatertime, energyregained,scorefood,scorewater,energyregained)
        gameDisplay.blit(ShelterImg, (shelter_startx, shelter_starty))

        distxfood = food_startx - x
        distyfood = food_starty - y
        distfood = round(sqrt(fabs(distxfood)+fabs(distyfood)),2)

        radfood = atan2(distxfood,distyfood)
        radfood %= 2*pi
        degfood = round(degrees(radfood),2)

        distxwater = water_startx - x
        distywater = water_starty - y
        distwater = round(sqrt(fabs(distxwater) + fabs(distywater)),2)

        radwater = atan2(distxwater, distywater)
        radwater %= 2 * pi
        degwater = round(degrees(radwater),2)

        try:
        	distcorner = round(sqrt(x+y),2)
        except ValueError:
        	distcorner = 0

        radcorner = atan2(y,x)
        radcorner %= 2 * pi
        degcorner = round(degrees(radcorner),2)

        try:
        	distcornerfood = round(sqrt(food_startx+food_starty),2)
        except ValueError:
        	distcornerfood = 0

        radcornerfood = atan2(food_starty, food_startx)
        radcornerfood %= 2 * pi
        degcornerfood = round(degrees(radcornerfood),2)

        try:
        	distcornerwater = round(sqrt(water_startx+water_starty),2)
        except ValueError:
        	distcornerwater = 0

        radcornerwater = atan2(water_starty, water_startx)
        radcornerwater %= 2 * pi
        degcornerwater = round(degrees(radcornerwater),2)

        distcornershelter = round(sqrt(shelter_starty+shelter_startx),2)

        radcornershelter = atan2(shelter_starty, shelter_startx)
        radcornershelter %= 2 * pi
        degcornershelter = round(degrees(radcornershelter), 2)

        foodtime = round(initialfoodtime - foodtimemultiplier*totaltimer) + foragefood
        watertime = round(initialwatertime - watertimemultiplier*totaltimer) + foragewater
        energytime = round(initialenergytime - energytimemultiplier*totaltimer) + energyregained

        singlerunfile.write(str(distfood) + "," + str(degfood) + "," + str(distwater) + "," + str(degwater) + "," + str(distcorner) + "," +
            str(degcorner) + "," + str(distcornerfood) + "," + str(degcornerfood) + "," + str(distcornerwater) + "," +
            str(degcornerwater) + "," + str(distcornershelter) + "," + str(degcornershelter) + "," + str(foodtime) + "," +
            str(watertime) + "," + str(energytime) + "," + str(daytimer) + "," + str(keypress) + "\n")
        singlerunfile.flush()

        X = [distfood,degfood,distwater,degwater,distcorner,degcorner,distcornerfood,degcornerfood,distcornerwater,degcornerwater,distcornershelter,degcornershelter,foodtime,watertime,energytime,daytimer]
        X = poly.fit_transform(X)	
        #AIPredict = use_deep_neural_network(X)
        arrayinput_data = np.array([X])
        AIPredict = rfclf.predict(arrayinput_data)
        print(AIPredict)

        if AIPredict == 4:
            x_change = -5
        if AIPredict == 6:
            x_change = 5
        if AIPredict == 8:
            y_change = -5
        if AIPredict == 2:
            y_change = 5
        if AIPredict == 1:
            x_change = -5
            y_change = 5
        if AIPredict == 5:
            x_change = 0
            y_change = 0
        if AIPredict == 3:
            x_change = 5
            y_change = 5
        if AIPredict == 7:
            x_change = -5
            y_change = -5
        if AIPredict == 9:
            x_change = 5
            y_change = -5

        x += x_change
        y += y_change

        if x-120 > display_width - animal_width:
        	x += 2
        if x < -105:
            x += -2

        if y-150 > display_height - animal_height: 
        	y = y+ 2
        if y < -105:
            y += -2

        shelteredgeverticle = np.arange(shelter_starty,shelter_starty+shelter_height)
        shelteredgehorizontal = np.arange(shelter_startx,shelter_startx+shelter_width)

        if daytimer >3:
            if y > min(shelteredgeverticle) and y < max(shelteredgeverticle)-animal_height and x > min(shelteredgehorizontal) and x < max(shelteredgehorizontal) - animal_width:
                if energytime > 100:
                    energyregained = energyregained
                else:
                    energyregained = energyregained +0.25

        foodedgeverticle = np.arange(food_starty,food_starty+food_height)
        foodedgehorizontal = np.arange(food_startx,food_startx+food_width)
        wateredgeverticle = np.arange(water_starty,water_starty+water_height)
        wateredgehorizontal = np.arange(water_startx,water_startx+water_width)
        animalverticle = np.arange(x,x+animal_height)
        animalhorizontal = np.arange(x,x+animal_width)

        if y > min(foodedgeverticle) and y < max(foodedgeverticle) and x > min(foodedgehorizontal) and x < max(foodedgehorizontal) or y+animal_height*.5 > min(foodedgeverticle) and y+animal_height*.5 < max(foodedgeverticle) and x+animal_width*.5 > min(foodedgehorizontal) and x+animal_width*.5 < max(foodedgehorizontal) or y+animal_width > min(foodedgeverticle) and y+animal_width < max(foodedgeverticle) and x+animal_width > min(foodedgehorizontal) and x+animal_width < max(foodedgehorizontal):
            food_startx = random.randrange(0 + display_width * 0.1, display_width - display_width * .1)
            food_starty = random.randrange(0 + display_height * 0.1, display_height - display_height * .1)
            scorefood += 1
            foragefood += 10
        if y > min(wateredgeverticle) and y < max(wateredgeverticle) and x > min(wateredgehorizontal) and x < max(wateredgehorizontal) or y+animal_height*.5 > min(wateredgeverticle) and y+animal_height*.5 < max(wateredgeverticle) and x+animal_width*.5 > min(wateredgehorizontal) and x+animal_width*.5 < max(wateredgehorizontal) or y+animal_width > min(wateredgeverticle) and y+animal_width < max(wateredgeverticle) and x+animal_width > min(wateredgehorizontal) and x+animal_width < max(wateredgehorizontal):
            water_startx = random.randrange(0 + display_width * 0.1, display_width - display_width * .1)
            water_starty = random.randrange(0 + display_height * 0.1, display_height - display_height * .1)
            scorewater += 1
            foragewater += 10

        pygame.display.update()
        clock.tick(60)

x = tf.placeholder('float', [None, n_features])
hidden_size = 20
prediction = deep_neural_network_model(x,hidden_size)
saver = tf.train.Saver()

game_intro()
game_loop()
pygame.quit()
quit()
file.close()
#output = 0,1,2,3    depending up,down,left,right