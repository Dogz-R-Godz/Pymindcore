import pygame as pg
import numpy as np
import pymind2 as pm
import random as rand
from math import ceil
import time
import rgb_arrays_to_video

randomise_network=True

def create_grids(size=50, head=None, body=None, apple=None):
    # Create three grids of the specified size with all zeros
    head_grid = np.zeros((size, size))
    body_grid = np.zeros((size, size))
    apple_grid = np.zeros((size, size))
    
    # Set the values based on the input coordinates
    if head:
        head_grid[head[0], head[1]] = 1
        
    if body:
        for coord in body:
            body_grid[coord[0], coord[1]] = 1
            
    if apple:
        apple_grid[apple[0], apple[1]] = 1
    
    return head_grid, body_grid, apple_grid

def merge_grids(head_grid, body_grid, apple_grid):
    # Flatten each 2d array and concatenate them to form a single 1d array
    return np.concatenate((head_grid.flatten(), body_grid.flatten(), apple_grid.flatten()))



# Test the function with a different grid size
board_size = 50 #0 - 49
head = (25, 25)
body = [(24, 25), (23, 25), (22, 25)]
apple_positions=np.random.randint(0,49,1000)
apple_positions_2=np.random.randint(0,49,1000)
apple_positions_2.astype(int)
apple_num=0
apple = (apple_positions[apple_num], apple_positions_2[apple_num])
while apple in body or apple == head:
    apple_num+=1
    if len(apple_positions)>=apple_num:
        apple_positions=np.random.uniform(0,49,1000)
        apple_positions_2=np.random.uniform(0,49,1000)
    apple = (apple_positions[apple_num], apple_positions_2[apple_num])



#Create the AI




pg.init()
#Define colours
colours = {'BLACK':(0,0,0),"WHITE":(255,255,255),"GREEN":( 0, 255, 0),"LIME":( 150, 255, 150),"RED":( 255, 0, 0),"BLUE":( 0, 0, 255),"DARK_GREY":( 43, 45, 47),"GREY":(75, 75, 75)}
#Define window size and initialise it.
size = (800, 600) #(0,0)-(600,600) is main board. (600,0)-(800,200) is the first thing of inputs, (600,200)-(800,400) is the second thing of inputs, (600,400)-(800,600) is the third thing of inputs. 
screen = pg.display.set_mode(size)

middle=np.array([2500,64])
inputs=7500 #50x50 board x 3 (apple, head, body)
outputs=4 #forward, backward, left, right
epochs=2 #30 epochs per session. 
DeepQLearning=pm.DQL(inputs,middle,outputs,"sig",0.05)
DeepQLearning.randomise_brain()
DeepQLearning.nn.network[1][0][0] = 1


#icon = pg.image.load('icon.png')
#pg.display.set_icon(icon)
pg.display.set_caption("""Deep Q Snakin! Meet "An AI" (Amazingly named by @That Guy#6482) """) 
carryOn = True
speed=10
mover={0:(1,0),1:(0,1),2:(-1,0),3:(0,-1)}
movement_checker={(1,0):[(1,0),(0,1),(0,-1)],(-1,0):[(-1,0),(0,1),(0,-1)],(0,1):[(1,0),(0,1),(-1,0)],(0,-1):[(1,0),(-1,0),(0,-1)]}
speed_toggler={10:50,50:100,100:10000,10000:5,5:10}
clock = pg.time.Clock()
direction=(1,0)
elipson_greedy=10000
#elipson_greedy = 0
elipson_greedy_error_multiplier=1000
moves=0
loops = 0
flist = []
while carryOn:
    moves+=1
    alive=True
    for event in pg.event.get():
        if event.type==pg.QUIT:
            carryOn=False
        if event.type==pg.KEYDOWN:
            if event.key==pg.K_SPACE:
                speed=speed_toggler[speed]
            

    head_grid,body_grid,apple_grid = create_grids(board_size, head, body, apple)
    inputs=merge_grids(head_grid,body_grid,apple_grid)
    terminal=False
    #p1 = time.time()
    #output=DeepQLearning.better_get_next_frame(inputs)
    p2 = time.time()
    output=DeepQLearning.better_get_next_frame(inputs)
    p3 = time.time()
    #a1 = np.array(output[1])
    #a2 = np.array(output2[1])
    #d = a1 [1][1][-1]- a2[1][1][-1]
    #print(f'time taken with better:{p2-p1}\n time taken with cgpt:{p3-p2}\n difference in output between the functions:{d}')
    #print(f'time taken with cgpt func:{p3-p2}')
    new_direction=mover[output[0]]
    if rand.randint(0,10000)/10 < elipson_greedy:
        direction=rand.choice(movement_checker[direction])
    else:
        if new_direction in movement_checker[direction]:
            direction=new_direction

    body.insert(0,head)
    if head == apple:
        apple_num+=1
        apple = (apple_positions[apple_num], apple_positions_2[apple_num])
        while apple in body or apple == head:
            apple_num+=1
            if len(apple_positions)>=apple_num:
                apple_num=0
                apple_positions=np.random.randint(0,49,1000)
                apple_positions_2=np.random.randint(0,49,1000)
            apple = (apple_positions[apple_num], apple_positions_2[apple_num])
        reward=1
    else:
        body.pop()
    head=(head[0]+direction[0],head[1]+direction[1])
    reward=0.5
    if head in body:
        alive=False
        terminal=True
        reward=0
    elif head[0]>=50 or head[0]<0 or head[1]>=50 or head[1]<0:
        alive=False
        terminal=True
        reward=0
    DeepQLearning.replay_buffer_adder(inputs,output[0],output,reward,terminal)
        
    if not alive:
        head = (25, 25)
        body = [(24, 25), (23, 25), (22, 25)]
        apple_num+=1
        apple = (apple_positions[apple_num], apple_positions_2[apple_num])
        while apple in body or apple == head:
            apple_num+=1
            if len(apple_positions)>=apple_num:
                apple_num=0
                apple_positions=np.random.randint(0,49,1000)
                apple_positions_2=np.random.randint(0,49,1000)
            apple = (apple_positions[apple_num], apple_positions_2[apple_num])
        direction=(1,0)
        if moves >= 500:
            print(moves)
            rgb_arrays_to_video.make_video(flist, 30, f'Recordings/snake_recording_before_backprop_number_{loops}')
            print('made video check if it worked or if it didnt or if it crashed!')
            flist = []
            moves=0
            loops += 1
            expected_outputs,states=DeepQLearning.prepare_for_backprop()
            DeepQLearning.complete_session(expected_outputs,states,epochs,None,True)
            print(f"Prevous Elipson Greedy: {elipson_greedy}")
            elipson_greedy=elipson_greedy_error_multiplier*DeepQLearning.nn.find_error(expected_outputs,states,"sig")
            print(f"New Elipson Greedy: {elipson_greedy}")
        else:
            print(f"Moves: {moves}")
    screen.fill(colours['DARK_GREY'])
    #split the screen
    pg.draw.line(screen,colours["WHITE"],(600,0),(600,600))
    pg.draw.line(screen,colours["WHITE"],(600,200),(800,200))
    pg.draw.line(screen,colours["WHITE"],(600,400),(800,400))

    #draw the apple
    pg.draw.rect(screen,colours["GREEN"],[apple[0]*12,apple[1]*12,12,12])
    pg.draw.rect(screen,colours["GREEN"],[apple[0]*4 + 600,apple[1]*4,4,4])

    #draw the head
    pg.draw.rect(screen,colours["BLUE"],[head[0]*12,head[1]*12,12,12])
    pg.draw.rect(screen,colours["BLUE"],[head[0]*4 + 600,head[1]*4 + 200,4,4])

    #draw the body
    for square in body:
        pg.draw.rect(screen,colours["RED"],[square[0]*12,square[1]*12,12,12])
        pg.draw.rect(screen,colours["RED"],[square[0]*4 + 600,square[1]*4 + 400,4,4])

    #display fps
    screen.blit(pg.font.Font(None,100).render(f'{ceil(clock.get_fps())}', False, (0,255,255)),(0,0))
    pg.display.flip()
    rgb_frame = np.zeros((600,800,3),dtype=np.uint8)
    rgb_frame[0:800,600] = np.array((255,255,255))
    rgb_frame[200,600:800] = np.array((255,255,255))
    rgb_frame[400,600:800] = np.array((255,255,255))
    rgb_frame[apple[0]*12:12+apple[0]*12,apple[1]*12:12+apple[1]*12,1] = 255
    rgb_frame[head[0]*12:12+head[0]*12,head[1]*12:12+head[1]*12,0] = 255
    rgb_frame[apple[0]*4:4+apple[0]*4,600+apple[1]*4:604+apple[1]*4,1] = 255
    rgb_frame[200+head[0]*4:204+head[0]*4,600+head[1]*4:604+head[1]*4,0] = 255
    for square in body:
        rgb_frame[square[0]*12:12+square[0]*12,square[1]*12:12+square[1]*12,2] = 255
        rgb_frame[400+square[0]*4:404+square[0]*4,600+square[1]*4:604+square[1]*4,2] = 255
    flist.append(rgb_frame)
    clock.tick(speed)