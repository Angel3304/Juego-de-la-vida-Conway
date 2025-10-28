import pygame
import numpy as np
import time

pygame.init()
width, height = 1000, 1000
screen = pygame.display.set_mode((width, height))
bg= 25,25,25
screen.fill(bg)
nxC,nyC = 40,40
dimCW=width/nxC
dimCH=height/nyC

#Estado de las celdas. Vivas=1; Muertas =0
gameState = np.zeros((nxC,nyC))

#Automata palo.
gameState[5,3]=1
gameState [5,4]=1
gameState [5,5]=1

#Bucle de ejecucion
while True:
    newGameState = np.copy(gameState)

    screen.fill(bg)
    time.sleep(0.1)

    for y in range(0,nxC):
        for x in range(0,nyC):

            #Calculamos el numero de vecinos
            n_neighbors = gameState[(x-1)%nxC,(y-1)%nyC]+ \
                          gameState[(x) %nxC,(y-1) %nyC ] + \
                          gameState[(x+1) %nxC,(y-1) %nyC] +\
                          gameState[(x-1) %nxC,(y) %nyC] + \
                          gameState[(x+1) %nxC,(y)%nyC] + \
                          gameState[(x-1) %nxC,(y+1) %nyC] + \
                          gameState[(x) %nxC, (y+1) %nyC] + \
                          gameState[(x+1) %nxC,(y+1) % nyC]

            #Primera regla: Si tiene 3 vecinos vivos, "revive"
            if gameState[(x,y)] == 0 and n_neighbors== 3:
                newGameState[x,y]=1

            #Segunda regla: La celda viva con menos de 2 o mas de 3 vecinos, "muere"
            elif gameState[(x,y)] == 1 and (n_neighbors < 2 or n_neighbors> 3):
                newGameState[x,y]=0

            #Creamos el poligono de cada celda a dibujar
            poly = [((x)*dimCW, y * dimCH),
                    ((x+1)*dimCW, y * dimCH),
                    ((x+1)*dimCW, (y+1) * dimCH),
                    ((x)*dimCW, (y+1)*dimCH)
            ]
            #Lo dibujamos
            if newGameState[x,y] == 0:
                pygame.draw.polygon(screen,(128,128,128),poly,1)
            else:
                pygame.draw.polygon(screen,(255,255,255),poly,0)
    #Actualizamos los estados de las celdas
    gameState = np.copy(newGameState)

    #Actualizamos la pantalla
    pygame.display.flip()


