import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import numpy as np
from collections import deque
from tryalgo.graph import add_reverse_arcs
import tryalgo
import copy

obraz = input('Podaj nazwe obrazu z rozszerzeniem (np. obraz.jpg, obraz.png):')
metoda = input('Podaj filter (sobel, prewitt, roberts):')

im = cv2.imread(obraz, 0)   #wczytanie obrazu

#storzenie macierzy z wierzchołkami, które są kolejnymi numerami pikseli
rozmiarwpionie = im.shape[0]
rozmiarwpoziomie = im.shape[1]
ilosc = rozmiarwpionie * rozmiarwpoziomie
G = np.arange(ilosc).reshape(rozmiarwpionie, rozmiarwpoziomie)

#oznaczanie obiektu (źródła)
ax = plt.gca()
fig = plt.gcf()
implot = ax.imshow(im, cmap='gray')
fig.canvas.set_window_title('Zaznacz obiekt do wycięcia')

line, = ax.plot([], [], 'g')

zrodla = []

def moved_and_pressed(event):
    if event.button==1:
        if G[int(round(event.ydata)),int(round(event.xdata))] not in zrodla:
            zrodla.append(G[int(round(event.ydata)), int(round(event.xdata))])
        x = np.append(line.get_xdata(), event.xdata)
        y = np.append(line.get_ydata(), event.ydata)
        line.set_data(x, y)
        fig.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', moved_and_pressed)
cid = fig.canvas.mpl_connect('motion_notify_event', moved_and_pressed)

#oznaczanie tła (ujścia)
plt.figure()
ax01 = plt.gca()
fig1 = plt.gcf()
implot1 = ax01.imshow(im, cmap='gray')
fig1.canvas.set_window_title('Zaznacz tło obiektu')

line1, = ax01.plot([], [], 'r')

ujscia = []

def moved_and_pressed1(event):
    if event.button==1:
        if G[int(round(event.ydata)),int(round(event.xdata))] not in zrodla:
            ujscia.append(G[int(round(event.ydata)), int(round(event.xdata))])
        x = np.append(line1.get_xdata(), event.xdata)
        y = np.append(line1.get_ydata(), event.ydata)
        line1.set_data(x, y)
        fig1.canvas.draw()

cid1 = fig1.canvas.mpl_connect('button_press_event', moved_and_pressed1)
cid1 = fig1.canvas.mpl_connect('motion_notify_event', moved_and_pressed1)

plt.show()


graf = {}   #graf

#dodanie źródła i połączenie go z pikselami oznaczającymi obiekt
tempdictz = {}
for i in zrodla:
    tempdictz[i] = 1000

graf[ilosc] = tempdictz


#funkcja z pakietu tryalgo do wyznaczania ścieżki powiększającej
#The MIT License (MIT)

#Copyright (c) 2016 Jill-Jênn Vie

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
def _augment(graph, capacity, flow, source, target):
    """find a shortest augmenting path
    """
    n = len(graph)
    A = [0] * n
    augm_path = [None] * n
    Q = deque()
    Q.append(source)
    augm_path[source] = source
    A[source] = float('inf')
    while Q:
        u = Q.popleft()
        for v in graph[u]:
            cuv = capacity[u][v]
            residual = cuv - flow[u][v]
            if residual > 0 and augm_path[v] is None:
                augm_path[v] = u
                A[v] = min(A[u], residual)
                if v == target:
                    break
                else:
                    Q.append(v)
    return (augm_path, A[target])



#funkcja z pakietu tryalgo do wyznaczania maksymalnego przepływu
#funkcja została zmodyfikowana tak, aby zwracała maksymalny przepływ w formie zagnieżdzonego słownika (wcześniej zwracała macierz przepływu)
def edmonds_karp(graph, capacity, source, target):
    """Maxmum flow by Edmonds-Karp
    :param graph: directed graph in listlist or listdict format
    :param capacity: in matrix format or same listdict graph
    :param int source: vertex
    :param int target: vertex
    :returns: flow dictdict, flow value
    :complexity: :math:`O(|V|*|E|^2)`
    """
    add_reverse_arcs(graph, capacity)
    V = range(len(graph))
    flow = copy.deepcopy(graf)
    for i in V:
        for j in flow[i]:
            flow[i][j] = 0
    while True:
        augm_path, delta = _augment(graph, capacity, flow, source, target)
        if delta == 0:
            break
        v = target
        while v != source:
            u = augm_path[v]
            flow[u][v] += delta
            flow[v][u] -= 0
            v = u
    a = 0
    for i in flow[source]:
        a = a + flow[source][i]
    return (flow, a)

#filtry Robertsa
roberts_y = np.array( [[ 0, 0, 0 ],
                             [ 0, 1, 0 ],
                             [ 0, 0,-1 ]] )

roberts_x = np.array( [[ 0, 0, 0 ],
                             [ 0, 0, 1 ],
                             [ 0,-1, 0 ]] )

#gradient obrazu
imm = im.astype('int32')

if metoda == 'roberts':
    dy = ndimage.convolve(imm, roberts_y)
    dx = ndimage.convolve(imm, roberts_x)
if metoda == 'prewitt':
    dx = ndimage.prewitt(imm, 0)
    dy = ndimage.prewitt(imm, 1)
if metoda == 'sobel':
    dx = ndimage.sobel(imm, 0)
    dy = ndimage.sobel(imm, 1)

mag = np.hypot(dx, dy)
mag *= 255.0 / np.max(mag)

fig2 = plt.figure(num='Obraz i gradient obrazu')
plt.gray()
ax1 = fig2.add_subplot(121)
ax2 = fig2.add_subplot(122)

cv2.imwrite('WYNIKgradient' + metoda + obraz, mag)

ax1.imshow(imm)
ax2.imshow(mag)
plt.show()

#zamiana gradientu obrazu na graf
for (x, y), value in np.ndenumerate(G):
    if x == 0 and y == 0:
        graf[value] = {G[x + 1, y]: np.exp(-1 * (mag[x, y] + mag[x + 1, y]) / 2),
                       G[x, y + 1]: np.exp(-1 * (mag[x, y] + mag[x, y + 1]) / 2)}
    elif x == 0 and y == G.shape[1] - 1:
        graf[value] = {G[x + 1, y]: np.exp(-1 * (mag[x, y] + mag[x + 1, y]) / 2),
                       G[x, y - 1]: np.exp(-1 * (mag[x, y] + mag[x, y - 1]) / 2)}
    elif x == G.shape[0] - 1 and y == 0:
        graf[value] = {G[x - 1, y]: np.exp(-1 * (mag[x, y] + mag[x - 1, y]) / 2),
                       G[x, y + 1]: np.exp(-1 * (mag[x, y] + mag[x, y + 1]) / 2)}
    elif x == G.shape[0] - 1 and y == G.shape[1] - 1:
        graf[value] = {G[x - 1, y]: np.exp(-1 * (mag[x, y] + mag[x - 1, y]) / 2),
                       G[x, y - 1]: np.exp(-1 * (mag[x, y] + mag[x, y - 1]) / 2)}
    elif x == 0 and y != 0 and y != G.shape[1] - 1:
        graf[value] = {G[x, y - 1]: np.exp(-1 * (mag[x, y] + mag[x, y - 1]) / 2),
                       G[x + 1, y]: np.exp(-1 * (mag[x, y] + mag[x + 1, y]) / 2),
                       G[x, y + 1]: np.exp(-1 * (mag[x, y] + mag[x, y + 1]) / 2)}
    elif x == G.shape[0] - 1 and y != 0 and y != G.shape[1] - 1:
        graf[value] = {G[x, y - 1]: np.exp(-1 * (mag[x, y] + mag[x, y - 1]) / 2),
                       G[x - 1, y]: np.exp(-1 * (mag[x, y] + mag[x - 1, y]) / 2),
                       G[x, y + 1]: np.exp(-1 * (mag[x, y] + mag[x, y + 1]) / 2)}
    elif y == 0 and x != 0 and x != G.shape[0] - 1:
        graf[value] = {G[x - 1, y]: np.exp(-1 * (mag[x, y] + mag[x - 1, y]) / 2),
                       G[x + 1, y]: np.exp(-1 * (mag[x, y] + mag[x + 1, y]) / 2),
                       G[x, y + 1]: np.exp(-1 * (mag[x, y] + mag[x, y + 1]) / 2)}
    elif y == G.shape[1] - 1 and x != 0 and x != G.shape[0] - 1:
        graf[value] = {G[x - 1, y]: np.exp(-1 * (mag[x, y] + mag[x - 1, y]) / 2),
                       G[x + 1, y]: np.exp(-1 * (mag[x, y] + mag[x + 1, y]) / 2),
                       G[x, y - 1]: np.exp(-1 * (mag[x, y] + mag[x, y - 1]) / 2)}
    else:
        graf[value] = {G[x - 1, y]: np.exp(-1 * (mag[x, y] + mag[x - 1, y]) / 2),
                       G[x, y - 1]: np.exp(-1 * (mag[x, y] + mag[x, y - 1]) / 2),
                       G[x + 1, y]: np.exp(-1 * (mag[x, y] + mag[x + 1, y]) / 2),
                       G[x, y + 1]: np.exp(-1 * (mag[x, y] + mag[x, y + 1]) / 2)}

#dodanie ujścia i połączenie go z pikselami oznaczającymi tło
for i in ujscia:
    graf[i].update({ilosc + 1: 1000})

graf[ilosc + 1] = {}

#wyznaczenie maksymalnego przepływu
#f-maksymalny przepływ v-wartość maksymalnego przepływu
f, v = edmonds_karp(graf, graf, ilosc, ilosc + 1)


#odejmowanie maksymalnego przepływu od przepustowości
grafzredukowany = copy.deepcopy(graf)
for i in range(len(graf)):
    for j in graf[i]:
        grafzredukowany[i][j] -= f[i][j]


plt.figure(num='Obraz')
plt.imshow(im)
plt.figure(num='Obraz z zaznaczonym obiektem i tłem')

#zapis obrazu z zaznaczonymi źródłami i ujściami
immm = np.dstack([im, im, im])
for i in zrodla:
    result = np.where(G==i)
    immm[result[0][0],result[1][0]] = (0,255,0)

for i in ujscia:
    result = np.where(G==i)
    immm[result[0][0],result[1][0]] = (0,0,255)

plt.imshow(immm)
cv2.imwrite('WYNIKzrodlaujscia' + metoda + obraz, immm)
plt.imshow(immm)

plt.figure(num='Gradient obrazu')
plt.imshow(mag)


#przeszukiwanie grafu, aby wyciąć pożądany obiekt
image = np.array(im)

odwiedzone = []
for i in zrodla:
    odwiedzone.append(i)

queue = []
for i in zrodla:
    queue.append(i)

while queue:
    s = queue.pop(0)
    for i in grafzredukowany[s]:
        if i not in odwiedzone:
            if grafzredukowany[s][i] > 0:
                odwiedzone.append(i)
                queue.append(i)

#wycięcie obiektu (zamiana pikseli tła na kolor biały)
for (x, y), value in np.ndenumerate(G):
    if value not in odwiedzone:
        image[x,y] = 255

#zapis obrazu z wyciętym obiektem
plt.figure(num='Wynik segmentacji obrazu')
plt.imshow(image)
cv2.imwrite('WYNIKsegmentacja' + metoda + obraz, image)
plt.show()
