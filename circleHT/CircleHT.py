#!/usr/bin/python3
# pylint: disable=no-name-in-module

import numpy as np
import argparse
from cv2 import circle, imread, imshow, waitKey, destroyAllWindows
from numba import njit


def show(img, normalize=False, title='DEFAULT', delay=0):
    ''' Function for image display.
        TODO: Display accumulator option.
    '''

    imshow(title, img)
    waitKey(delay)
    destroyAllWindows()


def constructIndices(edges):
    ''' This function returns the indices of the non zero edges     
    '''
    args = np.argwhere(edges > 0)
    I = args[:, 1]
    J = args[:, 0]
    return I, J


def findMaxima(hSpace, numOfMaxima, delta):
    ''' This function returns the local maximas of the accumulator.
        :param accum: The accumulator for the query image.
        :param delta: The radius around the local maxima to set to zero.
        :rtype: list(tuple)    
    '''
    maxs = []

    for _ in range(numOfMaxima):

        mx = np.argwhere(hSpace == np.max(hSpace))[0]
        y, x = mx[0], mx[1]
        maxs.append((x, y))
        hSpace[y - delta: y + delta, x - delta: x + delta] = 0

    return maxs


@njit
def checkBelongs(I, J, x_c, y_c, R):
    ''' This function find the indices that a point belongs to a circle.
    :param I: int[:] // Indicis of possible circle point 
    :param x_c: int // x coordinates of the circle.
    :param R: int // radius of the circle.
    :rtype: int[:]

    '''
    d = (I - x_c) ** 2 + (J - y_c) ** 2
    return np.where(np.abs(d - R ** 2) < 1e-3, 1, 0)


@njit(cache=True)
def constructAccum(I, J, R, shape):
    ''' This function construct the accumulator for a circle with radius R.
        :param shape: 
    '''
    m, n = shape
    accum = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            accum[i, j] += np.sum(checkBelongs(I, J, j, i, R))
    return accum


def load_image(query_name):
    ''' Loads and displays query_image
        :params query_image: Path to the query image.
    '''
    query = imread(query_name, 0)

    if query is None:
        print("Can't find query image")
        exit()

    img = 255 - query

    show(img, title='Query')

    return img

def search_for_circle(query_image, numOfObjects):
    ''' Loads the query_image, build the accumulator and finds local maxima.
        :params query_image: The query image.
    '''
    
    I, J = constructIndices(query_image)

    A = constructAccum(I, J, 58, query_image.shape)

    maxs = findMaxima(A, numOfObjects, 5)

    for mx in maxs:
        circle(query_image, mx, 58, (255, 255, 255), 3)

    show(query_image, title='Result')


# We fix the radius to 58
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required = True, help = "Path to the image")
    ap.add_argument("-n", "--nmax", required = True, help = "Number of maximas")

    args = vars(ap.parse_args())
    img = load_image(args['input'])
    search_for_circle(img, int(args['nmax']))

if __name__ == "__main__":
    main()