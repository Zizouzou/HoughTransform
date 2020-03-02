#!/usr/bin/python3
# pylint: disable=no-name-in-module

import numpy as np
import argparse
from sys import exit
from numba import njit
from cv2 import filter2D, imread, imshow, waitKey, destroyAllWindows
from collections import defaultdict


def show(img, normalize=False, title='DEFAULT', delay=0):
    ''' Function for image display.
        TODO: Display accumulator option.
    '''

    imshow(title, img)
    waitKey(delay)
    destroyAllWindows()


def getSobel(img, threshold=50):
    ''' This function approximates the derivatives of an image by convolving with Sobel kernels.
        :param img: Grayscale image of type np.ndarray.
        :param threshold: Threshold the results of the convolution with this value. Defaults to 50.
        :rtype: Tuple(int[:, :], int[:, :]).
    '''

    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    dX = filter2D(img, -1, sobel)
    dX = np.where(np.abs(dX) < threshold, 0, dX)

    dY = filter2D(img, -1, sobel.T)
    dY = np.where(np.abs(dY) < threshold, 0, dY)

    return (dX.astype(np.intc), dY.astype(np.intc))


@njit
def getOrientations(dX, dY):
    ''' This function returns the gradient orientations in degrees.
        :rtype: int[:, :]
    '''
    return np.degrees(np.arctan2(dY, dX)).astype(np.intc)


@njit(nogil=True, cache=True)
def getIndices(dX, dY):
    ''' This function returns the indices of the pixels that have at least one non-zero partial derivative.
        :rtype: tuple(int[:], int[:])
    '''
    m, n = dX.shape
    I = []
    J = []

    for i in range(m):
        for j in range(n):
            if dX[i, j] != 0 or dY[i, j] != 0:
                I.append(i)
                J.append(j)

    return np.array(I).astype(np.intc), np.array(J).astype(np.intc)


@njit(nogil=True, cache=True)
def getValidIndices(I, J, i_limit, j_limit):
    ''' This function is used to ensure that the candidates for reference point in the query image are in the bounds of the image.
        :params I, J: np.ndarray array must be of shape (N,).
        :params i_limit, j_limit: The upper bounds that the refPoint must respect.
        :rtype: Tuple(list).
    '''
    I_res = []
    J_res = []

    m = len(I)

    for i in range(m):
        if (I[i] >= 0 and I[i] < i_limit) and (J[i] >= 0 and J[i] < j_limit):
            I_res.append(I[i])
            J_res.append(J[i])

    return I_res, J_res


def buildTable(template, refPoint):
    ''' This function returns the RTable of a shape.
        :param template: Image of the template shape.
        :param refPoint: Reference point used for building the RTable.
        :rtype: dict(list) // dict keys are the gradient orientations at the edges and the lists contains
        the difference between the reference point and the edge's location.
    '''
    dX, dY = getSobel(template)
    I, J = getIndices(dX, dY)
    orientations = getOrientations(dX[I, J], dY[I, J])

    dR_x = I - refPoint[0]
    dR_y = J - refPoint[1]

    r_table = defaultdict(list)

    for k, phi in enumerate(orientations):
        r_table[phi].append((dR_x[k], dR_y[k]))

    return r_table


def buildAccum(query_img, r_table):
    ''' Takes as input a query image and the R table of the template returns the accumulator.
        :param query_image: Grayscale image.
        :param r_table: RTable of the target shape.
        :rtype: np.ndarray // Same shape as the query image.
    '''
    m, n = query_img.shape

    dX, dY = getSobel(query_img, 70)
    I, J = getIndices(dX, dY)
    orientations = getOrientations(dX[I, J], dY[I, J])

    accum = np.empty_like(query_img).astype(np.uint64)

    for phi in r_table.keys():
        phi_index = np.argwhere(orientations == phi)

        I_phi = I[phi_index]
        J_phi = J[phi_index]

        for dr in r_table[phi]:
            I_refs = I_phi + dr[0]
            J_refs = J_phi + dr[1]

            N = len(I_refs,)
            I_refs = I_refs.reshape((N,))
            J_refs = J_refs.reshape((N,))
            I_valids, J_valids = getValidIndices(I_refs, J_refs, m, n)

            accum[I_valids, J_valids] += 1

    return accum


def findMaxima(accum, numOfMaxima, delta):
    ''' This function returns the local maximas of the accumulator.
        :param accum: The accumulator for the query image.
        :param delta: The radius around the local maxima to set to zero.
        :rtype: list(tuple)    
    '''

    maximas = []

    for _ in range(numOfMaxima):

        maxima = np.argwhere(accum == np.max(accum))[0]
        x, y = maxima[0], maxima[1]
        maximas.append((x, y))
        accum[x - delta: x + delta, y - delta: y + delta] = 0

    return maximas


def build_from_template(tmple_name):
    ''' Loads template, displays it and builds the corresponding r_table.
        :params tmpl_image: Path to the template image.
    '''
    tmpl = imread(tmple_name, 0)

    if tmpl is None:
	    print("Can't find template_image")
	    exit()

    show(tmpl, title='Template')

    m, n = tmpl.shape
    refPoint = (m // 2, n // 2)

    return buildTable(tmpl, refPoint)


def search_for_template(query_image, r_table, numOfObjects):
    ''' Loads the query_image, build the accumulator and finds local maxima.
        :params query_image: Path to the query image.
    '''
    
    img = imread(query_image, 0)

    if img is None:
	    print("Can't find query_image")
	    exit()

    accum = buildAccum(img, r_table)

    for x in findMaxima(accum, numOfObjects, 30):
        img[x[0]-5:x[0]+5, x[1]-5:x[1]+5] = 0

    show(img, title='Query_Image')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tmpl", required = True, help = "Path to the template")
    ap.add_argument("-n", "--nmax", required = True, help = "Number of maximas")
    ap.add_argument("-q", "--query", required=True, help = "Path to the query image")

    args = vars(ap.parse_args())

    r_table = build_from_template(args["tmpl"])
    search_for_template(args['query'], r_table, int(args['nmax']))


if __name__ == "__main__":
    main()
