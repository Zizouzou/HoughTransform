# pylint: disable=maybe-no-member
import numpy as np
import cv2

class HoughLines:
    def __init__ (self, I, T=0):
        self.img = I
        self.i_max = I.shape[0]
        self.j_max = I.shape[1]
        self.rho_max = int(np.hypot(self.i_max, self.j_max))

        self.threshold = T
    
    def _get_indices(self):
        edges = cv2.Canny(self.img, 80, 90)
        indices = np.argwhere(edges > 0)
        return indices
    
    def _build_hspace(self):
        thetas = np.radians(np.arange(0,180,5))
        cos = np.cos(thetas)
        sin = np.sin(thetas)

        A = np.zeros((len(thetas), 2 * self.rho_max))

        indices = self._get_indices()

        for i,j in indices:
            rho = np.array(i * cos + j * sin, np.int64)
            for k, l in enumerate(rho):
                A[k, l + self.rho_max] += 1
        
        return np.where(A >= np.max(A) - self.threshold, 1, 0)

    def _getPoints(self, theta, rho):
        a, b = (np.cos(theta), np.sin(theta))

        if np.isclose(b, 0):

            p1 = (0, rho)
            p2 = (self.i_max, rho)
            return (p1, p2)
        
        elif np.isclose(a, 0):
            
            p1 = (rho, 0)
            p2 = (rho, self.j_max)
            return (p1, p2)

        else:

            c = rho/b
            p1 = (0, int(c))
            p2 = (self.i_max, int(c - self.i_max * (a/b)))
            return (p1, p2)

    def getLines(self):
        A = self._build_hspace()

        lines = []

        for theta, rho in np.argwhere(A>0):
            theta = np.radians(5 * theta)
            lines.append(self._getPoints(theta, rho - self.rho_max))

        return lines


    def drawLines(self):
        
        lines = self.getLines()
        color_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        for p1, p2 in lines:
            cv2.line(color_img, p1, p2, (0,255,0), 2)

        cv2.imshow('Hough Lines', color_img)
        cv2.waitKey()

def addNoise (I, sigma):
    noise = np.random.normal(0, sigma, I.shape)
    I_noised = noise + I
    cv2.normalize(I_noised, I_noised, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    return I_noised.astype(np.uint8)


img = cv2.imread('kygo.jpg', 0)
lines = HoughLines(img, 45)

lines.drawLines()

        