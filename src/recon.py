import numpy as np
import scipy
import scipy.misc
import scipy.ndimage
import scipy.io
import scipy.interpolate
import skimage
import time
import cv2
import os.path
import skimage.morphology
from scipy.ndimage.filters import uniform_filter


class Reconstruction(object):
    def __init__(self):
        super(object).__init__()
        # lambda is the wavelength in meters (i.e. 405nm = UV light)
        self.lmbda = 405e-9
        self.UpsampleFactor = 2
        self.delta2 = 2.2e-6
        self.Dz = 5e-4
        self.Threshold_objsupp = 0.09
        self.NumIteration = 30
        self.std_filter_size = 9
        self.dilation_size = 6
        self.debug = False

    def upsampling(self, data, dx1):
        dx2 = dx1 / (2 ** self.UpsampleFactor)
        x_size = ((2 ** self.UpsampleFactor) * data.shape[0]) - (2 ** (self.UpsampleFactor) - 1)
        y_size = ((2 ** self.UpsampleFactor) * data.shape[1]) - (2 ** (self.UpsampleFactor) - 1)
        upsampled = scipy.ndimage.zoom(data, [x_size / data.shape[0], y_size / data.shape[1]], order=3)
        self.debug_save_mat(upsampled, 'upsampledPy')
        return upsampled, dx2

    def ft2(self, g, delta):
        return np.fft.fftshift(np.fft.fft2((np.fft.fftshift(g)))) * delta ** 2

    def ift2(self, G, dfx, dfy):
        Nx, Ny = np.shape(G)
        return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(G))) * Nx * Ny * dfx * dfy

    def debug_save_mat(self, matrix, mname):
        if self.debug:
            if matrix is None:
                matrix = np.array([0])

            file_path = '../debug/' + mname + '.mat'
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            scipy.io.savemat(file_path, {mname: matrix})

    def compute(self, data):
        k = 2 * np.pi / self.lmbda
        subNormAmp = np.sqrt(data)
        # delta_prev = self.delta2

        if self.UpsampleFactor > 0:
            subNormAmp, self.delta2 = self.upsampling(subNormAmp, self.delta2)

        self.debug_save_mat(subNormAmp, 'subNormAmpPy')

        Nx, Ny = np.shape(subNormAmp)
        delta1 = self.delta2

        dfx = 1 / (Nx * self.delta2)
        dfy = 1 / (Ny * self.delta2)

        fx, fy = np.meshgrid(np.arange(-Ny / 2, Ny / 2, 1) * dfy,
                             np.arange(-Nx / 2, Nx / 2, 1) * dfx)

        Gbp = np.exp((1j * k * self.Dz) * np.sqrt(1 - self.lmbda ** 2 * fx ** 2 - self.lmbda ** 2 * fy ** 2))
        Gfp = np.exp((-1j * k * self.Dz) * np.sqrt(1 - self.lmbda ** 2 * fx ** 2 - self.lmbda ** 2 * fy ** 2))

        self.debug_save_mat(Gbp, 'GbpPy')
        Input = subNormAmp

        for k in range(self.NumIteration):
            F2 = self.ft2(Input, self.delta2)
            Recon1 = self.ift2(np.multiply(F2, Gbp), dfx, dfy)
            self.debug_save_mat(Recon1, 'Recon1Py')
            if k == 0:
                # abs(Recon1).*cos(angle(Recon1) == abs(real(Recon1)
                support = self.window_stdev(np.abs(np.real(Recon1)), self.std_filter_size / 2)
                self.debug_save_mat(support, 'supportStdPy')
                support = np.where(support > self.Threshold_objsupp, 1, 0)
                self.debug_save_mat(support, 'supportThresholdPy')
                support = scipy.ndimage.binary_dilation(support, structure=skimage.morphology.disk(self.dilation_size))
                self.debug_save_mat(support, 'supportDilatePy')

                # find circles in thresholded image
                #self.hough_circle_transform(support)


                # # TODO Fix hole-filling and bwareaopen replacements
                # segmentation = scipy.ndimage.binary_fill_holes(segmentation - 1)
                # scipy.ndimage.morphology.binary_opening(support, min_size=64, connectivity=2)
                # support = skimage.morphology.remove_small_objects(support, min_size=64, connectivity=2)
            Constraint = np.ones(Recon1.shape)
            Constraint = np.where(support == 1, np.abs(Recon1), 1)
            Constraint = np.where(np.abs(Recon1) > 1, 1, Constraint)
            self.debug_save_mat(Constraint, 'ConstraintPy')
            Recon1_update = np.multiply(Constraint, np.exp(1j * np.angle(Recon1)))

            F1 = self.ft2(Recon1_update, delta1)

            Output = self.ift2(np.multiply(F1, Gfp), dfx, dfy)

            Input = np.multiply(subNormAmp, np.exp(1j * np.angle(Output)))

            print("Completing Iteration {0} of {1}  -  {2:.2f}%".format(k, self.NumIteration,
                                                                        100. * k / self.NumIteration))

        F2 = self.ft2(Input, self.delta2)
        ReconImage = self.ift2(np.multiply(F2, Gbp), dfx, dfy)
        self.debug_save_mat(ReconImage, 'ReconImagePy')
        return ReconImage

    def process(self, image_path, reference_path):
        image = np.array(scipy.misc.imread(image_path))
        ref = np.array(scipy.misc.imread(reference_path))
        norm_factor = np.mean(ref) / np.mean(image)
        data = np.divide(image, ref) * norm_factor
        return self.compute(data)

    def window_stdev(self, arr, radius):
        diameter = int(round(radius * 2))
        c1 = uniform_filter(arr, diameter, mode='constant', origin=-int(round(radius)))
        c2 = uniform_filter(arr * arr, diameter, mode='constant', origin=-int(round(radius)))
        c_std = ((c2 - c1 * c1) ** .5)[:-diameter + 1, :-diameter + 1]
        # symmetric padding to match Matlab stdfilt:
        return np.pad(c_std, int(round(radius)), 'symmetric')

    def hough_circle_transform(self, support):
        support8bit = np.array(support * 255, dtype=np.uint8)
        circles = cv2.HoughCircles(support8bit, cv2.HOUGH_GRADIENT,
                                   1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
        self.debug_save_mat(circles, 'circlesPy')

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(support8bit, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(support8bit, (i[0], i[1]), 2, (0, 0, 255), 3)
        self.debug_save_mat(support8bit, 'supportCirclesPy')
        return support8bit


# Usage
recon = Reconstruction()
# change parameters if needed
# recon.lmbda = 405e-9
# recon.delta = 2.2e-6
recon.debug = True
t1 = time.time()
result = recon.process('../test/Daudi_Kconcentrated.png', '../test/reference_image.png')
print("Reconstruction runs in: ", t1 - time.time())
scipy.misc.imsave('output.png', np.abs(result))
