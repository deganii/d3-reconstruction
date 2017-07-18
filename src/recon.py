import numpy as np
import scipy
import skimage

class Reconstruction(object):
    def __init__(self):
        super(object).__init__()
        # lambda is the wavelength in meters (i.e. 405nm = UV light)
        self.lmbda = 405e-9
        self.UpsampleFactor = 2
        self.delta = 2.2e-6
        self.Dz = 5e-4
        self.Threshold_objsupp = 0.09
        self.NumIteration = 30

    def upsampling(self, data, dx1):
        # TODO: fix interpolation
        #upsampled = scipy.interpolate.interp2d(data, self.)
        # scipy.interpolate.griddata
        dx2 = dx1/(2 ** self.UpsampleFactor)


        upsampled = data
        dx2 = 1

        return upsampled, dx2

    def ft2(self, g, delta):
        return np.fftshift(np.fft2((g))) * delta**2

    def ift2(self, G, dfx, dfy):
        Nx, Ny = np.shape(G)
        return np.ifftshift(np.ifft2(np.ifftshift(G))) * Nx*Ny*dfx*dfy;


    def compute(self, data):
        k = 2*np.pi/self.lmbda
        subNormAmp = np.sqrt(data)
        delta_prev = self.delta2

        if self.UpsampleFactor > 0:
            subNormAmp,self.delta2 = self.upsampling(subNormAmp);

        Nx, Ny = np.shape(subNormAmp)
        delta1 = self.delta2

        dfx = 1 / (Nx * self.delta2)
        dfy = 1 / (Ny * self.delta2)

        fx, fy = np.meshgrid(np.arange(-Ny / 2, Ny / 2 - 1, dfy),
                             np.arange(-Nx / 2, Nx / 2 - 1, dfx))

        Gbp = np.zeros(Nx, Ny)
        Gfp = np.zeros(Nx, Ny)


        for n in range(len(fx)):
            for m in range(len(fx[0])):
                Gbp[n, m] = np.exp(1j * k * self.Dz * np.sqrt(1 - self.lmbda ** 2 * fx[n, m] ** 2 - self.lmbda ** 2 * fy[n, m] ** 2))
                Gfp[n, m] = np.exp(-1j * k * self.Dz * np.sqrt(1 - self.lmbda ** 2 * fx[n, m] ** 2 - self.lmbda ** 2 * fy[n, m] ** 2))

        Input = subNormAmp

        for k in range(self.NumIteration):
            F2 = self.ft2(Input, self.delta2);
            Recon1 = self.ift2(np.multiply(F2, Gbp), dfx, dfy);

            if k == 0:
                # abs(Recon1).*cos(angle(Recon1) == abs(real(Recon1)
                support = scipy.ndimage.filters.generic_filter(np.abs(np.real(Recon1)), function=np.std, size=(9, 9))
                support = np.where(support > self.Threshold_objsupp, 1, 0)
                support = scipy.ndimage.binary_dilation(support, structure=skimage.morphology.disk(6))

                # TODO Fix hole-filling and bwareaopen replacements
                #segmentation = scipy.ndimage.binary_fill_holes(segmentation - 1)
                # scipy.ndimage.morphology.binary_opening(support, min_size=64, connectivity=2)
                skimage.morphology.remove_small_objects(support, min_size=64, connectivity=2)

            Constraint = np.ones(Recon1.shape);
            for p in range(Recon1.shape[0]):
                for q in range(Recon1.shape[1]):
                    if support[p, q] == 1:
                        Constraint[p, q] = np.abs(Recon1[p, q])
                    # Transmission constraint
                    if np.abs(Recon1[p, q]) > 1:
                        Constraint[p, q] = 1

            Recon1_update =np.multiply(Constraint,  np.exp(1j * np.angle(Recon1)))

            F1 = self.ft2(Recon1_update, delta1)

            Output = self.ift2(np.multiply(F1, Gfp), dfx, dfy)

            Input =np.multiply(subNormAmp, np.exp(1j * np.angle(Output)))

            if k == self.NumIteration:
                return Input



# Usage
# recon = Reconstruction()
# recon.lmbda = 405
# recon.delta = 2.2e-6
# result = recon.compute(srcImage)