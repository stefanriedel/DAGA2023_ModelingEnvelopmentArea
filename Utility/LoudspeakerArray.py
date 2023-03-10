import numpy as np


class LoudspeakerArray:
    def __init__(self, array_name):
        # array_name != 'Cube' and array_name != 'Prod':
        if array_name not in ['Cube', 'Prod']:
            raise ValueError('We do not have an array with this name!')

        self.name = array_name
        # Azimuth,Elevation coordinates of array
        self.coord = np.array([])
        if array_name == 'Prod':
            self.coord = np.array([[30, 0], [-30, 0], [0, 0], [0, -90], [150, 0], [-150, 0], [
                                  90, 0], [-90, 0], [45, 40], [-45, 40], [150, 40], [-150, 40], [0, 90]])
        if array_name == 'Cube':
            self.coord = np.array([[0, 0], [-22.5, 0], [-45, 0], [-75, 0], [-105, 0], [-135, 0], [-180, 0], [135, 0], [105, 0], [75, 0], [45, 0], [22.5, 0],
                                   [-22.5, 30], [-72.5, 30], [-112.5, 30], [-157.5,
                                                                            30], [157.5, 30], [112.5, 30], [72.5, 30], [22.5, 30],
                                   [0, 60], [-90, 60], [-180, 60], [90, 60], [0, 90]])

    def getName(self):
        return self.name

    def getCoord(self):
        return self.coord

    def getHorizontalIndices(self):
        return np.argwhere(self.coord[:, 1] == 0).flatten()

    def getHeightIndices(self):
        return np.argwhere(self.coord[:, 1] > 0).flatten()

    def getHorizontalIndicesWithoutCenter(self):
        horizontal_indices = self.getHorizontalIndices()
        center_index = self.getCenterIndex()
        return np.delete(horizontal_indices, center_index, axis=0)

    def getCenterIndex(self):
        return np.argwhere(np.all((self.coord-np.array([0, 0])) == 0, axis=1)).flatten()

    def get2LSIndices(self):
        if self.name == 'Prod':
            return np.array([0, 1])
        if self.name == 'Cube':
            return np.array([2, 10])

    def get4LSIndices(self):
        if self.name == 'Prod':
            return np.array([0, 1, 6, 7])
        if self.name == 'Cube':
            return np.array([2, 10, 5, 7])

    def downmixHeightSignals(self, Y):
        if self.name == 'Cube':
            # Voice of God
            Y[:, 20:24] += np.sqrt(1/4) * np.tile(Y[:, 24]
                                                  [:, np.newaxis], (1, 4))
            Y[:, 24] = 0.0

            # Four speakers at 60 deg. elevation
            Y[:, [12, 19]] += np.sqrt(1/2) * \
                np.tile(Y[:, 20][:, np.newaxis], (1, 2))
            Y[:, [13, 14]] += np.sqrt(1/2) * \
                np.tile(Y[:, 21][:, np.newaxis], (1, 2))
            Y[:, [15, 16]] += np.sqrt(1/2) * \
                np.tile(Y[:, 22][:, np.newaxis], (1, 2))
            Y[:, [17, 18]] += np.sqrt(1/2) * \
                np.tile(Y[:, 23][:, np.newaxis], (1, 2))
            Y[:, 20:24] = 0.0

            # Eight speakers at 30 deg. elevation
            Y[:, 2] += Y[:, 12]
            Y[:, 3] += Y[:, 13]
            Y[:, 4] += Y[:, 14]
            Y[:, 5] += Y[:, 16]
            Y[:, 12:16] = 0.0

            Y[:, 7] += Y[:, 16]
            Y[:, 8] += Y[:, 17]
            Y[:, 9] += Y[:, 18]
            Y[:, 10] += Y[:, 19]
            Y[:, 16:20] = 0.0

        if self.name == 'Prod':
            Y[:, 8:12] += np.sqrt(1/4) * np.tile(Y[:, 12]
                                                 [:, np.newaxis], (1, 4))  # VoG distribution
            Y[:, 12] = 0.0
            Y[:, 6:8] += Y[:, 8:10]
            Y[:, 4:6] += Y[:, 10:12]
            Y[:, 8:12] = 0.0

        return Y

    def getVirtualLSCoord(self):
        # Azimuth,Elevation coordinates of virtual loudspeakes for ALLRAD
        if self.name == 'Cube':
            return np.array([[0, -90], [45, 45], [-45, 45], [135, 45], [-135, 45]])
        if self.name == 'Prod':
            return np.array([[0, -90], [180, 20]])
